from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Script disimpan di: Chatbot/ragas_eval/generate_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXCEL_DIR = PROJECT_ROOT / "data" / "Excel"
INPUT_FILE = EXCEL_DIR / "Dataset_Ragas.xlsx"
OUTPUT_FILE = EXCEL_DIR / "ragas_dataset_generation.xlsx"       # dibaca evaluate_generation_ragas.py
CHECKPOINT_FILE = EXCEL_DIR / "ragas_dataset_generation_checkpoint.xlsx"
SHEET_NAME = "Dataset Generation"

INPUT_SHEETS = ["Indonesia", "English", "日本語", "中文", "Русский"]
DELAY_SECONDS = 0.5            # jeda tambahan; RateLimiter chatbot tetap berlaku
CHECKPOINT_EVERY = 5
LIMIT = None                   # mis. 10 untuk uji coba cepat
EXCEL_CELL_LIMIT = 32_000      # batas Excel 32.767 karakter per sel
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ============================================================
# INPUT
# ============================================================

def load_questions() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {INPUT_FILE}")

    excel = pd.ExcelFile(INPUT_FILE)
    frames = []
    for sheet in INPUT_SHEETS:
        if sheet not in excel.sheet_names:
            print(f"[PERINGATAN] Sheet '{sheet}' tidak ditemukan, dilewati.")
            continue
        df = pd.read_excel(excel, sheet_name=sheet)
        missing = {"question_id", "topic", "question"} - set(df.columns)
        if missing:
            raise ValueError(f"Sheet '{sheet}' kehilangan kolom: {sorted(missing)}")
        if "reference" not in df.columns:
            df["reference"] = ""
        df = df[["question_id", "topic", "question", "reference"]].copy()
        for col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
        df["language"] = sheet
        frames.append(df[df["question"].ne("")])

    if not frames:
        raise ValueError("Tidak ada sheet pertanyaan yang berhasil dibaca.")

    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "row_id", range(1, len(result) + 1))
    return result


def read_checkpoint() -> pd.DataFrame:
    if not CHECKPOINT_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(CHECKPOINT_FILE, sheet_name=SHEET_NAME)
    except Exception as exc:
        print(f"[PERINGATAN] Checkpoint tidak dapat dibaca: {exc}")
        return pd.DataFrame()


# ============================================================
# SIGNATURE: versi kode + database yang menghasilkan jawaban
# ============================================================

def pipeline_signature() -> str:
    """
    Hash kode retrieval, kode LLM (prompt, formatter, generator), dan generation
    database aktif. Baris checkpoint dari versi lain dibuat ulang, sehingga
    jawaban lama dan baru tidak tercampur dalam satu dataset.
    """
    digest = hashlib.sha256()
    for folder in ("retrival", "LLM"):
        for path in sorted((PROJECT_ROOT / folder).glob("*.py")):
            digest.update(path.read_bytes())
    try:
        from database.chroma_db import get_active_generation
        digest.update(str(get_active_generation()).encode())
    except Exception:
        pass
    return digest.hexdigest()[:12]


# ============================================================
# GENERATE SATU BARIS
# ============================================================

def clean(text) -> str:
    return CONTROL_CHARS.sub("", str(text or "")).strip()


def fit_contexts(contexts: list[str]) -> tuple[list[str], int]:
    """Buang konteks berperingkat terendah bila JSON melebihi batas sel Excel."""
    kept = list(contexts)
    while len(kept) > 1 and len(json.dumps(kept, ensure_ascii=False)) > EXCEL_CELL_LIMIT:
        kept.pop()
    return kept, len(contexts) - len(kept)


def generate_row(chatbot, row, signature: str) -> dict:
    base = {
        "row_id": int(row["row_id"]), "question_id": str(row["question_id"]),
        "language": str(row["language"]), "topic": str(row["topic"]),
        "question": str(row["question"]), "reference": str(row["reference"]),
        "pipeline_signature": signature,
    }
    started = time.perf_counter()
    try:
        # Setiap pertanyaan berdiri sendiri: tanpa riwayat percakapan.
        reply = chatbot.ask(base["question"], evaluation=True)
    except Exception as exc:
        return {**base, "retrieved_contexts": "[]", "response": "",
                "generation_status": f"error: {type(exc).__name__}: {exc}"[:300],
                "llm_called": False, "found": False, "retrieval_status": "exception",
                "context_count": 0}

    answer = clean(reply.get("answer"))
    contexts, dropped = fit_contexts(
        [clean(c) for c in reply.get("evaluation_contexts") or [] if clean(c)])
    # Konteks dikirim ke LLM hanya bila retrieval menemukan jawaban.
    llm_called = bool(reply.get("prompt"))
    status = "success" if answer else "error: jawaban kosong"
    if len(answer) > EXCEL_CELL_LIMIT:
        answer, status = answer[:EXCEL_CELL_LIMIT], "success"

    analysis = reply.get("analysis") or {}
    score = reply.get("semantic_score") or {}
    timings = reply.get("timings") or {}
    return {
        **base,
        "retrieved_contexts": json.dumps(contexts, ensure_ascii=False),
        "response": answer,
        "generation_status": status,
        "llm_called": llm_called,
        "found": bool(reply.get("found")),
        "retrieval_status": reply.get("status"),
        "context_count": len(contexts),
        "contexts_dropped_for_excel": dropped,
        "resolved_heading": analysis.get("heading"),
        "resolved_heading_id": analysis.get("heading_id"),
        "resolved_subheading": analysis.get("subheading"),
        "resolution_reason": analysis.get("resolution_reason"),
        "heading_score": score.get("heading"),
        "subheading_score": score.get("subheading"),
        "prompt_chars": len(reply.get("prompt") or ""),
        "seconds_total": round(time.perf_counter() - started, 2),
        "seconds_llm": round(timings.get("llm_generate", 0.0), 2),
    }


# ============================================================
# OUTPUT
# ============================================================

def save_excel(df: pd.DataFrame, path: Path) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    path.parent.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("language", dropna=False)
        .agg(jumlah_pertanyaan=("question_id", "count"),
             jawaban_sukses=("generation_status", lambda s: int((s == "success").sum())),
             llm_dipanggil=("llm_called", lambda s: int(s.fillna(False).astype(bool).sum())),
             retrieval_found=("found", lambda s: int(s.fillna(False).astype(bool).sum())),
             rata_rata_context=("context_count", "mean"))
        .reset_index()
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=SHEET_NAME)
        summary.to_excel(writer, index=False, sheet_name="Ringkasan")
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
            for column in ws.columns:
                width = max((len(str(c.value)) for c in column[:100] if c.value is not None),
                            default=12)
                ws.column_dimensions[column[0].column_letter].width = min(max(width + 2, 12), 60)
            for row in ws.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["question_id", "language"], keep="last")
    return df.sort_values("row_id").reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                        help="abaikan checkpoint dan buat semua jawaban dari awal")
    args = parser.parse_args()

    questions = load_questions()
    if LIMIT:
        questions = questions.head(LIMIT)
    signature = pipeline_signature()

    print("=" * 72)
    print("GENERATE DATASET JAWABAN CHATBOT - KANTOR IMIGRASI DENPASAR")
    print("=" * 72)
    print(f"Pertanyaan     : {len(questions)}")
    print(f"Versi pipeline : {signature}")

    from LLM.chatbot import ImmigrationChatbot
    chatbot = ImmigrationChatbot()

    saved = pd.DataFrame() if args.fresh else read_checkpoint()
    done = set()
    if not saved.empty and {"question_id", "language", "generation_status"} <= set(saved.columns):
        current = saved.get("pipeline_signature", pd.Series([""] * len(saved))) == signature
        ok = current & saved["generation_status"].eq("success")
        done = set(zip(saved.loc[ok, "question_id"].astype(str), saved.loc[ok, "language"].astype(str)))
        print(f"Checkpoint     : {int(ok.sum())} baris dipakai, {len(saved) - int(ok.sum())} dibuat ulang "
              "(versi pipeline berbeda atau error sebelumnya)")
        saved = saved[ok].copy()
    else:
        saved = pd.DataFrame()

    pending = questions[[(str(q), str(l)) not in done
                         for q, l in zip(questions["question_id"], questions["language"])]]
    print(f"Belum diproses : {len(pending)}")
    print("-" * 72)

    results = [saved] if not saved.empty else []
    for count, (_, row) in enumerate(pending.iterrows(), start=1):
        out = generate_row(chatbot, row, signature)
        results.append(pd.DataFrame([out]))
        print(f"[{'OK' if out['generation_status'] == 'success' else '!!'}] {out['question_id']:<6} | "
              f"{out['language']:<10} | {out['context_count']:>2} context | "
              f"{out['retrieval_status']} | {out['generation_status'][:60]}")

        if count % CHECKPOINT_EVERY == 0:
            save_excel(combine(results), CHECKPOINT_FILE)
        time.sleep(DELAY_SECONDS)

    if not results:
        raise RuntimeError("Tidak ada hasil yang dapat disimpan.")

    final = combine(results)
    save_excel(final, CHECKPOINT_FILE)
    save_excel(final, OUTPUT_FILE)

    ok_count = int((final["generation_status"] == "success").sum())
    print("\n" + "=" * 72 + "\nSELESAI\n" + "=" * 72)
    print(f"Total pertanyaan   : {len(final)}")
    print(f"Jawaban sukses     : {ok_count} | gagal: {len(final) - ok_count}")
    print(f"LLM dipanggil      : {int(final['llm_called'].fillna(False).astype(bool).sum())}")
    print(f"Retrieval found    : {int(final['found'].fillna(False).astype(bool).sum())}")
    print(final["retrieval_status"].value_counts(dropna=False).to_string())
    print(f"File hasil         : {OUTPUT_FILE}")
    if len(final) - ok_count:
        print("Ada baris gagal: jalankan ulang skrip ini untuk mencoba kembali baris tersebut.")
    print("Langkah berikut    : jalankan evaluate_generation_ragas.py")


if __name__ == "__main__":
    main()