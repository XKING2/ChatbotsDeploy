from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Script disimpan di: Chatbot/ragas_eval/retrieve_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXCEL_DIR = PROJECT_ROOT / "data" / "Excel"
INPUT_FILE = EXCEL_DIR / "Dataset_Ragas.xlsx"
OUTPUT_FILE = EXCEL_DIR / "ragas_dataset_retrieved.xlsx"
CHECKPOINT_FILE = EXCEL_DIR / "ragas_dataset_retrieval_checkpoint.xlsx"

INPUT_SHEETS = ["Indonesia", "English", "日本語", "中文", "Русский"]
DELAY_SECONDS = 0.2
CHECKPOINT_EVERY = 5
MAX_CONTEXTS = None            # None = seluruh chunk dari pipeline
EXCEL_CELL_LIMIT = 32_000      # batas Excel 32.767 karakter per sel
RETRYABLE = {"exception", "embedding_failed", "search_error"}
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

        # Kolom reference ikut dibawa bila ada, agar tidak hilang saat dibuat ulang.
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
    return result[["row_id", "question_id", "language", "topic", "question", "reference"]]


def read_sheet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name="Dataset Evaluasi")
    except Exception as exc:
        print(f"[PERINGATAN] {path.name} tidak dapat dibaca: {exc}")
        return pd.DataFrame()


def existing_references() -> dict:
    """Reference yang sudah diisi manual pada hasil/checkpoint sebelumnya."""
    refs = {}
    for path in (CHECKPOINT_FILE, OUTPUT_FILE):
        df = read_sheet(path)
        if df.empty or "reference" not in df.columns:
            continue
        for _, r in df.iterrows():
            text = "" if pd.isna(r["reference"]) else str(r["reference"]).strip()
            if text:
                refs[(str(r["question_id"]), str(r["language"]))] = text
    return refs


# ============================================================
# SIGNATURE: menandai versi kode retrieval + database
# ============================================================

def pipeline_signature() -> str:
    """
    Hash kode retrieval dan generation database aktif. Baris checkpoint dari
    versi lain diproses ulang, jadi hasil lama dan baru tidak tercampur.
    """
    digest = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "retrival").glob("*.py")):
        digest.update(path.read_bytes())
    try:
        from database.chroma_db import get_active_generation
        digest.update(str(get_active_generation()).encode())
    except Exception:
        pass
    return digest.hexdigest()[:12]


# ============================================================
# CONTEXT
# ============================================================

def extract_contexts(retrieval_result) -> list[str]:
    """
    Chunk yang dikirim ke LLM (`relevant_chunks`), diurutkan dari skor tertinggi.
    LLM menerima urutan baca dokumen, tetapi Context Precision menilai
    peringkat retriever, jadi dataset evaluasi harus berurutan skor.
    """
    chunks = (retrieval_result or {}).get("relevant_chunks") or []
    ranked = sorted(
        (c for c in chunks if isinstance(c, dict)),
        key=lambda c: (c.get("score") is None, -(c.get("score") or 0.0)),
    )

    contexts, seen = [], set()
    for chunk in ranked:
        text = CONTROL_CHARS.sub("", str(chunk.get("content") or "")).strip()
        if text and text not in seen:
            seen.add(text)
            contexts.append(text)

    return contexts[:MAX_CONTEXTS] if MAX_CONTEXTS else contexts


def fit_excel_cell(contexts: list[str]) -> tuple[list[str], int]:
    """Buang konteks berperingkat terendah bila JSON melebihi batas sel Excel."""
    kept = list(contexts)
    while len(kept) > 1 and len(json.dumps(kept, ensure_ascii=False)) > EXCEL_CELL_LIMIT:
        kept.pop()
    return kept, len(contexts) - len(kept)


def retrieve_row(pipeline, row, signature: str) -> dict:
    base = {
        "row_id": int(row["row_id"]), "question_id": str(row["question_id"]),
        "language": str(row["language"]), "topic": str(row["topic"]),
        "question": str(row["question"]), "reference": str(row["reference"]),
        "pipeline_signature": signature,
    }
    try:
        result = pipeline.retrieve(
            question=base["question"], previous_heading=None, previous_heading_id=None)
    except Exception as exc:
        return {**base, "retrieved_contexts": "[]", "context_count": 0,
                "retrieval_found": False, "retrieval_status": "exception",
                "domain_relevant": False, "retrieval_error": f"{type(exc).__name__}: {exc}",
                "search_errors": "[]", "_contexts": []}

    context = result.get("context") or {}
    contexts, dropped = fit_excel_cell(extract_contexts(result))
    ranked = sorted(result.get("relevant_chunks") or [],
                    key=lambda c: (c.get("score") is None, -(c.get("score") or 0.0)))
    return {
        **base,
        "retrieved_contexts": json.dumps(contexts, ensure_ascii=False),
        "context_count": len(contexts),
        "retrieval_found": bool(contexts),
        "retrieval_status": result.get("status", "unknown"),
        "domain_relevant": bool(result.get("domain_relevant")),
        "resolved_heading": context.get("heading"),
        "resolved_heading_id": context.get("heading_id"),
        "resolved_subheading": context.get("subheading"),
        "resolved_subheading_id": context.get("subheading_id"),
        "resolution_reason": context.get("resolution_reason"),
        "is_followup": bool(context.get("is_followup")),
        "explicit_heading": context.get("explicit_heading"),
        "context_subheadings": json.dumps(
            [c.get("subheading") for c in ranked[:len(contexts)]], ensure_ascii=False),
        "context_scores": json.dumps(
            [round(c["score"], 4) if c.get("score") is not None else None
             for c in ranked[:len(contexts)]]),
        "contexts_dropped_for_excel": dropped,
        "retrieval_error": "",
        "search_errors": json.dumps(
            [str(e) for e in result.get("search_errors") or [] if e], ensure_ascii=False),
        "_contexts": contexts,
    }


# ============================================================
# OUTPUT
# ============================================================

def contexts_of(row) -> list[str]:
    value = row.get("_contexts")
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(row.get("retrieved_contexts", "[]")))
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def save_excel(df: pd.DataFrame, path: Path) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    path.parent.mkdir(parents=True, exist_ok=True)
    visible = df.drop(columns=["_contexts"], errors="ignore")

    per_context = []
    for _, row in df.iterrows():
        items = contexts_of(row) or [""]
        for rank, text in enumerate(items, start=1):
            per_context.append({
                "question_id": row["question_id"], "language": row["language"],
                "topic": row["topic"], "question": row["question"],
                "context_rank": rank if text else None, "retrieved_context": text,
            })

    counts = df.apply(lambda r: len(contexts_of(r)), axis=1)
    summary = (
        df.assign(_n=counts).groupby("language", dropna=False)
        .agg(jumlah_pertanyaan=("question_id", "count"),
             pertanyaan_dengan_context=("_n", lambda x: int((x > 0).sum())),
             total_context=("_n", "sum"), rata_rata_context=("_n", "mean"))
        .reset_index()
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        visible.to_excel(writer, index=False, sheet_name="Dataset Evaluasi")
        pd.DataFrame(per_context).to_excel(writer, index=False, sheet_name="Context per Baris")
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
                        help="abaikan checkpoint dan proses semua pertanyaan dari awal")
    args = parser.parse_args()

    questions = load_questions()
    signature = pipeline_signature()
    references = existing_references()
    questions["reference"] = [
        r if r else references.get((str(q), str(l)), "")
        for r, q, l in zip(questions["reference"], questions["question_id"], questions["language"])
    ]

    print("=" * 72)
    print("RETRIEVAL CONTEXT DATASET RAG - KANTOR IMIGRASI DENPASAR")
    print("=" * 72)
    print(f"Pertanyaan      : {len(questions)} | reference terisi: "
          f"{int(questions['reference'].ne('').sum())}")
    print(f"Versi pipeline  : {signature}")

    from LLM.chatbot import ImmigrationChatbot
    pipeline = ImmigrationChatbot().retrieval_pipeline

    saved = pd.DataFrame() if args.fresh else read_sheet(CHECKPOINT_FILE)
    done = set()
    if not saved.empty and {"question_id", "language", "retrieval_status"} <= set(saved.columns):
        current = (saved.get("pipeline_signature", pd.Series([""] * len(saved))) == signature)
        ok = current & ~saved["retrieval_status"].isin(RETRYABLE)
        done = set(zip(saved.loc[ok, "question_id"].astype(str), saved.loc[ok, "language"].astype(str)))
        print(f"Checkpoint      : {int(ok.sum())} baris dipakai, "
              f"{len(saved) - int(ok.sum())} diproses ulang "
              f"(versi pipeline berbeda atau error sebelumnya)")
        saved = saved[ok].copy()
    else:
        saved = pd.DataFrame()

    pending = questions[[(str(q), str(l)) not in done
                         for q, l in zip(questions["question_id"], questions["language"])]]
    print(f"Belum diproses  : {len(pending)}")
    print("-" * 72)

    results = [saved] if not saved.empty else []
    for count, (_, row) in enumerate(pending.iterrows(), start=1):
        out = retrieve_row(pipeline, row, signature)
        results.append(pd.DataFrame([out]))
        print(f"[{'OK' if out['retrieval_found'] else '--'}] {out['question_id']:<6} | "
              f"{out['language']:<10} | {out['context_count']:>2} context | {out['retrieval_status']}"
              + (f" | {out['retrieval_error']}" if out["retrieval_error"] else ""))

        if count % CHECKPOINT_EVERY == 0:
            save_excel(combine(results), CHECKPOINT_FILE)
        time.sleep(DELAY_SECONDS)

    if not results:
        raise RuntimeError("Tidak ada hasil retrieval yang dapat disimpan.")

    final = combine(results)
    save_excel(final, OUTPUT_FILE)
    save_excel(final, CHECKPOINT_FILE)

    print("\n" + "=" * 72 + "\nSELESAI\n" + "=" * 72)
    print(f"Total pertanyaan : {len(final)} | ada context: {int(final['retrieval_found'].sum())}")
    print(f"Reference kosong : {int(final['reference'].fillna('').astype(str).str.strip().eq('').sum())}")
    print(final["retrieval_status"].value_counts(dropna=False).to_string())
    print(f"File hasil       : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()