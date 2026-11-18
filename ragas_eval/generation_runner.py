from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd


# Simpan file ini di: Chatbot/ragas_eval/generate_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXCEL_DIR = PROJECT_ROOT / "data" / "Excel"
INPUT_FILE = EXCEL_DIR / "Dataset_Ragas.xlsx"
OUTPUT_FILE = EXCEL_DIR / "ragas_dataset_generation.xlsx"
CHECKPOINT_FILE = EXCEL_DIR / "ragas_dataset_generation_checkpoint.xlsx"
SHEET_NAME = "Dataset Generation"

INPUT_SHEETS = ["Indonesia", "English", "日本語", "中文", "Русский"]
DEFAULT_BATCH_SIZE = 10
DELAY_SECONDS = 0.5
CHECKPOINT_EVERY_BATCHES = 1
EXCEL_CELL_LIMIT = 32_000
BATCH_GENERATION_VERSION = "batch-no-chatbot-edit-v2"
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ============================================================
# INPUT
# ============================================================


def load_questions() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {INPUT_FILE}")

    excel = pd.ExcelFile(INPUT_FILE)
    frames: list[pd.DataFrame] = []

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
        for column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()

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
# SIGNATURE
# ============================================================


def pipeline_signature() -> str:
    """Memisahkan checkpoint batch ini dari hasil generator versi sebelumnya."""
    digest = hashlib.sha256()
    digest.update(BATCH_GENERATION_VERSION.encode("utf-8"))

    for folder in ("retrival", "LLM"):
        folder_path = PROJECT_ROOT / folder
        if not folder_path.exists():
            continue
        for path in sorted(folder_path.glob("*.py")):
            digest.update(path.read_bytes())

    try:
        from database.chroma_db import get_active_generation
        digest.update(str(get_active_generation()).encode("utf-8"))
    except Exception:
        # Signature tetap dapat dibuat bila fungsi generation database tidak tersedia.
        pass

    return digest.hexdigest()[:12]


# ============================================================
# HELPERS
# ============================================================


def clean(value: Any) -> str:
    if value is None:
        return ""
    return CONTROL_CHARS.sub("", str(value)).strip()


def flatten_contexts(values: Any) -> list[str]:
    """Menerima contexts flat maupun nested seperti format ChromaDB."""
    result: list[str] = []

    def walk(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
            return
        text = clean(value)
        if text:
            result.append(text)

    walk(values)
    return result


def fit_contexts(contexts: list[str]) -> tuple[list[str], int]:
    """Membatasi teks contexts yang disimpan ke satu sel Excel."""
    kept = list(contexts)
    while len(kept) > 1 and len(json.dumps(kept, ensure_ascii=False)) > EXCEL_CELL_LIMIT:
        kept.pop()
    return kept, len(contexts) - len(kept)


def base_row(row: pd.Series | dict, signature: str) -> dict:
    return {
        "row_id": int(row["row_id"]),
        "question_id": str(row["question_id"]),
        "language": str(row["language"]),
        "topic": str(row["topic"]),
        "question": str(row["question"]),
        "reference": str(row.get("reference", "")),
        "pipeline_signature": signature,
    }


def retrieval_error_row(row: pd.Series, signature: str, exc: Exception) -> dict:
    return {
        **base_row(row, signature),
        "retrieved_contexts": "[]",
        "response": "",
        "generation_status": f"retrieval error: {type(exc).__name__}: {exc}"[:300],
        "llm_called": False,
        "llm_request_count": 0,
        "found": False,
        "retrieval_status": "exception",
        "context_count": 0,
        "contexts_dropped_for_excel": 0,
        "prompt_chars": 0,
        "batch_prompt_chars": 0,
        "batch_size": 0,
        "generation_batch_id": "",
        "seconds_total": 0.0,
        "seconds_llm": 0.0,
        "seconds_llm_batch": 0.0,
        "shared_prompt_chars_saved": 0,
    }


# ============================================================
# PROMPT BATCHING
# ============================================================


def common_prefix_length(values: list[str]) -> int:
    if not values:
        return 0
    limit = min(map(len, values))
    index = 0
    while index < limit and all(value[index] == values[0][index] for value in values[1:]):
        index += 1
    return index


def common_suffix_length(values: list[str], prefix_length: int) -> int:
    if not values:
        return 0
    limit = min(len(value) - prefix_length for value in values)
    index = 0
    while index < limit and all(value[-(index + 1)] == values[0][-(index + 1)] for value in values[1:]):
        index += 1
    return index


def build_batch_prompt(items: list[dict]) -> tuple[str, int]:
    """Deduplicasi awalan/akhiran prompt chatbot yang sama, lalu kirim satu batch.

    Setiap prompt asli dapat direkonstruksi sebagai:
        shared_prefix + prompt_body + shared_suffix

    Instruksi asli dari build_prompt() tetap dipakai; tidak perlu mengganti
    prompt_builder.py ataupun chatbot.py.
    """
    prompts = [clean(item["prompt"]) for item in items]
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("Salah satu hasil chatbot.ask() tidak mengembalikan prompt.")

    if len(prompts) == 1:
        # Untuk satu item, simpan seluruh prompt pada body agar mudah dibaca model.
        prefix_length = 0
        suffix_length = 0
    else:
        prefix_length = common_prefix_length(prompts)
        # Potong hanya pada batas baris agar tidak memecah kata/isi pertanyaan.
        prefix_boundary = prompts[0].rfind("\n", 0, prefix_length)
        prefix_length = prefix_boundary + 1 if prefix_boundary >= 0 else 0

        suffix_length = common_suffix_length(prompts, prefix_length)
        if suffix_length:
            suffix_start = len(prompts[0]) - suffix_length
            suffix_boundary = prompts[0].find("\n", suffix_start)
            if suffix_boundary < 0:
                suffix_length = 0
            else:
                suffix_start = suffix_boundary + 1
                suffix_length = len(prompts[0]) - suffix_start

    shared_prefix = prompts[0][:prefix_length] if prefix_length else ""
    shared_suffix = prompts[0][len(prompts[0]) - suffix_length:] if suffix_length else ""

    payload = []
    original_total_chars = 0
    body_total_chars = 0

    for item, prompt in zip(items, prompts):
        body_end = len(prompt) - suffix_length if suffix_length else len(prompt)
        body = prompt[prefix_length:body_end]
        original_total_chars += len(prompt)
        body_total_chars += len(body)
        payload.append({
            "row_id": int(item["row"]["row_id"]),
            "prompt_body": body,
        })

    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    batch_prompt = f"""Anda akan memproses beberapa prompt chatbot layanan keimigrasian secara batch.

PROMPT BERSAMA (AWAL):
{shared_prefix}

ATURAN EKSEKUSI BATCH:
- Setiap item adalah satu permintaan yang berdiri sendiri.
- Prompt lengkap setiap item dibentuk dari PROMPT BERSAMA (AWAL) + prompt_body item tersebut + PROMPT BERSAMA (AKHIR).
- Terapkan instruksi dan konteks dari satu item hanya untuk item itu sendiri.
- Jangan mencampur konteks, pertanyaan, bahasa, atau jawaban antar-item.
- Ikuti seluruh instruksi pada prompt asli; jangan menambahkan informasi yang tidak didukung konteksnya.
- Kembalikan tepat satu jawaban untuk setiap row_id.

PROMPT BERSAMA (AKHIR):
{shared_suffix}

DAFTAR ITEM:
{payload_json}

FORMAT OUTPUT:
Keluarkan hanya JSON valid dengan struktur:
{{"answers":[{{"row_id":1,"answer":"jawaban sesuai prompt item"}}]}}
Jangan sertakan markdown, code fence, atau penjelasan di luar JSON.
"""

    # Estimasi karakter prompt template yang tidak perlu dikirim berulang.
    shared_chars_saved = max(0, original_total_chars - (len(shared_prefix) + len(shared_suffix) + body_total_chars))
    return batch_prompt, shared_chars_saved


def extract_json_object(raw_text: str) -> Any:
    """Membaca JSON dari respons LLM, termasuk bila dibungkus code fence."""
    text = clean(raw_text)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue

    raise ValueError("Respons LLM tidak berisi JSON valid.")


def parse_batch_answers(raw_text: str) -> dict[str, str]:
    parsed = extract_json_object(raw_text)

    if isinstance(parsed, dict):
        entries = parsed.get("answers", parsed.get("results"))
        if isinstance(entries, dict):
            return {str(key): clean(value) for key, value in entries.items()}
        if entries is None:
            if all(isinstance(value, (str, int, float)) for value in parsed.values()):
                return {str(key): clean(value) for key, value in parsed.items()}
            raise ValueError("JSON tidak memiliki field 'answers'.")
    elif isinstance(parsed, list):
        entries = parsed
    else:
        raise ValueError("Struktur JSON batch tidak dikenali.")

    answers: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        row_id = entry.get("row_id", entry.get("id"))
        answer = entry.get("answer", entry.get("response", entry.get("output", "")))
        if row_id is not None:
            answers[str(row_id)] = clean(answer)

    if not answers:
        raise ValueError("Tidak ada pasangan row_id dan answer pada respons LLM.")
    return answers


# ============================================================
# RETRIEVAL TANPA MENGUBAH CHATBOT.PY
# ============================================================


def retrieve_batch_without_generation(
    chatbot: Any,
    chatbot_module: Any,
    batch_rows: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """Panggil chatbot.ask() seperti biasa, tetapi tahan generator aslinya.

    chatbot.py tidak diedit. Fungsi generate_answer pada namespace modul
    LLM.chatbot sementara diganti dengan fungsi kosong. Dengan demikian,
    ask() tetap menjalankan language detection, context understanding,
    semantic retrieval, context formatting, dan build_prompt.
    """
    prepared: list[dict] = []
    errors: list[dict] = []

    original_generate_answer = chatbot_module.generate_answer
    original_rate_wait = chatbot.rate_limiter.wait

    def skip_generation(_prompt: str) -> str:
        return ""

    try:
        # Hentikan panggilan generator saat ask() dijalankan untuk retrieval.
        chatbot_module.generate_answer = skip_generation
        # Rate limiter hanya dibutuhkan saat request LLM yang benar-benar dikirim.
        chatbot.rate_limiter.wait = lambda: None

        for _, row in batch_rows.iterrows():
            started = time.perf_counter()
            try:
                reply = chatbot.ask(
                    question=str(row["question"]),
                    evaluation=True,
                )

                contexts = flatten_contexts(reply.get("evaluation_contexts") or [])
                contexts, dropped = fit_contexts(contexts)
                prompt = clean(reply.get("prompt"))

                if not prompt:
                    raise ValueError("chatbot.ask() tidak mengembalikan prompt evaluasi.")

                prepared.append({
                    "row": row.to_dict(),
                    "reply": reply,
                    "prompt": prompt,
                    "contexts": contexts,
                    "contexts_dropped": dropped,
                    "seconds_retrieval": round(time.perf_counter() - started, 2),
                })

                print(
                    f"[RETRIEVED] {row['question_id']:<8} | {row['language']:<10} | "
                    f"{len(contexts):>2} context | "
                    f"{reply.get('status', 'retrieval selesai')}"
                )
            except Exception as exc:
                errors.append(retrieval_error_row(row, "", exc))
                print(f"[RETRIEVAL ERROR] {row['question_id']}: {type(exc).__name__}: {exc}")
    finally:
        # WAJIB dipulihkan walaupun salah satu pertanyaan gagal.
        chatbot_module.generate_answer = original_generate_answer
        chatbot.rate_limiter.wait = original_rate_wait

    return prepared, errors


# ============================================================
# OUTPUT ROWS
# ============================================================


def make_output_row(
    item: dict,
    signature: str,
    answer: str,
    batch_id: str,
    batch_size: int,
    batch_prompt_chars: int,
    batch_seconds: float,
    request_made: bool,
    request_count: int,
    shared_prompt_chars_saved: int,
    error_message: str = "",
) -> dict:
    row = item["row"]
    reply = item["reply"]
    answer = clean(answer)

    status = "success" if answer else "error: jawaban kosong/tidak ditemukan dalam respons batch"
    if error_message:
        status = f"error: {error_message}"[:300]
    if len(answer) > EXCEL_CELL_LIMIT:
        answer = answer[:EXCEL_CELL_LIMIT]

    analysis = reply.get("analysis") or {}
    score = reply.get("semantic_score") or {}
    llm_share = batch_seconds / max(batch_size, 1)

    return {
        **base_row(row, signature),
        "retrieved_contexts": json.dumps(item["contexts"], ensure_ascii=False),
        "response": answer,
        "generation_status": status,
        "llm_called": bool(request_made),
        # Jumlah request hanya dihitung pada baris pertama batch.
        "llm_request_count": int(request_count),
        "found": bool(reply.get("found", bool(item["contexts"]))),
        "retrieval_status": reply.get("status", "unknown"),
        "context_count": len(item["contexts"]),
        "contexts_dropped_for_excel": item["contexts_dropped"],
        "resolved_heading": analysis.get("heading"),
        "resolved_heading_id": analysis.get("heading_id"),
        "resolved_subheading": analysis.get("subheading"),
        "resolution_reason": analysis.get("resolution_reason"),
        "heading_score": score.get("heading"),
        "subheading_score": score.get("subheading"),
        "prompt_chars": len(item["prompt"]),
        "batch_prompt_chars": batch_prompt_chars,
        "batch_size": batch_size,
        "generation_batch_id": batch_id,
        "shared_prompt_chars_saved": shared_prompt_chars_saved,
        "seconds_total": round(item["seconds_retrieval"] + llm_share, 2),
        "seconds_llm": round(llm_share, 2),
        "seconds_llm_batch": round(batch_seconds, 2),
    }


# ============================================================
# ONE BATCH = ONE ACTUAL LLM REQUEST
# ============================================================


def generate_batch(
    chatbot: Any,
    chatbot_module: Any,
    generate_answer_once: Callable[[str], str],
    batch_rows: pd.DataFrame,
    signature: str,
    batch_number: int,
) -> list[dict]:
    prepared, output_rows = retrieve_batch_without_generation(
        chatbot=chatbot,
        chatbot_module=chatbot_module,
        batch_rows=batch_rows,
    )

    # Retrieval error rows receive the current pipeline signature.
    for output in output_rows:
        output["pipeline_signature"] = signature

    if not prepared:
        return output_rows

    batch_id = f"{signature}-B{batch_number:05d}"
    batch_prompt, shared_chars_saved = build_batch_prompt(prepared)

    raw_response = ""
    answers: dict[str, str] = {}
    batch_seconds = 0.0
    request_made = False
    batch_error = ""

    try:
        # Rate limiter dipanggil satu kali, karena hanya ada satu request LLM.
        chatbot.rate_limiter.wait()
        llm_started = time.perf_counter()
        request_made = True
        raw_response = generate_answer_once(batch_prompt)
        batch_seconds = time.perf_counter() - llm_started
        answers = parse_batch_answers(raw_response)
    except Exception as exc:
        batch_seconds = time.perf_counter() - llm_started if "llm_started" in locals() else 0.0
        batch_error = f"{type(exc).__name__}: {exc}"
        print(f"[BATCH ERROR] {batch_id}: {batch_error}")

    expected_ids = {str(int(item["row"]["row_id"])) for item in prepared}
    unexpected_ids = set(answers) - expected_ids
    if unexpected_ids:
        print(f"[PERINGATAN] row_id tidak dikenal di output LLM: {sorted(unexpected_ids)}")

    for index, item in enumerate(prepared):
        row_id = str(int(item["row"]["row_id"]))
        answer = answers.get(row_id, "")
        row_error = batch_error
        if not row_error and not answer:
            row_error = f"jawaban untuk row_id {row_id} tidak ditemukan pada output batch"

        output = make_output_row(
            item=item,
            signature=signature,
            answer=answer,
            batch_id=batch_id,
            batch_size=len(prepared),
            batch_prompt_chars=len(batch_prompt),
            batch_seconds=batch_seconds,
            request_made=request_made,
            request_count=1 if request_made and index == 0 else 0,
            shared_prompt_chars_saved=shared_chars_saved,
            error_message=row_error,
        )
        output_rows.append(output)

        print(
            f"[{'OK' if output['generation_status'] == 'success' else '!!'}] "
            f"{item['row']['question_id']:<8} | {item['row']['language']:<10} | "
            f"{output['context_count']:>2} context | {output['generation_status'][:65]}"
        )

    return output_rows


# ============================================================
# OUTPUT EXCEL
# ============================================================


def save_excel(df: pd.DataFrame, path: Path) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    path.parent.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("language", dropna=False)
        .agg(
            jumlah_pertanyaan=("question_id", "count"),
            jawaban_sukses=("generation_status", lambda values: int((values == "success").sum())),
            baris_diproses_llm=("llm_called", lambda values: int(values.fillna(False).astype(bool).sum())),
            jumlah_request_llm=("llm_request_count", "sum"),
            retrieval_found=("found", lambda values: int(values.fillna(False).astype(bool).sum())),
            rata_rata_context=("context_count", "mean"),
        )
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
                width = max(
                    (len(str(cell.value)) for cell in column[:100] if cell.value is not None),
                    default=12,
                )
                ws.column_dimensions[column[0].column_letter].width = min(max(width + 2, 12), 60)

            for row_cells in ws.iter_rows():
                for cell in row_cells:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["question_id", "language"], keep="last")
    return combined.sort_values("row_id").reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate jawaban dataset dengan maksimal 10 pertanyaan per satu request LLM."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="abaikan checkpoint dan proses ulang semua pertanyaan",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"jumlah pertanyaan per request LLM (default: {DEFAULT_BATCH_SIZE}, maksimal 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="batasi jumlah pertanyaan untuk uji coba, misalnya --limit 10",
    )
    args = parser.parse_args()

    if not 1 <= args.batch_size <= 10:
        raise ValueError("--batch-size harus berada antara 1 sampai 10.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit harus lebih besar dari 0.")

    questions = load_questions()
    if args.limit:
        questions = questions.head(args.limit)

    signature = pipeline_signature()

    print("=" * 72)
    print("GENERATE DATASET JAWABAN CHATBOT - BATCH TANPA EDIT CHATBOT.PY")
    print("=" * 72)
    print(f"Pertanyaan       : {len(questions)}")
    print(f"Pertanyaan/batch : {args.batch_size}")
    print(f"Versi pipeline   : {signature}")

    # Import modul (bukan hanya class) agar fungsi generate_answer di namespace
    # LLM.chatbot bisa ditahan sementara selama proses retrieval.
    chatbot_module = importlib.import_module("LLM.chatbot")
    from LLM.generator import generate_answer as generate_answer_once

    chatbot = chatbot_module.ImmigrationChatbot()

    saved = pd.DataFrame() if args.fresh else read_checkpoint()
    done: set[tuple[str, str]] = set()

    if not saved.empty and {"question_id", "language", "generation_status"} <= set(saved.columns):
        current_signature = saved.get(
            "pipeline_signature", pd.Series([""] * len(saved))
        ).astype(str).eq(signature)
        successful = current_signature & saved["generation_status"].eq("success")
        done = set(
            zip(
                saved.loc[successful, "question_id"].astype(str),
                saved.loc[successful, "language"].astype(str),
            )
        )
        print(
            f"Checkpoint       : {int(successful.sum())} baris dipakai, "
            f"{len(saved) - int(successful.sum())} akan dibuat ulang"
        )
        saved = saved[successful].copy()
    else:
        saved = pd.DataFrame()

    pending = questions[
        [
            (str(question_id), str(language)) not in done
            for question_id, language in zip(questions["question_id"], questions["language"])
        ]
    ].reset_index(drop=True)

    estimated_requests = (len(pending) + args.batch_size - 1) // args.batch_size
    print(f"Belum diproses   : {len(pending)}")
    print(f"Estimasi request : {estimated_requests}")
    print("-" * 72)

    results = [saved] if not saved.empty else []

    for batch_number, start in enumerate(range(0, len(pending), args.batch_size), start=1):
        batch_rows = pending.iloc[start : start + args.batch_size]
        print(
            f"\n[BATCH {batch_number}/{estimated_requests}] "
            f"{len(batch_rows)} pertanyaan -> maksimal 1 request generation LLM"
        )

        batch_outputs = generate_batch(
            chatbot=chatbot,
            chatbot_module=chatbot_module,
            generate_answer_once=generate_answer_once,
            batch_rows=batch_rows,
            signature=signature,
            batch_number=batch_number,
        )
        if batch_outputs:
            results.append(pd.DataFrame(batch_outputs))

        if batch_number % CHECKPOINT_EVERY_BATCHES == 0 and results:
            save_excel(combine(results), CHECKPOINT_FILE)
            print(f"[CHECKPOINT] tersimpan: {CHECKPOINT_FILE}")

        if DELAY_SECONDS and batch_number < estimated_requests:
            time.sleep(DELAY_SECONDS)

    if not results:
        raise RuntimeError("Tidak ada hasil yang dapat disimpan.")

    final = combine(results)
    save_excel(final, CHECKPOINT_FILE)
    save_excel(final, OUTPUT_FILE)

    success_count = int((final["generation_status"] == "success").sum())
    request_count = int(final["llm_request_count"].fillna(0).sum())
    retrieval_found = int(final["found"].fillna(False).astype(bool).sum())

    print("\n" + "=" * 72)
    print("SELESAI")
    print("=" * 72)
    print(f"Total pertanyaan  : {len(final)}")
    print(f"Jawaban sukses    : {success_count} | gagal: {len(final) - success_count}")
    print(f"Total request LLM : {request_count}")
    print(f"Retrieval found   : {retrieval_found}")
    print(final["retrieval_status"].value_counts(dropna=False).to_string())
    print(f"File hasil        : {OUTPUT_FILE}")

    if len(final) - success_count:
        print("Ada baris gagal: jalankan ulang untuk mencoba kembali baris yang belum sukses.")
    print("Langkah berikut  : jalankan evaluate_generation_ragas.py")


if __name__ == "__main__":
    main()
