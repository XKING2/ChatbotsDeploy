"""
Evaluasi Generation Chatbot Imigrasi menggunakan METRIK RESMI RAGAS
Metrik: Faithfulness dan AnswerRelevancy/ResponseRelevancy
Input: data/Excel/ragas_dataset_generation.xlsx
Output: data/Excel/Hasil_Evaluasi_Generasi_RAGAS.xlsx

Jalankan dari root proyek:
    python -m ragas_eval.generation_runner

Catatan:
- Batch berisi beberapa baris dataset untuk mengurangi overhead evaluasi.
- RAGAS tetap menjalankan tahapan evaluasi internal per metrik/per baris;
  BATCH_SIZE bukan berarti satu request Gemini untuk seluruh batch.
- Evaluator LLM mengikuti daftar base.EVALUATORS di ragas_runner.py
  (model, API key, kuota, dan failover).
- Answer Relevancy membutuhkan embedding evaluator.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

try:
    from . import ragas_runner as base
except ImportError:
    import ragas_runner as base


# ============================================================
# KONFIGURASI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "Excel" / "ragas_dataset_generation.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data" / "Excel" / "Hasil_Evaluasi_Generasi_RAGAS.xlsx"

SHEET_NAME: Any = "Dataset Generation"

BATCH_SIZE = 10
MAX_PROMPT_TOKENS = 100_000
OUTPUT_TOKEN_RESERVE = 8_000
REQUEST_TIMEOUT = 300
PAUSE_BETWEEN_BATCHES = 1.0
MAX_ATTEMPTS_PER_ROW = 3
ONLY_SUCCESSFUL_GENERATION = True
RESUME = True

# Mengikuti model embedding yang digunakan project (config.py).
# Bisa ditimpa melalui .env dengan RAGAS_EMBEDDING_MODEL.
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-002"


# ============================================================
# DATASET
# ============================================================

def load_generation_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {path}")

    raw = pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl")
    required = ["question", "retrieved_contexts", "response", "language", "topic"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df = pd.DataFrame()
    df["row_id"] = raw["row_id"] if "row_id" in raw.columns else range(1, len(raw) + 1)
    df["question_id"] = (
        raw["question_id"] if "question_id" in raw.columns else df["row_id"]
    )
    df["language"] = raw["language"].apply(base.clean_text)
    df["topic"] = raw["topic"].apply(base.clean_text)
    df["user_input"] = raw["question"].apply(base.clean_text)
    df["retrieved_contexts"] = raw["retrieved_contexts"].apply(base.parse_contexts)
    df["response"] = raw["response"].apply(base.clean_text)

    if "generation_status" in raw.columns:
        generation_status = raw["generation_status"].apply(base.clean_text).str.lower()
    else:
        generation_status = pd.Series([""] * len(raw), index=raw.index)

    invalid_mask = (
        df["user_input"].eq("")
        | df["response"].eq("")
        | df["retrieved_contexts"].map(len).eq(0)
    )
    if ONLY_SUCCESSFUL_GENERATION and "generation_status" in raw.columns:
        invalid_mask |= ~generation_status.isin(
            ["success", "sukses", "ok", "berhasil"]
        )

    invalid = df.loc[invalid_mask].copy()
    valid = df.loc[~invalid_mask].copy().reset_index(drop=True)

    if valid.empty:
        raise ValueError(
            "Tidak ada data valid. Pastikan question, response, retrieved_contexts "
            "terisi dan generation_status bernilai success."
        )

    print("\nJumlah data valid berdasarkan bahasa:")
    print(valid.groupby("language", dropna=False).size().to_string())
    print(f"\nData valid: {len(valid)} | Data dilewati: {len(invalid)}")
    return valid, invalid


# ============================================================
# RAGAS METRICS / WRAPPERS
# ============================================================

def build_ragas_components(ev):
    """
    Membangun metrik resmi RAGAS dengan integrasi native Google Gemini.
    Tidak menggunakan OpenAI maupun wrapper LangChain yang deprecated.
    """
    try:
        from google import genai
        from ragas.llms import llm_factory
        from ragas.embeddings import GoogleEmbeddings
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy
    except ImportError as exc:
        raise ImportError(
            "Integrasi Gemini modern RAGAS belum tersedia. Pastikan paket "
            "'ragas' dan 'google-genai' terpasang pada venv project."
        ) from exc

    if not ev.api_key:
        raise EnvironmentError(f"API key {ev.key_env} tidak tersedia.")

    # Client resmi Google GenAI. LLM evaluator sepenuhnya menggunakan Gemini.
    client = genai.Client(api_key=ev.api_key)

    evaluator_llm = llm_factory(
        ev.model,
        provider="google",
        client=client,
    )

    # AnswerRelevancy membutuhkan embedding untuk menghitung kemiripan.
    # Gunakan provider Google agar tidak membutuhkan OpenAI API key.
    import os
    embedding_model_name = (
        os.getenv("RAGAS_EMBEDDING_MODEL") or "gemini-embedding-002"
    )
    evaluator_embeddings = GoogleEmbeddings(
        client=client,
        model=embedding_model_name,
    )

    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    relevance_metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        strictness=1,
    )

    return [faithfulness_metric, relevance_metric], "answer_relevancy"


def run_ragas_batch(batch: list[dict], ev) -> pd.DataFrame:
    from ragas import evaluate

    metrics, relevance_name = build_ragas_components(ev)
    rows = [
        {
            "user_input": row["user_input"],
            "response": row["response"],
            "retrieved_contexts": row["retrieved_contexts"],
        }
        for row in batch
    ]
    dataset = Dataset.from_list(rows)

    try:
        from ragas.run_config import RunConfig
        run_config = RunConfig(
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
            max_workers=1,
        )
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=metrics[0].llm,
            embeddings=metrics[1].embeddings,
            run_config=run_config,
            raise_exceptions=True,
        )
    except ImportError:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=metrics[0].llm,
            embeddings=metrics[1].embeddings,
            raise_exceptions=True,
        )

    scores = result.to_pandas()
    if "faithfulness" not in scores.columns:
        raise RuntimeError(
            f"Kolom faithfulness tidak ditemukan pada hasil RAGAS: {list(scores.columns)}"
        )

    # Normalisasi nama kolom agar konsisten pada output Excel.
    if "answer_relevancy" not in scores.columns:
        if "response_relevancy" in scores.columns:
            scores = scores.rename(columns={"response_relevancy": "answer_relevancy"})
        elif relevance_name in scores.columns:
            scores = scores.rename(columns={relevance_name: "answer_relevancy"})
        else:
            raise RuntimeError(
                "Kolom Answer/Response Relevancy tidak ditemukan. "
                f"Kolom hasil: {list(scores.columns)}"
            )

    return scores[["faithfulness", "answer_relevancy"]].reset_index(drop=True)


# ============================================================
# RINGKASAN DAN EXCEL
# ============================================================

def make_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    groups = [
        ("Keseluruhan", results),
        *[(f"Bahasa: {key}", group) for key, group in results.groupby("language", dropna=False)],
        *[(f"Topik: {key}", group) for key, group in results.groupby("topic", dropna=False)],
    ]

    output = []
    for label, group in groups:
        successful = group[group["status"].eq("success")]
        faith = pd.to_numeric(successful["faithfulness"], errors="coerce").dropna()
        relevance = pd.to_numeric(
            successful["answer_relevancy"], errors="coerce"
        ).dropna()

        output.append({
            "kelompok": label,
            "jumlah_data": len(group),
            "data_berhasil_dinilai": len(successful),
            "faithfulness_persen": (
                base.ratio_to_percentage(faith.mean()) if len(faith) else pd.NA
            ),
            "answer_relevance_persen": (
                base.ratio_to_percentage(relevance.mean())
                if len(relevance) else pd.NA
            ),
        })

    return pd.DataFrame(output)


def write_output(
    path: Path,
    results: pd.DataFrame,
    invalid: pd.DataFrame,
) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = make_summary(results)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="RAGAS Results", index=False)
        summary.to_excel(writer, sheet_name="RAGAS Summary", index=False)
        if not invalid.empty:
            invalid.drop(
                columns=["retrieved_contexts"], errors="ignore"
            ).to_excel(writer, sheet_name="Skipped Rows", index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(wrap_text=True, vertical="center")

            for idx in range(1, ws.max_column + 1):
                header = str(ws.cell(1, idx).value or "")
                width = 65 if header in {
                    "user_input", "response", "retrieved_contexts"
                } else min(max(len(header) + 3, 14), 35)
                ws.column_dimensions[get_column_letter(idx)].width = width
                if header.endswith("_persen"):
                    for row_idx in range(2, ws.max_row + 1):
                        ws.cell(row_idx, idx).number_format = '0.00"%"'

            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)


# ============================================================
# PROSES UTAMA
# ============================================================

def estimate_batch_tokens(batch: list[dict]) -> int:
    text = "".join(
        row["user_input"] + row["response"] + "".join(row["retrieved_contexts"])
        for row in batch
    )
    return base.estimate_tokens(text) + OUTPUT_TOKEN_RESERVE


def main() -> None:
    load_dotenv(override=True)

    if not 1 <= BATCH_SIZE <= 20:
        raise ValueError("BATCH_SIZE harus antara 1 dan 20.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data, invalid = load_generation_dataset(INPUT_FILE)

    state = base.RateState(base.STATE_FILE)
    # Jangan gunakan nama model spekulatif seperti gemini-3.5/3.6/3.7-flash.
    # Urutan failover: Flash 2.5, Flash-Lite 2.5, lalu Gemini 3 Flash Preview.
    evaluator_configs = [
        {"model": "gemini-3-flash-preview", "key_env": "GOOGLE_API_KEY"},
        {"model": "gemini-3.5-flash", "key_env": "GOOGLE_API_KEY"},
        {"model": "gemini-3.6-flash", "key_env": "GOOGLE_API_KEY"},
    ]
    evaluators = [base.Evaluator(cfg, state) for cfg in evaluator_configs]
    for ev in evaluators:
        if not ev.api_key:
            ev.disabled = True
            print(f"[SKIP KEY] {ev.key_env} tidak tersedia untuk {ev.model}")

    if all(ev.disabled for ev in evaluators):
        raise EnvironmentError(
            "Tidak ada API key evaluator. Periksa GOOGLE_API_KEY/GEMINI_API_KEY "
            "sesuai konfigurasi ragas_runner.py."
        )

    previous = pd.DataFrame()
    if RESUME and OUTPUT_FILE.exists():
        try:
            old = pd.read_excel(
                OUTPUT_FILE, sheet_name="RAGAS Results", engine="openpyxl"
            )
            if "status" in old.columns:
                previous = old[old["status"].eq("success")].copy()
        except Exception:
            previous = pd.DataFrame()

    done_ids = set(previous["row_id"]) if not previous.empty else set()
    pending_rows = [
        row for row in data.to_dict("records")
        if row["row_id"] not in done_ids
    ]
    pending: deque = deque(pending_rows)
    attempts: dict[Any, int] = {}
    failed_rows: dict[Any, dict] = {}
    frames = [previous] if not previous.empty else []
    call_no = 0
    stopped_early = False

    print("=" * 76)
    print("EVALUASI GENERATION DENGAN METRIK RESMI RAGAS")
    print("=" * 76)
    print(f"Input          : {INPUT_FILE}")
    print(f"Output         : {OUTPUT_FILE}")
    print(f"Data valid     : {len(data)}")
    print(f"Sudah sukses   : {len(done_ids)}")
    print(f"Akan dievaluasi: {len(pending)}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Metrics        : Faithfulness + AnswerRelevancy/ResponseRelevancy")
    print(f"Models         : {' -> '.join(ev.model for ev in evaluators)}")
    print("-" * 76)

    def checkpoint() -> None:
        parts = frames + (
            [pd.DataFrame(list(failed_rows.values()))] if failed_rows else []
        )
        if not parts:
            return
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.drop_duplicates(subset=["row_id"], keep="last")
        combined = combined.sort_values("row_id").reset_index(drop=True)
        write_output(OUTPUT_FILE, combined, invalid)

    while pending:
        batch = []
        token_estimate = 0
        while pending and len(batch) < BATCH_SIZE:
            candidate = pending[0]
            candidate_cost = base.estimate_tokens(
                candidate["user_input"]
                + candidate["response"]
                + "".join(candidate["retrieved_contexts"])
            ) + 100
            if batch and token_estimate + candidate_cost > MAX_PROMPT_TOKENS:
                break
            batch.append(pending.popleft())
            token_estimate += candidate_cost

        tokens = token_estimate + OUTPUT_TOKEN_RESERVE
        ev, wait = base.choose_evaluator(evaluators, tokens)

        if ev is None:
            pending.extendleft(reversed(batch))
            stopped_early = True
            print("[STOP] Semua evaluator mencapai batas atau tidak tersedia.")
            break

        if wait > 0:
            print(f"[WAIT] {wait:.0f} detik untuk batas kuota {ev.model}")
            base._sleep(wait)

        call_no += 1
        print(
            f"\n[BATCH {call_no}] {len(batch)} data | "
            f"Evaluator: {ev.model} | estimasi {tokens:,} token"
        )

        try:
            ev.record(tokens)
            scores = run_ragas_batch(batch, ev)
            ev.errors = 0
            ev.count_daily()

            if len(scores) != len(batch):
                raise RuntimeError(
                    f"Jumlah hasil RAGAS ({len(scores)}) tidak sama dengan "
                    f"jumlah data batch ({len(batch)})."
                )

            output_rows = []
            for row, score in zip(batch, scores.to_dict("records")):
                faith = pd.to_numeric(score.get("faithfulness"), errors="coerce")
                relevance = pd.to_numeric(
                    score.get("answer_relevancy"), errors="coerce"
                )
                output_rows.append({
                    "row_id": row["row_id"],
                    "question_id": row["question_id"],
                    "language": row["language"],
                    "topic": row["topic"],
                    "question": row["user_input"],
                    "faithfulness": faith,
                    "answer_relevancy": relevance,
                    "faithfulness_persen": base.ratio_to_percentage(faith),
                    "answer_relevance_persen": base.ratio_to_percentage(relevance),
                    "evaluator_model": ev.model,
                    "status": (
                        "success"
                        if pd.notna(faith) and pd.notna(relevance)
                        else "partial: skor metrik kosong"
                    ),
                })
                failed_rows.pop(row["row_id"], None)

            frames.append(pd.DataFrame(output_rows))
            print(
                f"[OK] RAGAS mengembalikan {len(output_rows)} hasil. "
                "Skor per baris disimpan."
            )

        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            upper_error = error_text.upper()

            # 503 adalah gangguan sementara; beri jeda lebih panjang lalu retry.
            if "503" in upper_error or "UNAVAILABLE" in upper_error:
                ev.errors += 1
                ev.blocked_until = base._now() + 45
                reason = "Gemini 503 sementara; retry setelah jeda 45 detik"

            # 400 INVALID_ARGUMENT biasanya konfigurasi/model tidak kompatibel.
            # Jangan mengulang request yang sama ke model yang sama.
            elif "400" in upper_error or "INVALID_ARGUMENT" in upper_error:
                ev.errors += 1
                ev.disabled = True
                reason = "model/config tidak kompatibel (400); pindah ke evaluator berikutnya"

            else:
                reason = base.handle_error(ev, error)
            print(f"[ERROR] {reason}: {error_text[:300]}")

            retry_rows = []
            for row in batch:
                attempts[row["row_id"]] = attempts.get(row["row_id"], 0) + 1
                if attempts[row["row_id"]] < MAX_ATTEMPTS_PER_ROW:
                    retry_rows.append(row)
                else:
                    failed_rows[row["row_id"]] = {
                        "row_id": row["row_id"],
                        "question_id": row["question_id"],
                        "language": row["language"],
                        "topic": row["topic"],
                        "question": row["user_input"],
                        "faithfulness": pd.NA,
                        "answer_relevancy": pd.NA,
                        "faithfulness_persen": pd.NA,
                        "answer_relevance_persen": pd.NA,
                        "evaluator_model": ev.model,
                        "status": f"failed: {error_text}",
                    }

            if retry_rows:
                pending.extendleft(reversed(retry_rows))

        checkpoint()
        time.sleep(PAUSE_BETWEEN_BATCHES)

    checkpoint()

    all_parts = frames + (
        [pd.DataFrame(list(failed_rows.values()))] if failed_rows else []
    )
    final = (
        pd.concat(all_parts, ignore_index=True)
        if all_parts else pd.DataFrame()
    )
    if not final.empty:
        final = final.drop_duplicates(subset=["row_id"], keep="last")
        final = final.sort_values("row_id").reset_index(drop=True)
        write_output(OUTPUT_FILE, final, invalid)

    success = int(final["status"].eq("success").sum()) if not final.empty else 0
    failed = int(final["status"].astype(str).str.startswith("failed").sum()) if not final.empty else 0

    print("\n" + "=" * 76)
    print("EVALUASI GENERATION RAGAS SELESAI" if not stopped_early else "EVALUASI DIHENTIKAN")
    print("=" * 76)
    print(f"Berhasil       : {success}")
    print(f"Gagal          : {failed}")
    print(f"Belum diproses : {len(pending)}")
    print(f"Dilewati       : {len(invalid)}")
    print(f"File hasil     : {OUTPUT_FILE}")
    if success:
        successful = final[final["status"].eq("success")]
        f = pd.to_numeric(successful["faithfulness"], errors="coerce").mean()
        a = pd.to_numeric(successful["answer_relevancy"], errors="coerce").mean()
        print(f"Faithfulness       : {base.ratio_to_percentage(f):.2f}%")
        print(f"Answer Relevancy   : {base.ratio_to_percentage(a):.2f}%")


if __name__ == "__main__":
    main()
