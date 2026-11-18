from __future__ import annotations

import hashlib
import json
import os
import re
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

try:
    from . import ragas_runner as base
except ImportError:
    import ragas_runner as base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "Excel" / "ragas_dataset_generation.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data" / "Excel" / "Hasil_Evaluasi_Generasi_RAGAS.xlsx"
SHEET_NAME = "Dataset Generation"

BATCH_SIZE = 10
MAX_PROMPT_TOKENS = 40_000
OUTPUT_TOKEN_RESERVE = 8_000
MAX_ATTEMPTS_PER_ROW = 3
ONLY_SUCCESSFUL_GENERATION = True
RESUME = True

# Naikkan angka ini setiap prompt juri diubah agar hasil lama tidak dipakai ulang.
PROMPT_VERSION = "v3-multi-key-model-pool"

# Key GEMINI_API_KEY, GEMINI_API_KEY_1..N dideteksi otomatis dari environment.
# GOOGLE_API_KEY tetap didukung untuk kompatibilitas dengan konfigurasi lama.
# Prioritas model: gunakan Gemini 2.5 Flash dahulu, kemudian Gemini 3 Flash,
# lalu Gemini 3.5 Flash. Jika kuota/model pertama habis, pool berpindah.
MODEL_PRIORITY = [
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
]



# ============================================================
# DATASET
# ============================================================

def fingerprint(row: dict) -> str:
    """Sidik jari input juri. Berubah bila jawaban, konteks, atau prompt berubah."""
    payload = json.dumps(
        [row["user_input"], row["response"], row["retrieved_contexts"], PROMPT_VERSION],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


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
    df["question_id"] = raw["question_id"] if "question_id" in raw.columns else df["row_id"]
    df["language"] = raw["language"].apply(base.clean_text)
    df["topic"] = raw["topic"].apply(base.clean_text)
    df["user_input"] = raw["question"].apply(base.clean_text)
    df["retrieved_contexts"] = raw["retrieved_contexts"].apply(base.parse_contexts)
    df["response"] = raw["response"].apply(base.clean_text)

    invalid_mask = df["user_input"].eq("") | df["response"].eq("")
    if ONLY_SUCCESSFUL_GENERATION and "generation_status" in raw.columns:
        status = raw["generation_status"].apply(base.clean_text).str.lower()
        invalid_mask |= ~status.isin(["success", "sukses", "ok", "berhasil", ""])

    invalid = df.loc[invalid_mask].drop(columns=["retrieved_contexts"]).copy()
    valid = df.loc[~invalid_mask].copy().reset_index(drop=True)
    if valid.empty:
        raise ValueError("Tidak ada baris valid untuk dievaluasi.")

    no_context = int(valid["retrieved_contexts"].map(len).eq(0).sum())
    print("\nJumlah data berdasarkan bahasa:")
    print(valid.groupby("language", dropna=False).size().to_string())
    print(f"\nData valid: {len(valid)} | Dilewati: {len(invalid)} | Tanpa konteks: {no_context}")
    return valid, invalid


# ============================================================
# JUDGE
# ============================================================

JUDGE_INSTRUCTIONS = """You are a strict, impartial multilingual evaluator.

Evaluate each ITEM independently. Use only its QUESTION, CONTEXTS, and ANSWER.
The question, contexts, and answer may be in different languages (Indonesian,
English, Mandarin, Japanese, Russian). Judge by meaning, not wording.

Each context starts with metadata lines (Dokumen, Kategori, Heading, Subheading)
followed by the text after "Informasi:". Metadata alone is never evidence.

For each ITEM:
1. Split the ANSWER into atomic informational claims. A claim is one distinct
   factual statement, requirement, procedure step, number, condition, or explanation.
   Do not count greetings, opening or closing courtesy sentences, apologies,
   formatting, or statements merely saying that information is unavailable.
2. For EVERY claim, evaluate two independent binary properties:
   - supported: true only if the claim is stated in or reasonably inferable from
     the CONTEXTS. Numbers, amounts of money, durations, counts, document names,
     and legal references must match the CONTEXTS exactly (after translating
     units). False if unsupported, only partly supported, or contradicted.
   - relevant: true only if the claim directly helps answer the QUESTION.
     A true and context-supported fact can still be irrelevant to the QUESTION.
     Do not use context support as a reason to mark relevance true.
3. evidence: for a supported claim, the 1-based index of the context that best
   supports it; otherwise null.
4. Do not omit claims merely because they are repetitive, inconvenient, or hard
   to judge. Keep each claim short and quote it in the ANSWER language.
5. Do not infer relevance from topical similarity alone.
6. If there are no informational claims, return an empty claims array.

Return ONLY valid JSON:
{"results":[{"id":1,"claims":[{"claim":"short claim quote","supported":true,"relevant":true,"evidence":2}]}]}
Include exactly one result for every ITEM id. Each claim must have claim,
supported, relevant (JSON booleans, not strings) and evidence (integer or null).
"""


def build_judge_prompt(batch: list[dict]) -> str:
    blocks = []
    for local_id, row in enumerate(batch, start=1):
        contexts = "\n".join(
            f"[{i}] {ctx}" for i, ctx in enumerate(row["retrieved_contexts"], 1)
        ) or "(no context retrieved)"
        blocks.append(
            f"=== ITEM {local_id} ===\n"
            f"LANGUAGE: {row['language']}\n"
            f"QUESTION:\n{row['user_input']}\n"
            f"CONTEXTS:\n{contexts}\n"
            f"ANSWER:\n{row['response']}\n"
        )
    return JUDGE_INSTRUCTIONS + "\n" + "\n".join(blocks)


def _extract_json(text: str):
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", (text or "").strip(),
                     flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Hanya JSON object lengkap; potongan parsial tidak diterima.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def parse_judge_response(text: str, n_contexts: list[int]) -> dict[int, dict]:
    """n_contexts[i] = jumlah konteks pada item ke-(i+1), untuk validasi `evidence`."""
    data = _extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return {}

    parsed: dict[int, dict] = {}
    for entry in data["results"]:
        if not isinstance(entry, dict):
            continue
        local_id, claims = entry.get("id"), entry.get("claims")
        if isinstance(local_id, bool) or not isinstance(local_id, int):
            continue
        if not 1 <= local_id <= len(n_contexts) or local_id in parsed:
            continue
        if not isinstance(claims, list):
            continue

        valid_claims, malformed = [], False
        for claim in claims:
            if not isinstance(claim, dict):
                malformed = True
                break
            quote = claim.get("claim")
            supported, relevant = claim.get("supported"), claim.get("relevant")
            if (not isinstance(quote, str) or not quote.strip()
                    or not isinstance(supported, bool) or not isinstance(relevant, bool)):
                malformed = True
                break
            evidence = claim.get("evidence")
            if (isinstance(evidence, bool) or not isinstance(evidence, int)
                    or not 1 <= evidence <= n_contexts[local_id - 1] or not supported):
                evidence = None
            valid_claims.append({"claim": quote.strip(), "supported": supported,
                                 "relevant": relevant, "evidence": evidence})
        if malformed:
            continue

        total = len(valid_claims)
        faithfulness = (sum(c["supported"] for c in valid_claims) / total) if total else pd.NA
        relevance = (sum(c["relevant"] for c in valid_claims) / total) if total else pd.NA
        parsed[local_id] = {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "total_claims": total,
            "supported_claims": sum(c["supported"] for c in valid_claims),
            "relevant_claims": sum(c["relevant"] for c in valid_claims),
            "claims_detail": json.dumps(valid_claims, ensure_ascii=False),
            "unsupported_claims": json.dumps(
                [c["claim"] for c in valid_claims if not c["supported"]], ensure_ascii=False),
            "irrelevant_claims": json.dumps(
                [c["claim"] for c in valid_claims if not c["relevant"]], ensure_ascii=False),
        }
    return parsed


# ============================================================
# RINGKASAN
# ============================================================

def make_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Rata-rata per jawaban (makro) dan per klaim (mikro). Jawaban tanpa klaim
    (mis. pesan di luar domain) tidak masuk rata-rata tetapi dihitung terpisah.
    """
    if results.empty:
        return pd.DataFrame()
    groups = [
        ("Keseluruhan", results),
        *[(f"Bahasa: {k}", g) for k, g in results.groupby("language", dropna=False)],
        *[(f"Topik: {k}", g) for k, g in results.groupby("topic", dropna=False)],
    ]
    if "evaluator_model" in results.columns:
        groups.extend(
            (f"Model evaluator: {model}", group)
            for model, group in results.groupby("evaluator_model", dropna=False)
        )
    rows = []
    for label, group in groups:
        ok = group[group["status"].eq("success")]
        with_claims = ok[pd.to_numeric(ok["total_claims"], errors="coerce").fillna(0) > 0]
        faith = pd.to_numeric(with_claims["faithfulness"], errors="coerce").dropna()
        relevance = pd.to_numeric(with_claims["answer_relevance"], errors="coerce").dropna()
        claims = pd.to_numeric(with_claims["total_claims"], errors="coerce").sum()
        supported = pd.to_numeric(with_claims["supported_claims"], errors="coerce").sum()
        relevant = pd.to_numeric(with_claims["relevant_claims"], errors="coerce").sum()
        rows.append({
            "kelompok": label,
            "jumlah_data": len(group),
            "data_berhasil_dinilai": len(ok),
            "data_tanpa_klaim": len(ok) - len(with_claims),
            "faithfulness_persen": base.ratio_to_percentage(faith.mean()) if len(faith) else pd.NA,
            "answer_relevance_persen": base.ratio_to_percentage(relevance.mean()) if len(relevance) else pd.NA,
            "total_klaim": int(claims),
            "faithfulness_per_klaim_persen": base.ratio_to_percentage(supported / claims) if claims else pd.NA,
            "answer_relevance_per_klaim_persen": base.ratio_to_percentage(relevant / claims) if claims else pd.NA,
        })
    return pd.DataFrame(rows)


# ============================================================
# BATCH & MAIN
# ============================================================

def take_batch(pending: deque) -> list[dict]:
    batch, tokens = [], 0
    while pending and len(batch) < BATCH_SIZE:
        row = pending[0]
        cost = base.estimate_tokens(
            row["user_input"] + row["response"] + "".join(row["retrieved_contexts"])
        ) + 100
        if batch and tokens + cost > MAX_PROMPT_TOKENS:
            break
        batch.append(pending.popleft())
        tokens += cost
    return batch


def load_previous(data: pd.DataFrame, allowed_models: set[str]) -> pd.DataFrame:
    """Pakai hasil sukses yang inputnya sama dan modelnya masih ada di pool."""
    if not (RESUME and OUTPUT_FILE.exists()):
        return pd.DataFrame()
    try:
        old = pd.read_excel(OUTPUT_FILE, sheet_name="RAGAS Results", engine="openpyxl")
    except Exception:
        return pd.DataFrame()
    required = {"input_fingerprint", "status", "evaluator_model", "row_id"}
    if not required.issubset(old.columns):
        print("Hasil lama tidak memiliki kolom resume yang lengkap; seluruh data dinilai ulang.")
        return pd.DataFrame()

    current = {r["row_id"]: fingerprint(r) for r in data.to_dict("records")}
    keep = (
        old["status"].eq("success")
        & old["evaluator_model"].astype(str).isin(allowed_models)
        & old["row_id"].map(current).eq(old["input_fingerprint"])
    )
    stale = int(old["status"].eq("success").sum() - keep.sum())
    if stale:
        print(f"{stale} hasil lama tidak berlaku (input/prompt/model berbeda); dinilai ulang.")
    return old[keep].copy()


def discover_api_key_envs() -> list[str]:
    """Deteksi GEMINI_API_KEY, GEMINI_API_KEY_1..N, dan GOOGLE_API_KEY."""
    names = ["GEMINI_API_KEY"]
    numbered = [
        name for name in os.environ
        if re.fullmatch(r"GEMINI_API_KEY_\d+", name)
    ]
    numbered.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    names.extend(numbered)
    names.append("GOOGLE_API_KEY")  # kompatibilitas dengan .env lama

    # Jaga urutan dan ambil hanya env yang benar-benar berisi key.
    return list(dict.fromkeys(
        name for name in names if os.getenv(name, "").strip()
    ))


def build_evaluator_pool(state):
    """Buat kombinasi semua API key aktif x semua model prioritas; deduplikasi key yang sama."""
    active_key_envs = discover_api_key_envs()
    if not active_key_envs:
        raise EnvironmentError(
            "Tidak ada API key aktif. Isi GEMINI_API_KEY, GEMINI_API_KEY_1, "
            "... GEMINI_API_KEY_5 di file .env."
        )

    configs = [
        {"model": model, "key_env": key_env}
        for model in MODEL_PRIORITY
        for key_env in active_key_envs
    ]

    evaluators = []
    seen = set()
    for cfg in configs:
        ev = base.Evaluator(cfg, state)
        # Evaluator.name menggunakan fingerprint secret + model pada ragas_runner.
        # Jika alias .env berisi API key yang sama, kombinasi itu cukup dibuat sekali.
        if ev.name in seen:
            continue
        seen.add(ev.name)
        if not ev.api_key:
            ev.disabled = True
        evaluators.append(ev)
    return evaluators


def choose_pool_evaluator(evaluators, tokens: int, key_cursor: dict[str, int]):
    """Pilih model sesuai prioritas dan rotasikan key untuk model yang sama.

    Model selanjutnya hanya digunakan ketika tidak ada key yang dapat dipakai pada
    model dengan prioritas lebih tinggi (kuota habis, key invalid, atau menunggu lama).
    """
    best_waiting = None

    for model in MODEL_PRIORITY:
        model_pool = [ev for ev in evaluators if ev.model == model]
        if not model_pool:
            continue

        start = key_cursor.get(model, 0) % len(model_pool)
        ordered = model_pool[start:] + model_pool[:start]
        available = [ev for ev in ordered if ev.usable(tokens)]
        if not available:
            continue

        # Coba key berikutnya dalam rotasi yang tidak sedang terkena rate limit lama.
        for ev in available:
            wait = ev.wait_needed(tokens)
            if wait <= getattr(base, "MAX_WAIT_SECONDS", 90):
                selected_index = model_pool.index(ev)
                key_cursor[model] = (selected_index + 1) % len(model_pool)
                return ev, wait
            if best_waiting is None or wait < best_waiting[1]:
                best_waiting = (ev, wait)

    # Jika semua kandidat hanya sedang menunggu sementara, tunggu kandidat tercepat.
    if best_waiting is not None:
        ev, wait = best_waiting
        model_pool = [item for item in evaluators if item.model == ev.model]
        if model_pool:
            key_cursor[ev.model] = (model_pool.index(ev) + 1) % len(model_pool)
        return ev, wait

    return None, 0.0


def main() -> None:
    load_dotenv(override=False)
    data, invalid = load_generation_dataset(INPUT_FILE)
    state = base.RateState(base.STATE_FILE)
    evaluators = build_evaluator_pool(state)

    print("\nPool evaluator aktif:")
    base.print_evaluator_status(evaluators)
    if not any(not ev.disabled for ev in evaluators):
        raise EnvironmentError("Tidak ada API key Gemini yang bisa digunakan.")

    allowed_models = set(MODEL_PRIORITY)
    previous = load_previous(data, allowed_models)
    done_ids = set(previous["row_id"]) if not previous.empty else set()
    pending = deque(r for r in data.to_dict("records") if r["row_id"] not in done_ids)
    frames = [previous] if not previous.empty else []
    attempts: dict[Any, int] = {}
    failed_rows: dict[Any, dict] = {}
    call_no = 0
    key_cursor = {model: 0 for model in MODEL_PRIORITY}

    calls_needed = -(-len(pending) // BATCH_SIZE)
    active_key_count = len({ev.name.split(":", 1)[0] for ev in evaluators})
    print(
        f"Sudah selesai: {len(done_ids)} | Akan dievaluasi: {len(pending)} "
        f"| estimasi request minimal: {calls_needed} "
        f"| API key unik terdeteksi: {active_key_count} "
        f"| model: {', '.join(MODEL_PRIORITY)}"
    )
    print(
        "Catatan kuota: beberapa API key dari Google Cloud project yang sama "
        "dapat berbagi kuota project; rotasi key tidak otomatis menambah kuota."
    )

    def checkpoint():
        parts = frames + ([pd.DataFrame(list(failed_rows.values()))] if failed_rows else [])
        if parts:
            result = pd.concat(parts, ignore_index=True)
            result = result.sort_values("row_id").drop_duplicates("row_id", keep="last")
            base.write_results(OUTPUT_FILE, result, invalid, make_summary(result))

    while pending:
        batch = take_batch(pending)
        prompt = build_judge_prompt(batch)
        tokens = base.estimate_tokens(prompt) + OUTPUT_TOKEN_RESERVE
        ev, wait = choose_pool_evaluator(evaluators, tokens, key_cursor)
        if ev is None:
            pending.extendleft(reversed(batch))
            print("Kuota/model evaluator tidak tersedia. Proses dihentikan sementara.")
            break
        if wait > 0:
            base._sleep(wait)

        call_no += 1
        print(f"[Batch {call_no}] {ev.model} | {ev.key_env} | {len(batch)} data")
        try:
            ev.record(tokens)
            response_text = base.call_llm(ev, prompt)
            ev.errors = 0
            ev.count_daily()
        except Exception as error:
            reason = base.handle_error(ev, error)
            print(f"Gagal memanggil {ev.model}: {reason}")
            if "kuota harian habis" in reason.lower():
                print(f"[FAILOVER] {ev.model} ditandai habis; batch akan dicoba ke model cadangan.")
            pending.extendleft(reversed(batch))

            # Jika semua model telah habis/dinonaktifkan, simpan progres lalu berhenti.
            if not any(candidate.usable(1) for candidate in evaluators):
                print("[STOP] Semua model cadangan tidak tersedia atau kuotanya habis.")
                break
            continue

        parsed = parse_judge_response(response_text, [len(r["retrieved_contexts"]) for r in batch])
        if len(parsed) < len(batch):
            debug_path = OUTPUT_FILE.with_name("generation_raw_response_debug.txt")
            debug_path.write_text(
                f"model={ev.model}\nfinish_reason={base.LAST_CALL.get('finish_reason')}\n"
                f"parsed={len(parsed)}/{len(batch)}\n\n{response_text}", encoding="utf-8")
            print(f"Respons juri tidak lengkap ({len(parsed)}/{len(batch)}). Disimpan: {debug_path}")

        ok_rows, retry = [], []
        for local_id, row in enumerate(batch, start=1):
            head = {
                "row_id": row["row_id"], "question_id": row["question_id"],
                "language": row["language"], "topic": row["topic"],
                "input_fingerprint": fingerprint(row),
            }
            if local_id in parsed:
                item = parsed[local_id]
                ok_rows.append({
                    **head, **item,
                    "faithfulness_persen": base.ratio_to_percentage(item["faithfulness"]),
                    "answer_relevance_persen": base.ratio_to_percentage(item["answer_relevance"]),
                    "batch": call_no, "evaluator_model": ev.model,
                    "evaluator_key_env": ev.key_env, "status": "success",
                })
                failed_rows.pop(row["row_id"], None)
            else:
                attempts[row["row_id"]] = attempts.get(row["row_id"], 0) + 1
                if attempts[row["row_id"]] < MAX_ATTEMPTS_PER_ROW:
                    retry.append(row)
                else:
                    failed_rows[row["row_id"]] = {
                        **head, "faithfulness": pd.NA, "answer_relevance": pd.NA,
                        "faithfulness_persen": pd.NA, "answer_relevance_persen": pd.NA,
                        "batch": call_no, "evaluator_model": ev.model,
                        "evaluator_key_env": ev.key_env,
                        "status": "failed: respons juri tidak valid",
                    }
        if ok_rows:
            frames.append(pd.DataFrame(ok_rows))
        pending.extendleft(reversed(retry))
        checkpoint()
        print(f"Berhasil {len(ok_rows)}/{len(batch)} | diulang {len(retry)}")

    checkpoint()
    final = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not final.empty:
        final = final.drop_duplicates("row_id", keep="last")
        summaries = make_summary(final)
        summary = summaries.iloc[0]
        print(f"\nFaithfulness gabungan: {summary['faithfulness_persen']:.2f}% "
              f"(per klaim {summary['faithfulness_per_klaim_persen']:.2f}%)")
        print(f"Answer Relevance gabungan: {summary['answer_relevance_persen']:.2f}% "
              f"(per klaim {summary['answer_relevance_per_klaim_persen']:.2f}%)")
        print(f"Jawaban tanpa klaim (tidak masuk rata-rata): {int(summary['data_tanpa_klaim'])}")
        model_rows = summaries[summaries["kelompok"].astype(str).str.startswith("Model evaluator:")]
        if not model_rows.empty:
            print("\nRingkasan per model evaluator:")
            print(model_rows[["kelompok", "jumlah_data", "faithfulness_persen", "answer_relevance_persen"]].to_string(index=False))
    print(f"Output: {OUTPUT_FILE}")
    print(f"Berhasil: {len(final)} | Gagal: {len(failed_rows)} | Belum diproses: {len(pending)}")


if __name__ == "__main__":
    main()