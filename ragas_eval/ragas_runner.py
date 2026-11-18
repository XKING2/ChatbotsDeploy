from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = PROJECT_ROOT / "data" / "Excel" / "ragas_rate_state.json"

# Gunakan satu model untuk seluruh eksperimen. Ganti hanya jika seluruh dataset
# akan dievaluasi ulang dengan model pengganti yang sama.
DEFAULT_LIMITS = {"rpm": 5, "rpd": 20, "tpm": 250_000}
# SATU model juri untuk seluruh dataset. Beberapa API key (kuota berbeda) boleh
# dipakai bergantian, tetapi modelnya harus sama agar skor sebanding.
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gemini-3-flash-preview")
JUDGE_KEY_ENVS = [
    name.strip()
    for name in os.getenv("RAGAS_JUDGE_KEY_ENVS", "GOOGLE_API_KEY").split(",")
    if name.strip()
]
EVALUATORS = [{"model": JUDGE_MODEL, "key_env": env} for env in JUDGE_KEY_ENVS]
MAX_WAIT_SECONDS = 90
REQUEST_TIMEOUT = 300
_now = time.time
_sleep = time.sleep
_clients: dict[str, Any] = {}
LAST_CALL: dict[str, Any] = {}


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def parse_contexts(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            try:
                items = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                items = [raw]
    else:
        items = [value]
    if not isinstance(items, (list, tuple)):
        items = [items]
    return [str(x).strip() for x in items if x is not None and str(x).strip()]


def ratio_to_percentage(value: Any):
    score = pd.to_numeric(value, errors="coerce")
    return pd.NA if pd.isna(score) else float(score) * 100.0


def estimate_tokens(text: str) -> int:
    # Estimasi konservatif; bukan tokenizer resmi Gemini.
    non_ascii = sum(ord(c) > 127 for c in text)
    return max(1, int((len(text) - non_ascii) / 3.5 + non_ascii * 1.1) + 1)


def _quota_day() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d")


class RateState:
    def __init__(self, path: Path):
        self.path = path
        self.day = _quota_day()
        self.used: dict[str, int] = {}
        self.exhausted: dict[str, str] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("day") == self.day:
                self.used = dict(data.get("used", {}))
                saved = data.get("exhausted", {})
                self.exhausted = dict(saved) if isinstance(saved, dict) else {}
        except (OSError, ValueError, TypeError):
            pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "day": self.day, "used": self.used, "exhausted": self.exhausted
        }, indent=2), encoding="utf-8")


class Evaluator:
    def __init__(self, cfg: dict, state: RateState):
        self.model = cfg["model"]
        self.key_env = cfg.get("key_env", "GOOGLE_API_KEY")
        limits = {**DEFAULT_LIMITS, **{k: cfg[k] for k in DEFAULT_LIMITS if k in cfg}}
        self.rpm, self.rpd, self.tpm = limits["rpm"], limits["rpd"], limits["tpm"]
        self.state = state
        key = self.api_key
        fingerprint = hashlib.sha256(key.encode()).hexdigest()[:8] if key else self.key_env
        self.name = f"{fingerprint}:{self.model}"
        self.window: deque[tuple[float, int]] = deque()
        self.blocked_until = 0.0
        self.disabled = not bool(key)
        self.errors = 0

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.key_env) or (
            os.getenv("GEMINI_API_KEY") if self.key_env == "GOOGLE_API_KEY" else None
        )

    def usable(self, tokens: int) -> bool:
        return (not self.disabled and self.name not in self.state.exhausted
                and self.state.used.get(self.name, 0) < self.rpd and tokens <= self.tpm)

    def wait_needed(self, tokens: int) -> float:
        now = _now()
        while self.window and self.window[0][0] <= now - 60:
            self.window.popleft()
        wait = max(0.0, self.blocked_until - now)
        if len(self.window) >= self.rpm:
            wait = max(wait, self.window[0][0] + 60 - now)
        total = sum(t for _, t in self.window)
        if total + tokens > self.tpm:
            running = total
            for stamp, used in self.window:
                running -= used
                if running + tokens <= self.tpm:
                    wait = max(wait, stamp + 60 - now)
                    break
        return wait + (1.0 if wait else 0.0)

    def record(self, tokens: int) -> None:
        self.window.append((_now(), tokens))

    def count_daily(self) -> None:
        self.state.used[self.name] = self.state.used.get(self.name, 0) + 1
        self.state.save()

    def status_text(self) -> str:
        if self.disabled:
            return f"tidak aktif (API key {self.key_env} kosong / dinonaktifkan)"
        if self.name in self.state.exhausted:
            return f"ditandai habis harian: {self.state.exhausted[self.name]}"
        return f"siap ({self.state.used.get(self.name, 0)}/{self.rpd})"


def build_evaluators(configs: list[dict], state: RateState) -> list[Evaluator]:
    return [Evaluator(cfg, state) for cfg in configs]


def print_evaluator_status(evaluators: list[Evaluator]) -> None:
    print("Status evaluator:")
    for ev in evaluators:
        print(f"  {ev.model} [{ev.key_env}] -> {ev.status_text()}")


def choose_evaluator(evaluators: list[Evaluator], tokens: int):
    for ev in evaluators:
        if ev.usable(tokens):
            wait = ev.wait_needed(tokens)
            if wait <= MAX_WAIT_SECONDS:
                return ev, wait
    candidates = [ev for ev in evaluators if ev.usable(tokens)]
    if candidates:
        ev = min(candidates, key=lambda x: x.wait_needed(tokens))
        return ev, ev.wait_needed(tokens)
    return None, 0.0


def call_llm(ev: Evaluator, prompt: str) -> str:
    from google import genai
    from google.genai import types

    key = ev.api_key
    if key not in _clients:
        _clients[key] = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),
        )
    response = _clients[key].models.generate_content(
        model=ev.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            max_output_tokens=40_000,
        ),
    )
    try:
        LAST_CALL["finish_reason"] = str(response.candidates[0].finish_reason)
    except (AttributeError, IndexError, TypeError):
        LAST_CALL["finish_reason"] = "unknown"
    return response.text or ""


def handle_error(ev: Evaluator, error: Exception) -> str:
    message = str(error)
    upper = message.upper()
    ev.errors += 1
    if any(token in upper for token in ("429", "RESOURCE_EXHAUSTED", "QUOTA")):
        if "PERDAY" in upper or "PER_DAY" in upper or "PER DAY" in upper:
            ev.state.exhausted[ev.name] = message[:300]
            ev.state.save()
            return "kuota harian habis"
        ev.blocked_until = _now() + 65
        return "rate limit sementara; tunggu sekitar 65 detik"
    if any(token in upper for token in ("401", "403", "404", "NOT_FOUND", "PERMISSION_DENIED")):
        ev.disabled = True
        return "model atau API key tidak dapat digunakan"
    if ev.errors >= 5:
        ev.disabled = True
        return "dinonaktifkan setelah 5 error beruntun"
    ev.blocked_until = _now() + 20
    return f"error sementara: {type(error).__name__}"


def write_results(output_path, results: pd.DataFrame, invalid: pd.DataFrame,
                  summary: pd.DataFrame | None = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="RAGAS Results", index=False)
        if summary is not None:
            summary.to_excel(writer, sheet_name="RAGAS Summary", index=False)
        if not invalid.empty:
            invalid.drop(columns=["retrieved_contexts"], errors="ignore").to_excel(
                writer, sheet_name="Skipped Rows", index=False
            )
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            for idx in range(1, ws.max_column + 1):
                header = str(ws.cell(1, idx).value or "")
                ws.column_dimensions[get_column_letter(idx)].width = (
                    28 if "context" in header else min(max(len(header) + 3, 14), 42)
                )
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)


# ============================================================
# RETRIEVAL EVALUATION: Context Precision & Context Recall
# ============================================================

RETRIEVAL_INPUT_FILE = PROJECT_ROOT / "data" / "Excel" / "ragas_dataset_retrieved.xlsx"
RETRIEVAL_OUTPUT_FILE = PROJECT_ROOT / "data" / "Excel" / "Hasil_Evaluasi_Ragas.xlsx"
RETRIEVAL_SHEET_NAME = "Dataset Evaluasi"
RETRIEVAL_BATCH_SIZE = 5          # batch kecil: juri lebih teliti per item
RETRIEVAL_MAX_PROMPT_TOKENS = 40_000
RETRIEVAL_OUTPUT_TOKEN_RESERVE = 6_000


def split_sentences(text: str) -> list[str]:
    """
    Pecah reference per kalimat. Tanda akhir kalimat CJK (。！？) tidak diikuti
    spasi, sehingga dipecah tanpa syarat spasi; tanda latin (.!?) hanya
    dipecah bila diikuti spasi agar angka seperti Rp1.000.000 tidak terpotong.
    """
    parts = re.split(r"(?<=[。！？])|(?<=[.!?])\s+|\n+", str(text or "").strip())
    sentences = [part.strip() for part in parts if part and part.strip()]
    return sentences or ([str(text).strip()] if str(text or "").strip() else [])


def context_precision(flags: list[int]) -> float:
    """Average precision pada urutan konteks yang diterima dari retriever."""
    hits = 0
    precision_sum = 0.0
    for rank, relevant in enumerate(flags, start=1):
        if relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / hits if hits else 0.0


def context_recall(attributed: set[int], n_sentences: int) -> float:
    return len(attributed) / n_sentences if n_sentences else 0.0


def load_retrieval_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {path}")
    raw = pd.read_excel(path, sheet_name=RETRIEVAL_SHEET_NAME, engine="openpyxl")
    required = ["question", "retrieved_contexts", "reference", "language", "topic"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df = pd.DataFrame()
    df["row_id"] = raw["row_id"] if "row_id" in raw.columns else range(1, len(raw) + 1)
    df["question_id"] = raw["question_id"] if "question_id" in raw.columns else df["row_id"]
    df["language"] = raw["language"].apply(clean_text)
    df["topic"] = raw["topic"].apply(clean_text)
    df["user_input"] = raw["question"].apply(clean_text)
    df["retrieved_contexts"] = raw["retrieved_contexts"].apply(parse_contexts)
    df["reference"] = raw["reference"].apply(clean_text)

    invalid_mask = (
        df["user_input"].eq("") | df["reference"].eq("")
        | df["retrieved_contexts"].map(len).eq(0)
    )
    invalid = df.loc[invalid_mask].copy()
    valid = df.loc[~invalid_mask].copy().reset_index(drop=True)
    if valid.empty:
        raise ValueError("Tidak ada baris retrieval valid.")
    return valid, invalid


RETRIEVAL_JUDGE_PROMPT = """You are a strict, impartial multilingual retrieval evaluator.
Evaluate each ITEM independently using only its QUESTION, REFERENCE SENTENCES,
and ordered CONTEXTS. Questions, references and contexts may be in different
languages; judge by meaning, not by wording.

Each context begins with metadata lines (Dokumen, Kategori, Heading,
Subheading) followed by the text after "Informasi:". Metadata alone is never
evidence; only the information text counts.

For each item:
- relevant_contexts: list the 1-based context indices that contain at least one
  fact the reference answer relies on, even if the context also contains other
  material. Exclude contexts that are merely on the same topic, unrelated, or
  that repeat nothing the reference needs. Judge each context independently.
- attributed_sentences: list the 1-based reference sentence indices whose
  information is explicitly supported or reasonably inferable from the union
  of the contexts. A sentence is supported only if the contexts contain enough
  information for its substantive meaning. Do not use outside knowledge.
- Do not mark a context relevant merely because it mentions the same entity.
- reasoning: one short sentence per item, written BEFORE the lists, naming
  which context facts match which reference sentences.
Return ONLY JSON:
{"results":[{"id":1,"reasoning":"...","relevant_contexts":[1,3],"attributed_sentences":[1,2]}]}
Include one result for every ITEM id. Use [] when none qualify.
"""


def build_retrieval_prompt(batch: list[dict]):
    blocks, metadata = [], []
    for local_id, row in enumerate(batch, start=1):
        sentences = split_sentences(row["reference"])
        contexts = row["retrieved_contexts"]
        metadata.append({"n_sentences": len(sentences), "n_contexts": len(contexts)})
        ref_text = "\n".join(f"[{i}] {x}" for i, x in enumerate(sentences, 1))
        ctx_text = "\n".join(f"[{i}] {x}" for i, x in enumerate(contexts, 1))
        blocks.append(
            f"=== ITEM {local_id} ===\nQUESTION:\n{row['user_input']}\n"
            f"REFERENCE SENTENCES:\n{ref_text}\nCONTEXTS:\n{ctx_text}\n"
        )
    return RETRIEVAL_JUDGE_PROMPT + "\n".join(blocks), metadata


def _validated_index_list(value: Any, upper: int) -> list[int] | None:
    if not isinstance(value, list):
        return None
    output = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        if not 1 <= item <= upper:
            return None
        if item not in output:
            output.append(item)
    return output


def parse_retrieval_response(text: str, metadata: list[dict]):
    data = None
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", (text or "").strip(),
                     flags=re.IGNORECASE | re.MULTILINE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return {}
    parsed = {}
    for entry in data["results"]:
        if not isinstance(entry, dict):
            continue
        local_id = entry.get("id")
        if isinstance(local_id, bool) or not isinstance(local_id, int):
            continue
        if not 1 <= local_id <= len(metadata) or local_id in parsed:
            continue
        info = metadata[local_id - 1]
        relevant = _validated_index_list(entry.get("relevant_contexts"), info["n_contexts"])
        attributed = _validated_index_list(entry.get("attributed_sentences"), info["n_sentences"])
        if relevant is None or attributed is None:
            continue
        flags = [int(i in relevant) for i in range(1, info["n_contexts"] + 1)]
        parsed[local_id] = {
            "context_precision": context_precision(flags),
            "context_recall": context_recall(set(attributed), info["n_sentences"]),
            "relevant_contexts": json.dumps(relevant),
            "attributed_sentences": f"{len(attributed)}/{info['n_sentences']}",
        }
    return parsed


def make_retrieval_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    groups = [
        ("Keseluruhan", results),
        *[(f"Bahasa: {k}", g) for k, g in results.groupby("language", dropna=False)],
        *[(f"Topik: {k}", g) for k, g in results.groupby("topic", dropna=False)],
    ]
    output = []
    for label, group in groups:
        ok = group[group["status"].eq("success")]
        precision = pd.to_numeric(ok["context_precision"], errors="coerce").dropna()
        recall = pd.to_numeric(ok["context_recall"], errors="coerce").dropna()
        output.append({
            "kelompok": label,
            "jumlah_data": len(group),
            "data_berhasil_dinilai": len(ok),
            "context_precision_persen": ratio_to_percentage(precision.mean()) if len(precision) else pd.NA,
            "context_recall_persen": ratio_to_percentage(recall.mean()) if len(recall) else pd.NA,
        })
    return pd.DataFrame(output)


def main() -> None:
    load_dotenv()
    if not 1 <= RETRIEVAL_BATCH_SIZE <= 20:
        raise ValueError("RETRIEVAL_BATCH_SIZE harus antara 1 dan 20.")
    data, invalid = load_retrieval_dataset(RETRIEVAL_INPUT_FILE)
    state = RateState(STATE_FILE)
    evaluators = build_evaluators(EVALUATORS, state)
    print_evaluator_status(evaluators)
    if all(ev.disabled for ev in evaluators):
        raise EnvironmentError("Tidak ada API key Gemini yang aktif.")

    previous = pd.DataFrame()
    if RETRIEVAL_OUTPUT_FILE.exists():
        try:
            old = pd.read_excel(RETRIEVAL_OUTPUT_FILE, sheet_name="RAGAS Results", engine="openpyxl")
            previous = old[old["status"].eq("success") & old["evaluator_model"].eq(JUDGE_MODEL)].copy()
        except Exception:
            previous = pd.DataFrame()
    done = set(previous["row_id"]) if not previous.empty else set()
    pending = deque(r for r in data.to_dict("records") if r["row_id"] not in done)
    calls_needed = -(-len(pending) // RETRIEVAL_BATCH_SIZE)
    calls_left = sum(max(0, ev.rpd - ev.state.used.get(ev.name, 0))
                     for ev in evaluators if not ev.disabled)
    print(f"Baris tersisa: {len(pending)} | panggilan dibutuhkan: ~{calls_needed} "
          f"| sisa kuota harian: {calls_left} panggilan")
    if calls_needed > calls_left:
        print("Kuota hari ini tidak cukup; proses berhenti saat kuota habis dan "
              "dilanjutkan otomatis pada hari berikutnya (hasil tersimpan per batch).")
    frames = [previous] if not previous.empty else []
    attempts: dict[Any, int] = {}
    failed: dict[Any, dict] = {}
    call_no = 0

    def checkpoint():
        parts = frames + ([pd.DataFrame(list(failed.values()))] if failed else [])
        if parts:
            result = pd.concat(parts, ignore_index=True).drop_duplicates("row_id", keep="last")
            result = result.sort_values("row_id").reset_index(drop=True)
            write_results(RETRIEVAL_OUTPUT_FILE, result, invalid, make_retrieval_summary(result))

    while pending:
        batch, estimated = [], 0
        while pending and len(batch) < RETRIEVAL_BATCH_SIZE:
            row = pending[0]
            cost = estimate_tokens(row["user_input"] + row["reference"] + "".join(row["retrieved_contexts"])) + 100
            if batch and estimated + cost > RETRIEVAL_MAX_PROMPT_TOKENS:
                break
            batch.append(pending.popleft())
            estimated += cost
        prompt, metadata = build_retrieval_prompt(batch)
        token_count = estimate_tokens(prompt) + RETRIEVAL_OUTPUT_TOKEN_RESERVE
        ev, wait = choose_evaluator(evaluators, token_count)
        if ev is None:
            pending.extendleft(reversed(batch))
            print("Kuota/model evaluator tidak tersedia; proses dihentikan.")
            break
        if wait > 0:
            _sleep(wait)
        call_no += 1
        try:
            ev.record(token_count)
            raw = call_llm(ev, prompt)
            ev.errors = 0
            ev.count_daily()
        except Exception as error:
            print(f"Gagal: {handle_error(ev, error)}")
            pending.extendleft(reversed(batch))
            continue

        parsed = parse_retrieval_response(raw, metadata)
        ok_rows, retry = [], []
        for local_id, row in enumerate(batch, start=1):
            head = {
                "row_id": row["row_id"], "question_id": row["question_id"],
                "language": row["language"], "topic": row["topic"],
            }
            if local_id in parsed:
                values = parsed[local_id]
                ok_rows.append({
                    **head, **values,
                    "context_precision_persen": ratio_to_percentage(values["context_precision"]),
                    "context_recall_persen": ratio_to_percentage(values["context_recall"]),
                    "batch": call_no, "evaluator_model": ev.model, "status": "success",
                })
            else:
                attempts[row["row_id"]] = attempts.get(row["row_id"], 0) + 1
                if attempts[row["row_id"]] < 3:
                    retry.append(row)
                else:
                    failed[row["row_id"]] = {
                        **head, "context_precision": pd.NA, "context_recall": pd.NA,
                        "context_precision_persen": pd.NA, "context_recall_persen": pd.NA,
                        "batch": call_no, "evaluator_model": ev.model,
                        "status": "failed: respons juri tidak valid",
                    }
        if ok_rows:
            frames.append(pd.DataFrame(ok_rows))
        pending.extendleft(reversed(retry))
        checkpoint()
        print(f"[Batch {call_no}] {ev.model}: {len(ok_rows)}/{len(batch)} berhasil")

    checkpoint()
    print(f"Hasil retrieval: {RETRIEVAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()