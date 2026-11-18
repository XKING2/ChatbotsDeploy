import random
import threading
import time
from datetime import datetime, timedelta

from google import genai
from google.genai import types

import config as _cfg

# ============================================================
# KONFIGURASI
# ============================================================

MAX_RETRIES = 4                 # percobaan per model (lintas key)
INITIAL_DELAY, MAX_DELAY = 3, 30
RATE_LIMIT_COOLDOWN = 65        # detik; batas per-menit sebuah key
MAX_WAIT = RATE_LIMIT_COOLDOWN + 5   # tunggu paling lama ini bila semua key cooldown
UNKNOWN_DAILY_COOLDOWN = 6 * 3600

THINKING_LEVEL = getattr(_cfg, "LLM_THINKING_LEVEL", "low")   # model 3.x: minimal|low|medium|high
THINKING_BUDGET = getattr(_cfg, "LLM_THINKING_BUDGET", None)  # model 2.5: None = bawaan model
TIMEOUT_SECONDS = getattr(_cfg, "LLM_TIMEOUT_SECONDS", 60)
MAX_OUTPUT_TOKENS = getattr(_cfg, "LLM_MAX_OUTPUT_TOKENS", None)

# Google menyarankan temperature bawaan untuk Gemini 3.x, jadi tidak dikirim
# kecuali LLM_TEMPERATURE diisi di config. Untuk Gemini 2.5 bawaannya 0.2.
_TEMPERATURE = getattr(_cfg, "LLM_TEMPERATURE", None)

# Tiga model yang didukung. LLM_MODEL boleh memakai nama lengkap atau alias.
MODELS = {
    "gemini-2.5-flash": {"family": "2.5", "aliases": ("2.5-flash", "gemini-2.5")},
    "gemini-3-flash-preview": {"family": "3",
                               "aliases": ("gemini-3-flash", "3-flash", "gemini-3")},
    "gemini-3.5-flash": {"family": "3", "aliases": ("3.5-flash", "gemini-3.5")},
}
_ALIASES = {alias: name for name, info in MODELS.items()
            for alias in (name, *info["aliases"])}


def resolve_model(value):
    """
    Terima nama model apa adanya dari config, termasuk awalan "models/" seperti
    pada daftar model Gemini API. Tiga model di atas dikenali lewat alias; nama
    "gemini-..." lain tetap dipakai dengan peringatan (keluarga ditebak dari nama).
    """
    key = str(value or "").strip().strip("'\"").lower()
    if key.startswith("models/"):
        key = key[len("models/"):]
    if key in _ALIASES:
        return _ALIASES[key]
    if key.startswith("gemini-"):
        family = "2.5" if key.startswith("gemini-2") else "3"
        MODELS[key] = {"family": family, "aliases": ()}
        print(f"[Gemini] PERINGATAN: model '{key}' bukan salah satu dari tiga model yang "
              f"diuji ({', '.join(m for m in MODELS if m != key)}); dipakai sebagai keluarga {family}.")
        return key
    raise ValueError(
        f"LLM_MODEL '{value}' tidak dikenali. Pilihan: {', '.join(MODELS)}."
    )


LLM_MODEL = resolve_model(getattr(_cfg, "LLM_MODEL", None))

# Model cadangan (opsional). Kosong = tidak ada pergantian model otomatis,
# supaya jawaban evaluasi tidak berpindah model tanpa sepengetahuan Anda.
_fallback = getattr(_cfg, "LLM_FALLBACK_MODELS", None) or []
if isinstance(_fallback, str):
    _fallback = [m for m in _fallback.split(",") if m.strip()]
MODEL_CHAIN = list(dict.fromkeys([LLM_MODEL, *(resolve_model(m) for m in _fallback)]))

# ============================================================
# API KEY: primary + hingga 5 key cadangan
# ============================================================

KEY_NAMES = ("GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2",
             "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY_5")
KEYS = list(dict.fromkeys(
    (name, str(getattr(_cfg, name, "") or "").strip())
    for name in KEY_NAMES if str(getattr(_cfg, name, "") or "").strip()
))
if not KEYS:
    raise ValueError("Tidak ada Gemini API key yang dikonfigurasi.")

_http = types.HttpOptions(timeout=int(TIMEOUT_SECONDS * 1000)) if TIMEOUT_SECONDS else None
CLIENTS = {name: genai.Client(api_key=key, http_options=_http) for name, key in KEYS}

_lock = threading.Lock()
_cooldown = {}      # (key_name, model) -> waktu (epoch) key boleh dipakai lagi
_dead_keys = set()  # key ditolak (401/403) selama proses berjalan
LAST_CALL = {}      # {"model": ..., "key": ...} pemanggilan sukses terakhir


# ============================================================
# KONFIGURASI PER MODEL
# ============================================================

def _config(model, thinking=True):
    options = {
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    temperature = _TEMPERATURE if _TEMPERATURE is not None else (
        0.2 if MODELS[model]["family"] == "2.5" else None)
    if temperature is not None:
        options["temperature"] = temperature
    if MAX_OUTPUT_TOKENS:
        options["max_output_tokens"] = MAX_OUTPUT_TOKENS

    if thinking:
        try:
            if MODELS[model]["family"] == "3" and THINKING_LEVEL:
                options["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
            elif MODELS[model]["family"] == "2.5" and THINKING_BUDGET is not None:
                options["thinking_config"] = types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
        except Exception as error:  # SDK lama tanpa parameter ini
            print(f"[Gemini] pengaturan thinking tidak didukung SDK ini: {error}")
    return types.GenerateContentConfig(**options)


def _thinking_label(model):
    if MODELS[model]["family"] == "3":
        return THINKING_LEVEL or "bawaan"
    return "bawaan" if THINKING_BUDGET is None else f"budget={THINKING_BUDGET}"


# ============================================================
# KLASIFIKASI ERROR
# ============================================================

def _code(error):
    for attr in ("code", "status_code"):
        try:
            return int(getattr(error, attr, None))
        except (TypeError, ValueError):
            pass
    return None


def _has(error, markers):
    text = str(error).upper()
    return any(marker in text for marker in markers)


def _is_key_error(error):
    return _code(error) in (401, 403) or _has(error, (
        "API KEY NOT VALID", "INVALID API KEY", "API_KEY_INVALID",
        "PERMISSION_DENIED", "UNAUTHENTICATED", "API KEY EXPIRED",
    ))


def _is_quota_error(error):
    return _code(error) == 429 or _has(error, ("RESOURCE_EXHAUSTED", "TOO MANY REQUESTS", "QUOTA"))


def _is_daily_quota(error):
    return _has(error, ("PERDAY", "PER_DAY", "PER DAY", "DAILY"))


class EmptyAnswer(RuntimeError):
    """Gemini membalas tanpa teks; biasanya sementara, jadi layak diulang."""


def _is_retryable(error):
    return isinstance(error, EmptyAnswer) or _code(error) in (408, 500, 502, 503, 504) or _has(error, (
        "UNAVAILABLE", "INTERNAL", "TIMEOUT", "TIMED OUT",
        "DEADLINE_EXCEEDED", "CONNECTION RESET", "CONNECTION ERROR",
    ))


def _is_thinking_rejected(error):
    text = str(error).lower()
    return "thinking" in text and any(w in text for w in ("not supported", "invalid", "unsupported"))


def _next_quota_reset():
    """Kuota harian Gemini direset tengah malam waktu Pasifik."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/Los_Angeles"))
        reset = (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
        return time.time() + (reset - now).total_seconds()
    except Exception:
        return time.time() + UNKNOWN_DAILY_COOLDOWN


# ============================================================
# PEMILIHAN KEY
# ============================================================

def _pick_key(model):
    """(nama_key, tunggu_detik): key pertama yang tidak sedang cooldown."""
    now = time.time()
    with _lock:
        live = [name for name, _ in KEYS if name not in _dead_keys]
        if not live:
            return None, 0.0
        waits = {name: max(0.0, _cooldown.get((name, model), 0.0) - now) for name in live}
    ready = [name for name in live if waits[name] == 0]
    if ready:
        return ready[0], 0.0
    best = min(live, key=waits.get)
    return best, waits[best]


def _mark(key_name, model, seconds):
    with _lock:
        _cooldown[(key_name, model)] = time.time() + seconds


def _log_usage(response, elapsed):
    usage = getattr(response, "usage_metadata", None)
    tokens = (
        f" | token: prompt={getattr(usage, 'prompt_token_count', None)} "
        f"berpikir={getattr(usage, 'thoughts_token_count', None)} "
        f"jawaban={getattr(usage, 'candidates_token_count', None)}"
    ) if usage else ""
    print(f"[Gemini] Selesai dalam {elapsed:.1f} detik{tokens}")


# ============================================================
# GENERATE
# ============================================================

def _try_model(model, prompt):
    """Jawaban dari satu model dengan rotasi key; None bila model tidak bisa dipakai."""
    config, thinking, last_error = _config(model), True, None
    print(f"[Gemini] model={model} | prompt={len(prompt)} karakter | "
          f"thinking={_thinking_label(model)} | timeout={TIMEOUT_SECONDS}s | key aktif={len(KEYS) - len(_dead_keys)}")

    attempt, calls = 0, 0
    max_calls = MAX_RETRIES + 3 * len(KEYS)   # pengaman: rotasi key tidak dihitung percobaan
    while attempt < MAX_RETRIES and calls < max_calls:
        key_name, wait = _pick_key(model)
        if key_name is None:
            break
        if wait > MAX_WAIT:
            print(f"[Gemini] Semua key {model} sedang cooldown (~{int(wait)} dtk); coba model lain.")
            break
        if wait > 0:
            print(f"[Gemini] Semua key sedang cooldown; menunggu {wait:.0f} dtk.")
            time.sleep(wait + 0.5)

        attempt += 1
        calls += 1
        try:
            print(f"[Gemini] {key_name} percobaan {attempt}/{MAX_RETRIES}")
            started = time.perf_counter()
            response = CLIENTS[key_name].models.generate_content(
                model=model, contents=prompt, config=config)
            answer = getattr(response, "text", None)
            if not isinstance(answer, str) or not answer.strip():
                raise EmptyAnswer("Gemini mengembalikan jawaban kosong.")
            _log_usage(response, time.perf_counter() - started)
            LAST_CALL.update({"model": model, "key": key_name})
            return answer.strip()

        except Exception as error:
            last_error = error
            print(f"[Gemini] ERROR {type(error).__name__}: {str(error)[:300]}")

            if thinking and _is_thinking_rejected(error):
                print("[Gemini] Model menolak pengaturan thinking; mengulang tanpa itu.")
                thinking, config = False, _config(model, False)
                attempt -= 1
                continue
            if _is_key_error(error):
                with _lock:
                    _dead_keys.add(key_name)
                print(f"[Gemini] {key_name} ditolak; dipindah ke key berikutnya.")
                attempt -= 1
                continue
            if _is_quota_error(error):
                # Kuota dihitung per key: langsung coba key lain tanpa menunggu.
                if _is_daily_quota(error):
                    with _lock:
                        _cooldown[(key_name, model)] = _next_quota_reset()
                    print(f"[Gemini] Kuota harian {key_name} untuk {model} habis.")
                else:
                    _mark(key_name, model, RATE_LIMIT_COOLDOWN)
                    print(f"[Gemini] {key_name} kena batas per menit; cooldown {RATE_LIMIT_COOLDOWN} dtk.")
                attempt -= 1   # rotasi key bukan percobaan ulang
                continue
            if not _is_retryable(error):
                raise
            if attempt < MAX_RETRIES:
                delay = min(INITIAL_DELAY * 2 ** (attempt - 1), MAX_DELAY)
                time.sleep(delay + random.uniform(0, min(2, delay * 0.2)))

    print(f"[Gemini] Model {model} tidak berhasil"
          + (f": {type(last_error).__name__}" if last_error else "."))
    return None


def generate_answer(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt kosong atau tidak valid.")

    for index, model in enumerate(MODEL_CHAIN):
        answer = _try_model(model, prompt)
        if answer is not None:
            if index:
                print(f"[Gemini] Jawaban dari model cadangan: {model}")
            return answer

    raise RuntimeError(
        "Semua API key Gemini gagal. Periksa kuota, API key, akses model, dan koneksi."
    )