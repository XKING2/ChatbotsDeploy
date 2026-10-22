import random
import time

from google import genai
from google.genai import types

import config as _cfg
from config import GEMINI_API_KEY, GEMINI_API_KEY_1, GEMINI_API_KEY_2, LLM_MODEL

MAX_RETRIES = 4
INITIAL_DELAY, MAX_DELAY = 3, 30

# Bisa ditimpa dari config.py. Thinking "low" mempercepat jawaban RAG
# (bawaan model flash terbaru "medium"); None = bawaan model.
THINKING_LEVEL = getattr(_cfg, "LLM_THINKING_LEVEL", "low")
TIMEOUT_SECONDS = getattr(_cfg, "LLM_TIMEOUT_SECONDS", 60)
MAX_OUTPUT_TOKENS = getattr(_cfg, "LLM_MAX_OUTPUT_TOKENS", None)
TEMPERATURE = getattr(_cfg, "LLM_TEMPERATURE", 0.2)

if not LLM_MODEL:
    raise ValueError("LLM_MODEL belum dikonfigurasi.")

# Primary -> failover 1 -> failover 2, tanpa key ganda.
KEYS = list(dict.fromkeys(
    (name, key.strip())
    for name, key in (
        ("GEMINI_API_KEY", GEMINI_API_KEY),
        ("GEMINI_API_KEY_1", GEMINI_API_KEY_1),
        ("GEMINI_API_KEY_2", GEMINI_API_KEY_2),
    )
    if key and key.strip()
))
if not KEYS:
    raise ValueError("Tidak ada Gemini API key yang dikonfigurasi.")

_http = types.HttpOptions(timeout=int(TIMEOUT_SECONDS * 1000)) if TIMEOUT_SECONDS else None
CLIENTS = [(name, genai.Client(api_key=key, http_options=_http)) for name, key in KEYS]


def _config(thinking=True):
    options = {
        "temperature": TEMPERATURE,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if MAX_OUTPUT_TOKENS:
        options["max_output_tokens"] = MAX_OUTPUT_TOKENS
    if thinking and THINKING_LEVEL:
        try:
            options["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
        except Exception as error:  # SDK lama tanpa thinking_level
            print(f"[Gemini] thinking_level tidak didukung SDK ini: {error}")
    return types.GenerateContentConfig(**options)


def _code(error):
    for attr in ("code", "status_code"):
        try:
            return int(getattr(error, attr))
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


def _is_retryable(error):
    return _code(error) in (408, 429, 500, 502, 503, 504) or _has(error, (
        "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "TIMEOUT", "TIMED OUT",
        "DEADLINE_EXCEEDED", "CONNECTION RESET", "CONNECTION ERROR", "TOO MANY REQUESTS",
    ))


def _is_thinking_rejected(error):
    text = str(error).lower()
    return "thinking" in text and any(w in text for w in ("not supported", "invalid", "unsupported"))


def _log_usage(response, elapsed):
    usage = getattr(response, "usage_metadata", None)
    tokens = (
        f" | token: prompt={getattr(usage, 'prompt_token_count', None)} "
        f"berpikir={getattr(usage, 'thoughts_token_count', None)} "
        f"jawaban={getattr(usage, 'candidates_token_count', None)}"
    ) if usage else ""
    print(f"[Gemini] Selesai dalam {elapsed:.1f} detik{tokens}")


def generate_answer(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt kosong atau tidak valid.")

    print(f"[Gemini] model={LLM_MODEL} | prompt={len(prompt)} karakter | "
          f"thinking={THINKING_LEVEL or 'bawaan'} | timeout={TIMEOUT_SECONDS}s")

    config, thinking, last_error = _config(), True, None

    for key_name, client in CLIENTS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[Gemini] {key_name} percobaan {attempt}/{MAX_RETRIES}")
                started = time.perf_counter()
                response = client.models.generate_content(
                    model=LLM_MODEL, contents=prompt, config=config
                )
                answer = getattr(response, "text", None)
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("Gemini mengembalikan jawaban kosong.")
                _log_usage(response, time.perf_counter() - started)
                return answer.strip()

            except Exception as error:
                last_error = error
                print(f"[Gemini] ERROR {type(error).__name__}: {error}")

                if thinking and _is_thinking_rejected(error):
                    print("[Gemini] Model menolak thinking_level; mengulang tanpa itu.")
                    thinking, config = False, _config(False)
                    continue
                if _is_key_error(error):
                    break  # key ditolak: pindah ke key berikutnya
                if not _is_retryable(error):
                    raise
                if attempt < MAX_RETRIES:
                    delay = min(INITIAL_DELAY * 2 ** (attempt - 1), MAX_DELAY)
                    time.sleep(delay + random.uniform(0, min(2, delay * 0.2)))

    raise RuntimeError(
        "Semua API key Gemini gagal. Periksa kuota, API key, akses model, dan koneksi."
    ) from last_error