
import time
import random
import re

from google import genai
from config import GEMINI_API_KEY, LLM_MODEL


client = genai.Client(api_key=GEMINI_API_KEY)

MAX_RETRIES = 4
INITIAL_DELAY = 3
MAX_DELAY = 30


def _is_retryable_error(error):
    """Mengenali error sementara dari Gemini."""
    message = str(error).upper()

    # HTTP status code dari exception SDK
    code = getattr(error, "code", None)
    if code is None:
        code = getattr(error, "status_code", None)

    if code is not None:
        try:
            if int(code) in (408, 429, 500, 502, 503, 504):
                return True
        except (TypeError, ValueError):
            pass

    # Fallback jika SDK tidak menyediakan status code.
    return any(
        marker in message
        for marker in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "INTERNAL",
            "TIMEOUT",
            "DEADLINE_EXCEEDED",
            "SERVICE UNAVAILABLE",
            "TEMPORARILY UNAVAILABLE",
        )
    )


def generate_answer(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt kosong atau tidak valid.")

    print("=" * 80)
    print("GEMINI REQUEST")
    print("=" * 80)
    print("MODEL:", LLM_MODEL)
    print("PROMPT LENGTH:", len(prompt))
    print("=" * 80)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[Gemini] Request {attempt}/{MAX_RETRIES}")

            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )

            answer = getattr(response, "text", None)

            if not answer or not answer.strip():
                raise RuntimeError(
                    "Gemini mengembalikan jawaban kosong."
                )

            print("[Gemini] Response berhasil.")
            return answer.strip()

        except Exception as error:
            last_error = error

            print("=" * 80)
            print("LLM ERROR")
            print("=" * 80)
            print(f"Jenis error: {type(error).__name__}")
            print(f"Detail: {error}")
            print("=" * 80)

            if not _is_retryable_error(error):
                print("[Gemini] Error tidak dapat di-retry.")
                raise

            if attempt == MAX_RETRIES:
                break

            # 3, 6, 12, 24 detik + jitter acak.
            delay = min(
                INITIAL_DELAY * (2 ** (attempt - 1)),
                MAX_DELAY,
            )
            delay += random.uniform(0, min(2, delay * 0.2))

            print(
                f"[Gemini] Menunggu {delay:.2f} detik "
                "sebelum mencoba kembali..."
            )
            time.sleep(delay)

    print("[Gemini] Semua percobaan gagal.")

    raise RuntimeError(
        "Layanan Gemini sedang tidak tersedia. "
        "Silakan coba kembali beberapa saat lagi."
    ) from last_error
