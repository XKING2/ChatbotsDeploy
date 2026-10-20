
import time
import random

from google import genai
from config import (
    GEMINI_API_KEY,
    GEMINI_API_KEY_1,
    GEMINI_API_KEY_2,
    LLM_MODEL,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RETRIES = 4
INITIAL_DELAY = 3
MAX_DELAY = 30

# Prioritas API key: primary -> failover 1 -> failover 2.
API_KEYS = []
_seen_keys = set()

for key_name, key_value in (
    ("GEMINI_API_KEY", GEMINI_API_KEY),
    ("GEMINI_API_KEY_1", GEMINI_API_KEY_1),
    ("GEMINI_API_KEY_2", GEMINI_API_KEY_2),
):
    if key_value and key_value.strip():
        key_value = key_value.strip()

        # Hindari mencoba key yang sama berulang kali.
        if key_value not in _seen_keys:
            API_KEYS.append((key_name, key_value))
            _seen_keys.add(key_value)

if not LLM_MODEL:
    raise ValueError("LLM_MODEL belum dikonfigurasi.")

if not API_KEYS:
    raise ValueError(
        "Tidak ada Gemini API key yang dikonfigurasi. "
        "Periksa GEMINI_API_KEY, GEMINI_API_KEY_1, "
        "dan GEMINI_API_KEY_2."
    )


# Buat client hanya untuk key yang tersedia.
CLIENTS = [
    (key_name, genai.Client(api_key=key_value))
    for key_name, key_value in API_KEYS
]


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def _get_error_code(error):
    """Ambil HTTP status code jika tersedia."""
    for attr in ("code", "status_code"):
        value = getattr(error, attr, None)

        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

    return None


def _is_key_error(error):
    """
    Error yang biasanya berkaitan dengan API key atau izin.
    Lewati key ini dan lanjutkan ke key berikutnya.
    """
    code = _get_error_code(error)
    message = str(error).upper()

    if code in (401, 403):
        return True

    key_error_markers = (
        "API KEY NOT VALID",
        "INVALID API KEY",
        "API_KEY_INVALID",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
        "API KEY EXPIRED",
    )

    return any(marker in message for marker in key_error_markers)


def _is_retryable_error(error):
    """Error sementara yang layak dicoba ulang pada key yang sama."""
    code = _get_error_code(error)
    message = str(error).upper()

    if code in (408, 429, 500, 502, 503, 504):
        return True

    retryable_markers = (
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "INTERNAL",
        "TIMEOUT",
        "TIMED OUT",
        "DEADLINE_EXCEEDED",
        "SERVICE UNAVAILABLE",
        "TEMPORARILY UNAVAILABLE",
        "CONNECTION RESET",
        "CONNECTION ERROR",
        "TOO MANY REQUESTS",
        "429",
        "500",
        "502",
        "503",
        "504",
    )

    return any(marker in message for marker in retryable_markers)


def _retry_delay(attempt):
    """Exponential backoff dengan jitter."""
    delay = min(
        INITIAL_DELAY * (2 ** (attempt - 1)),
        MAX_DELAY,
    )
    return delay + random.uniform(0, min(2, delay * 0.2))


# ============================================================
# GENERATE ANSWER WITH FAILOVER
# ============================================================

def generate_answer(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt kosong atau tidak valid.")

    print("=" * 80)
    print("GEMINI REQUEST")
    print("=" * 80)
    print("MODEL:", LLM_MODEL)
    print("PROMPT LENGTH:", len(prompt))
    print("AVAILABLE API KEYS:", len(CLIENTS))
    print("=" * 80)

    last_error = None

    for key_index, (key_name, client) in enumerate(CLIENTS):
        print(
            f"[Gemini] Menggunakan {key_name} "
            f"({key_index + 1}/{len(CLIENTS)})"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(
                    f"[Gemini] Request "
                    f"{attempt}/{MAX_RETRIES} menggunakan {key_name}"
                )

                response = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt,
                )

                answer = getattr(response, "text", None)

                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError(
                        "Gemini mengembalikan jawaban kosong."
                    )

                print(
                    f"[Gemini] Response berhasil "
                    f"dengan {key_name}."
                )

                return answer.strip()

            except Exception as error:
                last_error = error

                print("=" * 80)
                print("GEMINI ERROR")
                print("=" * 80)
                print("API KEY:", key_name)
                print("ATTEMPT:", f"{attempt}/{MAX_RETRIES}")
                print("ERROR TYPE:", type(error).__name__)
                print("ERROR:", error)
                print("=" * 80)

                # Error key/izin: jangan retry key yang sama.
                if _is_key_error(error):
                    print(
                        f"[Gemini] {key_name} ditolak. "
                        "Beralih ke API key berikutnya."
                    )
                    break

                # Error permanen lain: tidak perlu menghabiskan
                # seluruh key untuk masalah yang sama.
                if not _is_retryable_error(error):
                    print(
                        "[Gemini] Error bukan error sementara. "
                        "Retry dan failover dihentikan."
                    )
                    raise

                # Masih ada kesempatan pada key yang sama.
                if attempt < MAX_RETRIES:
                    delay = _retry_delay(attempt)

                    print(
                        f"[Gemini] Retry dalam {delay:.2f} detik..."
                    )
                    time.sleep(delay)
                    continue

                # Semua retry pada key ini sudah habis.
                print(
                    f"[Gemini] Retry {key_name} habis. "
                    "Beralih ke API key berikutnya."
                )
                break

        # Lanjut ke API key berikutnya jika tersedia.
        if key_index < len(CLIENTS) - 1:
            next_key_name = CLIENTS[key_index + 1][0]
            print(
                f"[Gemini] Failover: {key_name} -> {next_key_name}"
            )

    print("[Gemini] Semua API key gagal.")

    raise RuntimeError(
        "Semua API key Gemini gagal memproses permintaan. "
        "Periksa kuota, status API key, akses model, "
        "dan koneksi internet."
    ) from last_error
