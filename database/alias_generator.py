
import json
import re
import threading
import time

import config

from google import genai
from google.genai import types


# =========================================================
# CONFIGURATION
# =========================================================

API_KEYS = list(dict.fromkeys(
    key.strip()
    for key in [
        getattr(config, "GEMINI_API_KEY", None),
        getattr(config, "GEMINI_API_KEY_2", None),
    ]
    if isinstance(key, str) and key.strip()
))

LLM_MODEL = config.LLM_MODEL

BATCH_SIZE = 20

MAX_CALLS_BEFORE_COOLDOWN = 5
COOLDOWN_SECONDS = 90

MAX_RETRIES_PER_KEY = 3
INITIAL_RETRY_DELAY = 4
MAX_RETRY_DELAY = 30

MAX_ALIASES = 20

ALIAS_LANGUAGES = [
    "Indonesian",
    "English",
    "Mandarin Chinese",
    "Japanese",
    "Russian",
]


# =========================================================
# GLOBAL STATE
# =========================================================

_clients = {}

_client_lock = threading.Lock()
_rate_limit_lock = threading.Lock()

_call_count = 0
_next_allowed_time = 0


# =========================================================
# GEMINI CLIENT
# =========================================================

def _get_client(api_key):
    """
    Membuat dan menyimpan client Gemini.

    Client digunakan kembali agar tidak perlu
    membuat koneksi baru untuk setiap request.
    """

    with _client_lock:

        if api_key not in _clients:
            _clients[api_key] = genai.Client(
                api_key=api_key
            )

        return _clients[api_key]


# =========================================================
# RATE LIMITER
# =========================================================

def _wait_and_register_call():
    """
    Membatasi jumlah request secara global.

    Maksimal 5 request sebelum cooldown 90 detik.
    Penghitungan dan penguncian dilakukan secara atomik.
    """

    global _call_count
    global _next_allowed_time

    while True:

        with _rate_limit_lock:

            now = time.monotonic()

            if now < _next_allowed_time:

                wait_time = _next_allowed_time - now

            elif _call_count >= MAX_CALLS_BEFORE_COOLDOWN:

                _call_count = 0

                _next_allowed_time = (
                    now + COOLDOWN_SECONDS
                )

                wait_time = COOLDOWN_SECONDS

                print(
                    f"[ALIAS] Cooldown "
                    f"{COOLDOWN_SECONDS} detik."
                )

            else:

                _call_count += 1

                return

        time.sleep(wait_time)


# =========================================================
# ERROR HANDLING
# =========================================================

def _get_error_code(error):
    """
    Mengambil HTTP status code dari exception Gemini.
    """

    code = getattr(error, "code", None)

    if code is None:
        code = getattr(error, "status_code", None)

    if code is None:
        response = getattr(error, "response", None)

        if response is not None:
            code = getattr(response, "status_code", None)

    try:
        return int(code) if code is not None else None

    except (TypeError, ValueError):
        return None


def _is_retryable_error(error):
    """
    Menentukan apakah error dapat dicoba kembali.
    """

    code = _get_error_code(error)

    if code in (408, 429, 500, 502, 503, 504):
        return True

    message = str(error).upper()

    temporary_errors = (
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "TOO MANY REQUESTS",
        "INTERNAL SERVER ERROR",
        "SERVICE UNAVAILABLE",
        "DEADLINE EXCEEDED",
        "TIMED OUT",
        "TIMEOUT",
        "CONNECTION RESET",
        "CONNECTION ERROR",
    )

    return any(
        keyword in message
        for keyword in temporary_errors
    )


def _retry_delay(attempt, error):
    """
    Menghitung waktu tunggu exponential backoff.
    """

    code = _get_error_code(error)

    if code == 429:
        return MAX_RETRY_DELAY

    delay = INITIAL_RETRY_DELAY * (2 ** attempt)

    return min(
        delay,
        MAX_RETRY_DELAY
    )


# =========================================================
# JSON RESPONSE
# =========================================================

def _clean_json_response(response_text):
    """
    Membersihkan dan memvalidasi JSON dari Gemini.
    """

    if not isinstance(response_text, str):
        return None

    text = response_text.strip()

    if not text:
        return None

    # Menghapus Markdown code fence.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        result = json.loads(text)

    except json.JSONDecodeError:
        return None

    if not isinstance(result, dict):
        return None

    return result


# =========================================================
# GEMINI REQUEST
# =========================================================

def _generate_content_with_failover(prompt):
    """
    Mengirim prompt ke Gemini dengan:

    1. AFC dinonaktifkan.
    2. Output JSON.
    3. Retry untuk error sementara.
    4. Failover ke API key berikutnya.
    """

    if not API_KEYS:

        print(
            "[ALIAS] Tidak ada Gemini API key tersedia."
        )

        return None

    for key_index, api_key in enumerate(API_KEYS):

        client = _get_client(api_key)

        print(
            f"[ALIAS] Menggunakan API key "
            f"{key_index + 1}/{len(API_KEYS)}"
        )

        for attempt in range(MAX_RETRIES_PER_KEY):

            try:

                # Rate limiter.
                _wait_and_register_call()

                response = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(

                        # Memastikan respons berupa JSON.
                        response_mime_type="application/json",

                        # Menonaktifkan Automatic Function Calling.
                        automatic_function_calling=(
                            types.AutomaticFunctionCallingConfig(
                                disable=True
                            )
                        ),

                        temperature=0.2,
                    ),
                )

                response_text = response.text

                result = _clean_json_response(
                    response_text
                )

                if result is None:

                    print(
                        "[ALIAS] Respons Gemini bukan JSON valid."
                    )

                    # Coba API key berikutnya.
                    break

                return result

            except Exception as error:

                retryable = _is_retryable_error(error)

                print(
                    f"[ALIAS] Request gagal: {error}"
                )

                if not retryable:

                    print(
                        "[ALIAS] Error tidak dapat di-retry. "
                        "Mencoba API key berikutnya."
                    )

                    break

                if attempt == MAX_RETRIES_PER_KEY - 1:

                    print(
                        "[ALIAS] Batas retry tercapai. "
                        "Mencoba API key berikutnya."
                    )

                    break

                wait_time = _retry_delay(
                    attempt,
                    error,
                )

                print(
                    f"[ALIAS] Retry "
                    f"{attempt + 1}/{MAX_RETRIES_PER_KEY - 1} "
                    f"dalam {wait_time} detik."
                )

                time.sleep(wait_time)

    print(
        "[ALIAS] Semua API key gagal menghasilkan respons."
    )

    return None


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def _normalize_text(text):
    return " ".join(
        str(text or "").split()
    )


def _clean_alias_list(
    aliases,
    original_name=None,
    max_aliases=MAX_ALIASES,
):
    """
    Membersihkan alias dan menghapus duplikat.
    """

    if not isinstance(aliases, list):
        return []

    cleaned = []
    seen = set()

    original_normalized = (
        _normalize_text(original_name).casefold()
        if original_name
        else None
    )

    for alias in aliases:

        if not isinstance(alias, str):
            continue

        alias = _normalize_text(alias)

        if not alias:
            continue

        normalized = alias.casefold()

        if normalized == original_normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(alias)

        if len(cleaned) >= max_aliases:
            break

    return cleaned


# =========================================================
# HEADING ALIAS GENERATION
# =========================================================

def _generate_heading_batch(heading_names):

    prompt = f"""
Anda adalah ahli pengolahan bahasa untuk sistem
chatbot layanan keimigrasian.

Buat alias atau variasi penyebutan untuk setiap heading
berikut dalam bahasa:

{", ".join(ALIAS_LANGUAGES)}

Ketentuan:
1. Maksimal 20 alias untuk setiap heading.
2. Alias harus memiliki makna yang sama.
3. Gunakan variasi istilah, sinonim, dan terjemahan.
4. Jangan membuat informasi baru.
5. Jangan mengulang nama heading asli.
6. Hindari alias yang terlalu umum atau tidak relevan.
7. Hasil harus berupa JSON valid.
8. Gunakan nama heading asli sebagai key.
9. Value setiap key harus berupa array string.

Data heading:

{json.dumps(heading_names, ensure_ascii=False)}

Format:

{{
    "Nama Heading": [
        "alias 1",
        "alias 2"
    ]
}}
"""

    return _generate_content_with_failover(prompt)


def generate_heading_aliases_batch(heading_names):
    """
    Menghasilkan alias untuk beberapa heading.

    Output:
        {
            "Nama Heading": ["alias 1", "alias 2"]
        }
    """

    results = {}

    heading_names = list(dict.fromkeys(
        _normalize_text(name)
        for name in heading_names
        if _normalize_text(name)
    ))

    for start in range(
        0,
        len(heading_names),
        BATCH_SIZE,
    ):

        batch = heading_names[
            start:start + BATCH_SIZE
        ]

        print(
            f"[ALIAS] Heading batch "
            f"{start // BATCH_SIZE + 1}"
        )

        response = _generate_heading_batch(batch)

        if not isinstance(response, dict):
            continue

        for heading in batch:

            aliases = response.get(heading)

            if aliases is None:
                continue

            results[heading] = _clean_alias_list(
                aliases,
                original_name=heading,
            )

    return results


# =========================================================
# SUBHEADING ALIAS GENERATION
# =========================================================

def _generate_subheading_batch(subheadings):

    prompt_data = [
        {
            "heading_id": item["heading_id"],
            "heading": item["heading"],
            "subheading_id": item["subheading_id"],
            "subheading": item["subheading"],
        }
        for item in subheadings
    ]

    prompt = f"""
Anda adalah ahli pengolahan bahasa untuk chatbot
layanan keimigrasian.

Buat alias untuk setiap subheading berdasarkan konteks
heading induknya.

Bahasa alias:

{", ".join(ALIAS_LANGUAGES)}

Ketentuan:
1. Buat 5 sampai 15 alias jika memungkinkan.
2. Maksimal 20 alias untuk setiap subheading.
3. Alias harus sesuai dengan konteks heading.
4. Gunakan sinonim, variasi pertanyaan, dan terjemahan.
5. Jangan mengubah makna subheading.
6. Jangan mengulang nama subheading asli.
7. Jangan membuat informasi baru.
8. Hasil harus berupa JSON valid.
9. Gunakan heading_id::subheading_id sebagai key.

Data:

{json.dumps(prompt_data, ensure_ascii=False)}

Format:

{{
    "heading_id::subheading_id": [
        "alias 1",
        "alias 2"
    ]
}}
"""

    return _generate_content_with_failover(prompt)


def generate_subheading_aliases_batch(subheadings):
    """
    Menghasilkan alias untuk beberapa subheading.

    Output:
        {
            "heading_id::subheading_id": [
                "alias 1",
                "alias 2"
            ]
        }
    """

    results = {}

    valid_items = [
        item
        for item in subheadings
        if isinstance(item, dict)
        and all(
            item.get(field)
            for field in (
                "heading_id",
                "heading",
                "subheading_id",
                "subheading",
            )
        )
    ]

    for start in range(
        0,
        len(valid_items),
        BATCH_SIZE,
    ):

        batch = valid_items[
            start:start + BATCH_SIZE
        ]

        print(
            f"[ALIAS] Subheading batch "
            f"{start // BATCH_SIZE + 1}"
        )

        response = _generate_subheading_batch(batch)

        if not isinstance(response, dict):
            continue

        for item in batch:

            key = (
                f"{item['heading_id']}::"
                f"{item['subheading_id']}"
            )

            aliases = response.get(key)

            if aliases is None:
                continue

            results[key] = _clean_alias_list(
                aliases,
                original_name=item["subheading"],
            )

    return results
