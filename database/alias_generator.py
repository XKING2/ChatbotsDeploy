import json
import re
import threading
import time

from google import genai
from google.genai import types

import config

# ============================================================
# CONFIG
# ============================================================

API_KEYS = list(dict.fromkeys(
    key.strip()
    for key in (
        getattr(config, "GEMINI_API_KEY", None),
        getattr(config, "GEMINI_API_KEY_2", None),
    )
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


# ============================================================
# PROMPT
# ============================================================

# Daftar acuan. LLM hanya boleh memakainya bila relevan dengan layanan.
# Tambahkan singkatan lain yang benar-benar dipakai di domain Anda.
ABBREVIATION_GLOSSARY = """
Referensi singkatan dan istilah populer
(gunakan HANYA yang relevan dengan layanan ini, jangan disalin semua):
- GV = Golden Visa
- KITAS / ITAS = (Kartu) Izin Tinggal Terbatas
- KITAP / ITAP = (Kartu) Izin Tinggal Tetap
- VITAS = Visa Izin Tinggal Terbatas
- VOA / e-VOA = Visa on Arrival / electronic Visa on Arrival
- REP = Re-entry Permit (izin masuk kembali)
- EPO = Exit Permit Only (izin keluar sekali jalan)
- SPLP = Surat Perjalanan Laksana Paspor
- M-Paspor = aplikasi permohonan paspor
- WNA / WNI = Warga Negara Asing / Warga Negara Indonesia
- Kanim = Kantor Imigrasi
- TPI = Tempat Pemeriksaan Imigrasi
- Rudenim = Rumah Detensi Imigrasi
- Ditjenim = Direktorat Jenderal Imigrasi
- SKTT = Surat Keterangan Tempat Tinggal
- STM = Surat Tanda Melapor
""".strip()

ALIAS_STYLE = f"""
Gaya alias (penting):
1. Natural: tulis seperti orang sungguhan menyebut atau mencari layanan
   ini di chat, bukan terjemahan kaku atau bahasa formal dokumen.
2. Sertakan jika benar-benar dipakai orang:
   a. Singkatan resmi maupun populer (contoh: GV untuk Golden Visa).
   b. Kepanjangan lengkap dari singkatan pada nama asli, dan sebaliknya.
   c. Bahasa sehari-hari dan gaul anak muda yang sopan
      (contoh: "bikin paspor", "urus visa", "perpanjang KITAS").
   d. Variasi ejaan yang umum (contoh: "e-paspor", "epaspor",
      "paspor elektronik").
   e. Terjemahan natural ke bahasa: {", ".join(ALIAS_LANGUAGES)}.
3. Akurasi singkatan:
   - Hanya pakai singkatan yang benar-benar dikenal dan umum dipakai.
   - Jangan mengarang singkatan baru. Jika ragu, jangan dimasukkan.
   - Singkatan harus merujuk ke layanan yang SAMA PERSIS. Jangan pakai
     nama atau singkatan layanan lain (contoh: KITAS bukan KITAP).
   - Jangan memakai kata yang terlalu umum dan bisa berarti banyak
     layanan (contoh: "visa" atau "paspor" saja).
4. Tidak ada kata kasar. Alias berupa frasa pendek.
5. Urutkan dari yang paling mungkin diketik pengguna.

{ABBREVIATION_GLOSSARY}
""".strip()


def _heading_prompt(heading_names):
    return f"""
Anda adalah ahli pengolahan bahasa untuk chatbot layanan keimigrasian.

Buat alias atau variasi penyebutan untuk setiap heading berikut.

Ketentuan umum:
1. Maksimal {MAX_ALIASES} alias untuk setiap heading.
2. Alias harus memiliki makna yang sama dengan heading.
3. Jangan membuat informasi baru.
4. Jangan mengulang nama heading asli.
5. Alias heading berupa nama atau sebutan layanan, bukan kalimat tanya.
6. Hasil harus berupa JSON valid.
7. Gunakan nama heading asli sebagai key.
8. Value setiap key harus berupa array string.

{ALIAS_STYLE}

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


def _subheading_prompt(subheadings):
    data = [
        {
            "heading_id": item["heading_id"],
            "heading": item["heading"],
            "subheading_id": item["subheading_id"],
            "subheading": item["subheading"],
        }
        for item in subheadings
    ]

    return f"""
Anda adalah ahli pengolahan bahasa untuk chatbot layanan keimigrasian.

Buat alias untuk setiap subheading berdasarkan konteks heading
induknya.

Ketentuan umum:
1. Buat 5 sampai 15 alias jika memungkinkan.
2. Maksimal {MAX_ALIASES} alias untuk setiap subheading.
3. Alias harus sesuai dengan konteks heading induk.
4. Boleh berupa sebutan singkat atau pertanyaan singkat yang natural
   (contoh: "syarat GV apa aja", "biaya perpanjang KITAS").
5. Jangan mengubah makna subheading.
6. Jangan mengulang nama subheading asli.
7. Jangan membuat informasi baru.
8. Hasil harus berupa JSON valid.
9. Gunakan heading_id::subheading_id sebagai key.

{ALIAS_STYLE}

Data:

{json.dumps(data, ensure_ascii=False)}

Format:

{{
    "heading_id::subheading_id": [
        "alias 1",
        "alias 2"
    ]
}}
"""


# ============================================================
# GEMINI CLIENT & RATE LIMITER
# ============================================================

_clients = {}
_client_lock = threading.Lock()
_rate_limit_lock = threading.Lock()
_call_count = 0
_next_allowed_time = 0


def _get_client(api_key):
    with _client_lock:
        if api_key not in _clients:
            _clients[api_key] = genai.Client(api_key=api_key)

        return _clients[api_key]


def _wait_and_register_call():
    global _call_count, _next_allowed_time

    while True:
        with _rate_limit_lock:
            now = time.monotonic()

            if now < _next_allowed_time:
                wait_time = _next_allowed_time - now

            elif _call_count >= MAX_CALLS_BEFORE_COOLDOWN:
                _call_count = 0
                _next_allowed_time = now + COOLDOWN_SECONDS
                wait_time = COOLDOWN_SECONDS
                print(f"[ALIAS] Cooldown {COOLDOWN_SECONDS} detik.")

            else:
                _call_count += 1
                return

        time.sleep(wait_time)


# ============================================================
# ERROR HANDLING
# ============================================================

def _get_error_code(error):
    code = getattr(error, "code", None)

    if code is None:
        code = getattr(error, "status_code", None)

    if code is None:
        code = getattr(
            getattr(error, "response", None), "status_code", None
        )

    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable_error(error):
    if _get_error_code(error) in (408, 429, 500, 502, 503, 504):
        return True

    message = str(error).upper()

    return any(
        keyword in message
        for keyword in (
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
    )


def _retry_delay(attempt, error):
    if _get_error_code(error) == 429:
        return MAX_RETRY_DELAY

    return min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)


# ============================================================
# GEMINI REQUEST
# ============================================================

def _clean_json_response(response_text):
    if not isinstance(response_text, str):
        return None

    text = response_text.strip()

    if not text:
        return None

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None

    return result if isinstance(result, dict) else None


def _generate_content_with_failover(prompt):
    if not API_KEYS:
        print("[ALIAS] Tidak ada Gemini API key tersedia.")
        return None

    request_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        temperature=0.2,
    )

    for key_index, api_key in enumerate(API_KEYS):
        client = _get_client(api_key)

        print(
            f"[ALIAS] Menggunakan API key "
            f"{key_index + 1}/{len(API_KEYS)}"
        )

        for attempt in range(MAX_RETRIES_PER_KEY):
            try:
                _wait_and_register_call()

                response = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt,
                    config=request_config,
                )

                result = _clean_json_response(response.text)

                if result is None:
                    print("[ALIAS] Respons Gemini bukan JSON valid.")
                    break

                return result

            except Exception as error:
                print(f"[ALIAS] Request gagal: {error}")

                if not _is_retryable_error(error):
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

                wait_time = _retry_delay(attempt, error)

                print(
                    f"[ALIAS] Retry {attempt + 1}/"
                    f"{MAX_RETRIES_PER_KEY - 1} dalam {wait_time} detik."
                )

                time.sleep(wait_time)

    print("[ALIAS] Semua API key gagal menghasilkan respons.")

    return None


# ============================================================
# ALIAS CLEANING
# ============================================================

def _normalize_text(text):
    return " ".join(str(text or "").split())


def _clean_alias_list(aliases, original_name=None, max_aliases=MAX_ALIASES):
    if not isinstance(aliases, list):
        return []

    cleaned = []
    seen = set()

    original = (
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

        key = alias.casefold()

        if key == original or key in seen:
            continue

        seen.add(key)
        cleaned.append(alias)

        if len(cleaned) >= max_aliases:
            break

    return cleaned


# ============================================================
# PUBLIC API
# ============================================================

def generate_heading_aliases_batch(heading_names):
    """
    Output: {"Nama Heading": ["alias 1", "alias 2"]}

    Key hasil memakai nama heading seperti yang dikirim pemanggil.
    """

    originals = {}

    for name in heading_names:
        normalized = _normalize_text(name)

        if normalized:
            originals.setdefault(normalized, name)

    names = list(originals)
    results = {}

    for start in range(0, len(names), BATCH_SIZE):
        batch = names[start:start + BATCH_SIZE]

        print(f"[ALIAS] Heading batch {start // BATCH_SIZE + 1}")

        response = _generate_content_with_failover(_heading_prompt(batch))

        if not isinstance(response, dict):
            continue

        for heading in batch:
            aliases = response.get(heading)

            if aliases is None:
                continue

            results[originals[heading]] = _clean_alias_list(
                aliases, original_name=heading
            )

    return results


def generate_subheading_aliases_batch(subheadings):
    """
    Output: {"heading_id::subheading_id": ["alias 1", "alias 2"]}
    """

    fields = ("heading_id", "heading", "subheading_id", "subheading")

    valid_items = [
        item
        for item in subheadings
        if isinstance(item, dict) and all(item.get(f) for f in fields)
    ]

    results = {}

    for start in range(0, len(valid_items), BATCH_SIZE):
        batch = valid_items[start:start + BATCH_SIZE]

        print(f"[ALIAS] Subheading batch {start // BATCH_SIZE + 1}")

        response = _generate_content_with_failover(
            _subheading_prompt(batch)
        )

        if not isinstance(response, dict):
            continue

        for item in batch:
            key = f"{item['heading_id']}::{item['subheading_id']}"
            aliases = response.get(key)

            if aliases is None:
                continue

            results[key] = _clean_alias_list(
                aliases, original_name=item["subheading"]
            )

    return results