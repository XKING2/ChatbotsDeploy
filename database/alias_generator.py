import json
import re
import time
import threading

from google import genai
import config
from config import GEMINI_API_KEY, LLM_MODEL


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY_2 = getattr(config, "GEMINI_API_KEY_2", None)

API_KEYS = [
    key.strip()
    for key in (GEMINI_API_KEY, GEMINI_API_KEY_2)
    if isinstance(key, str) and key.strip()
]

# Maksimal 20 item alias dalam satu request API.
BATCH_SIZE = 20

# Setelah 5 percobaan request API, tunggu 90 detik.
MAX_CALLS_BEFORE_COOLDOWN = 5
COOLDOWN_SECONDS = 90

ALIAS_LANGUAGES = [
    "Bahasa Indonesia",
    "English",
    "Mandarin Chinese",
    "Japanese",
    "Russian",
]

# State dipakai bersama oleh generator entity dan subtopic.
_call_count = 0
_call_lock = threading.Lock()
_cooldown_lock = threading.Lock()

if not API_KEYS:
    raise ValueError(
        "GEMINI_API_KEY belum diisi di config.py"
    )


# ============================================================
# API CLIENT / FAILOVER
# ============================================================

def _clean_json_response(text):
    """Membersihkan JSON dari markdown code fence."""
    text = (text or "").strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def _wait_if_needed():
    """
    Menunggu 90 detik setiap 5 percobaan API.
    Lock mencegah beberapa thread memulai cooldown bersamaan.
    """
    global _call_count

    with _cooldown_lock:
        with _call_lock:
            should_wait = (
                _call_count >= MAX_CALLS_BEFORE_COOLDOWN
            )

            if should_wait:
                _call_count = 0

        if should_wait:
            print(
                f"[AliasGenerator] Cooldown "
                f"{COOLDOWN_SECONDS} detik setelah "
                f"{MAX_CALLS_BEFORE_COOLDOWN} percobaan API."
            )
            time.sleep(COOLDOWN_SECONDS)


def _generate_content_with_failover(prompt):
    """
    Mencoba API key secara bergantian jika terjadi error.

    Setiap percobaan request dihitung, termasuk request yang gagal.
    """
    global _call_count

    _wait_if_needed()

    errors = []

    for key_index, api_key in enumerate(API_KEYS):
        try:
            with _call_lock:
                _call_count += 1
                current_count = _call_count

            print(
                f"[AliasGenerator] Request API "
                f"ke-{current_count} menggunakan key "
                f"{key_index + 1}/{len(API_KEYS)}"
            )

            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )

            text = _clean_json_response(response.text)

            if not text:
                raise ValueError("Respons Gemini kosong.")

            return text

        except Exception as exc:
            errors.append(
                f"Key {key_index + 1}: {exc}"
            )

            print(
                f"[AliasGenerator] Key {key_index + 1} "
                f"gagal: {exc}"
            )

            # Key berikutnya dicoba secara otomatis.
            # Cooldown akan diterapkan sebelum request selanjutnya.

    print(
        "[AliasGenerator] Semua API key gagal: "
        + " | ".join(errors)
    )

    return None


# ============================================================
# ENTITY ALIASES
# ============================================================

def generate_aliases_batch(topic_names):
    """
    Membuat alias untuk entity/layanan.

    Input:
        ["Layanan A", "Layanan B"]

    Output:
        {
            "Layanan A": ["alias 1", "alias 2"],
            "Layanan B": ["alias 3"]
        }
    """
    if not topic_names:
        return {}

    all_aliases = {}

    for i in range(0, len(topic_names), BATCH_SIZE):
        batch = topic_names[i:i + BATCH_SIZE]

        print(
            f"[AliasGenerator] Batch entity "
            f"{i + 1}-{i + len(batch)} "
            f"dari {len(topic_names)}"
        )

        batch_result = _generate_entity_batch(batch)

        # Jika request gagal, nama TIDAK dimasukkan ke hasil.
        # Pemanggil membedakan "gagal" (tidak ada key, dicoba lagi
        # pada run berikutnya) dari "tidak ada alias" (list kosong).
        if batch_result is not None:
            all_aliases.update(batch_result)

    return all_aliases


def _generate_entity_batch(topic_names):
    topic_list_str = "\n".join(
        f'{idx + 1}. "{name}"'
        for idx, name in enumerate(topic_names)
    )

    prompt = f"""
Kamu adalah asisten pembuat alias untuk layanan keimigrasian.

DAFTAR LAYANAN:
{topic_list_str}

Untuk setiap layanan, buat alias yang mencakup:
- Singkatan umum Bahasa Indonesia jika benar-benar spesifik.
- Nama sehari-hari yang merujuk spesifik ke layanan.
- Istilah relevan dalam:
{", ".join(ALIAS_LANGUAGES)}

ATURAN:
1. Jangan memberikan singkatan seperti KITAS, KITAP, VOA,
   atau ITK kepada layanan umum yang tidak spesifik.
2. Alias harus membedakan layanan yang mirip.
3. Hindari istilah terlalu umum atau berpotensi bentrok.
4. Jangan memasukkan nama asli layanan sebagai alias.
5. Buat maksimal 20 alias per layanan. Utamakan kualitas.
6. Gunakan alias yang relevan untuk pencocokan maksud pengguna.
7. Jangan mengarang prosedur atau fakta keimigrasian.

Jawab HANYA dengan JSON object valid.
Setiap key harus sama persis dengan nama layanan.

Contoh:
{{
  "Nama Layanan A": ["alias 1", "alias 2"],
  "Nama Layanan B": ["alias 1", "alias 2"]
}}
"""

    text = _generate_content_with_failover(prompt)

    if text is None:
        return None

    try:
        result = json.loads(text)

        if not isinstance(result, dict):
            return None

        cleaned = {}

        for name in topic_names:
            if name not in result:
                continue

            aliases = result[name]

            if not isinstance(aliases, list):
                continue

            cleaned[name] = _clean_alias_list(
                aliases,
                original_name=name,
            )

        return cleaned

    except (json.JSONDecodeError, TypeError) as exc:
        print(
            f"[AliasGenerator] JSON entity tidak valid: {exc}"
        )
        return None


# ============================================================
# SUBTOPIC ALIASES
# ============================================================

def generate_subtopic_aliases_batch(subtopics):
    """
    Membuat alias subtopic secara batch.

    Input:
        [
            {
                "entity_id": "ent_001",
                "entity": "Layanan Izin Tinggal",
                "subtopic_id": "sub_001",
                "subtopic": "Perpanjangan Izin Tinggal"
            }
        ]

    Output:
        {
            "ent_001::sub_001": ["alias 1", "alias 2"]
        }
    """
    if not subtopics:
        return {}

    all_aliases = {}

    for i in range(0, len(subtopics), BATCH_SIZE):
        batch = subtopics[i:i + BATCH_SIZE]

        print(
            f"[AliasGenerator] Batch subtopic "
            f"{i + 1}-{i + len(batch)} "
            f"dari {len(subtopics)}"
        )

        batch_result = _generate_subtopic_batch(batch)

        for item in batch:
            key = (
                f'{item["entity_id"]}::'
                f'{item["subtopic_id"]}'
            )

            # Gagal / key tidak dikembalikan Gemini -> tidak dimasukkan.
            if batch_result is not None and key in batch_result:
                all_aliases[key] = batch_result[key]

    return all_aliases


def _generate_subtopic_batch(subtopics):
    """
    Satu request menangani maksimal BATCH_SIZE subtopic.
    Identitas entity dan subtopic dipakai untuk menghindari bentrok.
    """
    items = []

    for item in subtopics:
        items.append({
            "key": (
                f'{item["entity_id"]}::'
                f'{item["subtopic_id"]}'
            ),
            "entity": item["entity"].strip(),
            "subtopic": item["subtopic"].strip(),
        })

    prompt = f"""
Kamu membuat alias untuk context understanding chatbot
layanan keimigrasian.

Buat alias untuk setiap pasangan ENTITY dan SUBTOPIC berikut:
{json.dumps(items, ensure_ascii=False, indent=2)}

Bahasa alias:
{", ".join(ALIAS_LANGUAGES)}

ATURAN:
1. Setiap key output harus sama persis dengan key input.
2. Buat sekitar 5-15 alias yang berkualitas untuk setiap subtopic.
3. Alias harus merujuk spesifik pada subtopic tersebut.
4. Sertakan variasi pertanyaan sehari-hari yang wajar.
5. Sertakan sinonim atau terjemahan relevan dalam bahasa yang diminta.
6. Jangan memasukkan nama asli subtopic sebagai alias.
7. Hindari alias generik dan alias yang dapat merujuk ke subtopic lain.
8. Jangan mengarang prosedur, syarat, biaya, atau fakta.
9. Jika tidak yakin, kembalikan array kosong.
10. Jangan menghasilkan lebih dari 20 alias untuk satu subtopic.

Jawab HANYA dengan JSON object valid dalam format:
{{
  "entity_id::subtopic_id": ["alias 1", "alias 2"]
}}
"""

    text = _generate_content_with_failover(prompt)

    if text is None:
        return None

    try:
        result = json.loads(text)

        if not isinstance(result, dict):
            return None

        cleaned = {}

        for item in items:
            key = item["key"]

            if key not in result:
                continue

            aliases = result[key]

            if not isinstance(aliases, list):
                continue

            cleaned[key] = _clean_alias_list(
                aliases,
                original_name=item["subtopic"],
            )[:20]

        return cleaned

    except (json.JSONDecodeError, TypeError) as exc:
        print(
            f"[AliasGenerator] JSON subtopic tidak valid: {exc}"
        )
        return None


# ============================================================
# ALIAS CLEANER
# ============================================================

def _clean_alias_list(aliases, original_name=None):
    cleaned = []

    seen = set()

    if original_name:
        seen.add(original_name.strip().casefold())

    for alias in aliases:
        if not isinstance(alias, str):
            continue

        alias = alias.strip()
        normalized = alias.casefold()

        if alias and normalized not in seen:
            cleaned.append(alias)
            seen.add(normalized)

    return cleaned[:20]