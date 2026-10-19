import json

from google import genai
from config import GEMINI_API_KEY, LLM_MODEL


client = genai.Client(
    api_key=GEMINI_API_KEY
)

ALIAS_LANGUAGES = [
    "Bahasa Indonesia",
    "English",
    "Mandarin Chinese",
    "Japanese",
    "Russian",
]

BATCH_SIZE = 15


def generate_aliases_batch(topic_names):
    if not topic_names:
        return {}

    all_aliases = {}

    for i in range(0, len(topic_names), BATCH_SIZE):

        batch = topic_names[i:i + BATCH_SIZE]

        print(
            f"Batch alias "
            f"{i + 1}-{i + len(batch)} "
            f"dari {len(topic_names)}"
        )

        batch_result = _generate_batch(batch)

        if batch_result is None:
            for name in batch:
                all_aliases[name] = []
        else:
            all_aliases.update(batch_result)

    return all_aliases


def _generate_batch(topic_names):

    topic_list_str = "\n".join(
        f'{idx + 1}. "{name}"'
        for idx, name in enumerate(topic_names)
    )

    prompt = f"""Kamu adalah asisten yang membuat daftar alias/sebutan alternatif
untuk beberapa jenis layanan keimigrasian.

DAFTAR LAYANAN:
{topic_list_str}

Untuk setiap layanan, buat alias yang mencakup:
- Singkatan umum Bahasa Indonesia jika benar-benar spesifik.
- Nama sehari-hari yang merujuk spesifik ke layanan.
- Terjemahan istilah dalam:
{", ".join(ALIAS_LANGUAGES)}

ATURAN:
1. Jangan memberikan singkatan spesifik seperti KITAS, KITAP, VOA,
   atau ITK kepada layanan yang bersifat umum atau payung.
2. Jika terdapat layanan yang mirip, alias harus tetap membedakan
   masing-masing layanan.
3. Untuk layanan umum, gunakan istilah yang menggambarkan cakupannya.
4. Jika ragu apakah alias terlalu umum atau berpotensi bentrok,
   jangan masukkan alias tersebut.
5. Jangan masukkan nama asli layanan sebagai alias.

Jawab HANYA dalam JSON object.

Key harus sama persis dengan nama layanan.

Contoh:
{{
  "Nama Layanan A": ["alias 1", "alias 2"],
  "Nama Layanan B": ["alias 1", "alias 2"]
}}
"""

    try:

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )

        text = response.text.strip()

        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        result = json.loads(text)

        if not isinstance(result, dict):
            return None

        cleaned = {}

        for name in topic_names:

            aliases = result.get(name, [])

            if isinstance(aliases, list):

                cleaned[name] = [
                    alias.strip()
                    for alias in aliases
                    if isinstance(alias, str)
                    and alias.strip()
                ]

            else:
                cleaned[name] = []

        return cleaned

    except Exception as e:

        print(
            f"[AliasGenerator] "
            f"Gagal generate alias: {e}"
        )

        return None