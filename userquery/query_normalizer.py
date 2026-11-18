import re
import unicodedata


# ============================================================
# KOREKSI TYPO DAN BAHASA INFORMAL INDONESIA
# ============================================================

WORD_REPLACEMENTS = {
    # Typo umum
    "infromasi": "informasi",
    "informsi": "informasi",
    "informasih": "informasi",
    "pengertianya": "pengertiannya",
    "pengertin": "pengertian",
    "persyaratn": "persyaratan",
    "persyaratanx": "persyaratan",

    # Istilah imigrasi
    "viza": "visa",
    "pasport": "paspor",
    "passport": "paspor",

    # Bahasa informal Indonesia
    "gimana": "bagaimana",
    "gmn": "bagaimana",
    "knp": "kenapa",
    "apaan": "apa",
    "tau": "tahu",
    "udh": "sudah",
    "udah": "sudah",
    "blm": "belum",
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "enggak": "tidak",
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "krn": "karena",
    "sy": "saya",
    "sya": "saya",
    "aja": "saja",
    "gitu": "begitu",
}


# ============================================================
# NORMALISASI
# ============================================================

def normalize_query(query: str) -> str:
    """
    Normalisasi pertanyaan pengguna tanpa mengubah bahasa asli.

    Mendukung:
    - Indonesia
    - English
    - Mandarin
    - Japanese
    - Russian

    Tidak melakukan:
    - Penerjemahan otomatis
    - Penambahan ekspansi kata
    - Penghapusan kata penting
    """

    if not isinstance(query, str):
        return ""

    query = unicodedata.normalize("NFKC", query)
    query = query.strip()

    if not query:
        return ""

    # Normalisasi whitespace
    query = re.sub(r"\s+", " ", query)

    # Normalisasi karakter berulang
    query = re.sub(r"([!?.,])\1+", r"\1", query)

    # Koreksi kata berdasarkan kecocokan satu kata utuh
    def replace_word(match):
        word = match.group(0)
        replacement = WORD_REPLACEMENTS.get(word.casefold())

        if replacement:
            return replacement

        return word

    query = re.sub(
        r"[^\W_]+(?:[-'][^\W_]+)*",
        replace_word,
        query,
        flags=re.UNICODE
    )

    # Rapikan spasi sebelum tanda baca
    query = re.sub(r"\s+([,.!?;:])", r"\1", query)

    return query.strip()