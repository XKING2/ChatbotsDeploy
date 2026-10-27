import re
import unicodedata


WORD_REPLACEMENTS = {
    "infromasi": "informasi",
    "informsi": "informasi",
    "informasih": "informasi",
    "pengertianya": "pengertiannya",
    "pengertin": "pengertian",
    "goldan": "golden",
    "viza": "visa",
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
    "dr": "dari",
    "krn": "karena",
    "sy": "saya",
    "sya": "saya",
    "aja": "saja",
    "gitu": "begitu",
    "tu": "itu",
    "nih": "",
    "sih": "",
    "dong": "",
    "deh": "",
}


ENGLISH_INTENT_EXPANSIONS = {
    "extend": "perpanjang perpanjangan",
    "extending": "perpanjang perpanjangan",
    "extension": "perpanjang perpanjangan",
    "renew": "perpanjang perpanjangan",
    "renewal": "perpanjang perpanjangan",
    "prolong": "perpanjang perpanjangan",
    "apply": "pembuatan pendaftaran",
    "application": "pembuatan pendaftaran",
    "cost": "biaya tarif",
    "costs": "biaya tarif",
    "fee": "biaya tarif",
    "fees": "biaya tarif",
    "price": "biaya tarif",
    "tariff": "biaya tarif",
    "requirement": "persyaratan syarat",
    "requirements": "persyaratan syarat",
    "procedure": "prosedur mekanisme",
    "procedures": "prosedur mekanisme",
    "lost": "hilang",
    "damaged": "rusak",
}


def normalize_query(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.strip()

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    # Normalisasi tanda baca berulang.
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    # Ganti variasi kata yang sudah dikenal.
    # Pencocokan per kata agar tidak mengubah bagian kata lain.
    def replace_word(match):
        word = match.group(0)
        replacement = WORD_REPLACEMENTS.get(word.casefold())

        if replacement is None:
            return word

        return replacement

    text = re.sub(
        r"[^\W_]+(?:[-'][^\W_]+)*",
        replace_word,
        text,
        flags=re.UNICODE,
    )

    # Rapikan spasi setelah kata yang dihapus.
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Tambahkan istilah Indonesia untuk kata niat bahasa Inggris (tanpa
    # menghapus kata aslinya, dan tanpa duplikasi).
    tokens = [t.casefold() for t in re.findall(r"[^\W_]+", text, flags=re.UNICODE)]
    extras = []
    for token in tokens:
        expansion = ENGLISH_INTENT_EXPANSIONS.get(token)
        if not expansion:
            continue
        for word in expansion.split():
            if word not in tokens and word not in extras:
                extras.append(word)
    if extras:
        text = f"{text} {' '.join(extras)}"

    return text