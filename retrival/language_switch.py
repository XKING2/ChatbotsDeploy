import re
from rapidfuzz import fuzz

LANGUAGE_NAME_TARGETS = {
    "en": ["bahasa inggris", "english"],
    "id": ["bahasa indonesia", "indonesian"],
    "zh": ["bahasa mandarin", "mandarin", "chinese", "中文"],
    "ja": ["bahasa jepang", "japanese", "日本語"],
    "ru": ["bahasa rusia", "russian"],
    "ko": ["bahasa korea", "korean"],
    "de": ["bahasa jerman", "german"],
    "fr": ["bahasa prancis", "bahasa perancis", "french"],
    "es": ["bahasa spanyol", "spanish"],
    "ar": ["bahasa arab", "arabic"],
    "th": ["bahasa thailand", "thai"],
    "vi": ["bahasa vietnam", "vietnamese"],
}

SWITCH_TRIGGER_WORDS = [
    "gunakan", "pakai", "pake", "ganti", "ubah", "rubah", "switch",
    "translate", "terjemahkan", "dalam bahasa", "in language",
]

FUZZY_THRESHOLD = 85  # toleransi typo untuk nama bahasa


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_language_switch_request(question: str) -> str | None:


    text_norm = _normalize(question)

    has_trigger = any(word in text_norm for word in SWITCH_TRIGGER_WORDS)

    if not has_trigger:
        return None

    best_lang = None
    best_score = 0

    for lang_code, target_phrases in LANGUAGE_NAME_TARGETS.items():

        for phrase in target_phrases:

            # partial_ratio: cari kecocokan substring terbaik phrase
            # di dalam kalimat, toleran typo dan tidak butuh posisi persis
            score = fuzz.partial_ratio(phrase, text_norm)

            if score >= FUZZY_THRESHOLD and score > best_score:
                best_score = score
                best_lang = lang_code

    return best_lang