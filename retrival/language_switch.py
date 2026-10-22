
import re

try:
    from rapidfuzz import fuzz

    def _similarity(a: str, b: str) -> float:
        return fuzz.ratio(a, b)

except ImportError:  # cadangan agar modul tetap bisa diuji tanpa rapidfuzz
    from difflib import SequenceMatcher

    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100


# ---------------------------------------------------------------------------
# Nama bahasa
# ---------------------------------------------------------------------------
# Nama beraksara Latin dicocokkan per kata dengan toleransi typo (fuzzy).
LANGUAGE_NAME_TARGETS = {
    "id": ["bahasa indonesia", "bahasa indo", "indonesian"],
    "en": ["bahasa inggris", "inggris", "english"],
    "zh": ["bahasa mandarin", "bahasa cina", "mandarin", "chinese"],
    "ja": ["bahasa jepang", "jepang", "japanese"],
    "ru": ["bahasa rusia", "rusia", "russian"],

}

# Nama beraksara non-Latin (Mandarin, Jepang, Korea, Rusia) dicocokkan sebagai
# potongan teks, karena tidak dipisah spasi dan kata Rusia berubah bentuk.
NATIVE_LANGUAGE_NAMES = {
    "id": ["印尼语", "印尼文", "印度尼西亚语", "インドネシア語", "индонезийск"],
    "en": ["英语", "英文", "英語", "английск"],
    "zh": ["中文", "汉语", "漢語", "华语", "普通话", "国语", "中国語", "中国语", "китайск"],
    "ja": ["日本語", "日本语", "日语", "日文", "日語", "японск"],
    "ru": ["俄语", "俄文", "俄語", "ロシア語", "русск"],
    "ko": ["韩语", "韩文", "韓国語", "한국어", "корейск"],
}

# ---------------------------------------------------------------------------
# Kata pemicu
# ---------------------------------------------------------------------------
# Latin: kata utuh ("use" tidak boleh cocok dengan "because").
SWITCH_TRIGGER_WORDS = [
    # Indonesia
    "gunakan", "pakai", "pake", "ganti", "ubah", "rubah", "terjemahkan",
    "terjemah", "translate", "switch", "jawab", "balas", "dalam bahasa",
    "ke bahasa", "pakai bahasa", "berbahasa",
    # Inggris
    "use", "answer in", "reply in", "respond in", "write in", "speak",
    "in language", "switch to", "change to", "can you", "could you", "please",
]

# Non-Latin: potongan teks.
NATIVE_TRIGGERS = [
    # Mandarin
    "用", "请", "請", "使用", "改用", "换成", "換成", "切换", "切換", "翻译",
    "翻譯", "回答", "答复", "回复", "说", "講", "讲",
    # Jepang
    "で答え", "で回答", "に翻訳", "翻訳", "でお願い", "使って", "使用", "切り替え",
    "変更", "答えて", "回答", "お願い", "ください", "教えて",
    # Rusia
    "используй", "ответь", "отвечай", "переведи", "перевод", "пожалуйста",
    "смени", "поменяй",
]

# ---------------------------------------------------------------------------
# Kata pengisi (tidak dihitung sebagai isi pertanyaan)
# ---------------------------------------------------------------------------
FILLER_WORDS = {
    # Indonesia
    "tolong", "mohon", "dong", "ya", "yah", "aja", "saja", "deh", "kak",
    "bisa", "bisakah", "dengan", "ke", "dalam", "bahasa", "untuk", "yang",
    "tadi", "itu", "ini", "informasi", "info", "jawaban", "jawab", "balas",
    "sebelumnya", "barusan", "lagi", "ulang", "dan", "atau", "di", "pada",
    "gunakan", "pakai", "pake", "ganti", "ubah", "rubah", "terjemahkan",
    "terjemah", "berbahasa", "kamu", "anda", "saya", "aku", "jadi", "menjadi",
    # Inggris
    "please", "pls", "plz", "use", "using", "in", "to", "for", "that", "this",
    "the", "it", "information", "info", "answer", "reply", "respond", "write",
    "speak", "translate", "switch", "change", "can", "could", "you", "me",
    "language", "again", "previous", "last", "above", "and", "or", "be", "a",
    "your", "my", "of", "with", "now", "just", "only", "then", "so", "also",
    "would", "will", "i", "want", "need", "like", "prefer",
    # Rusia
    "пожалуйста", "ответь", "ответьте", "отвечай", "используй", "используйте",
    "переведи", "переведите", "перевод", "на", "по", "языке", "язык", "это",
    "эту", "информацию", "информации", "ещё", "еще", "снова", "мне", "можно",
    "ты", "вы", "для", "того", "же", "смени", "поменяй", "и",
}

NATIVE_FILLERS = [
    # Mandarin
    "一下", "吧", "的", "信息", "内容", "那个", "这个", "刚才", "上面", "再",
    "重新", "以", "资料", "資料", "呢", "啊", "吗", "嗎",
    # Jepang
    "して", "その", "この", "情報", "を", "に", "で", "は", "も", "ね", "よ",
    "もう一度", "さっき", "先ほど", "について", "します", "ます", "ですか",
    "ください", "下さい", "いただけ", "頂け", "くれ", "ませんか", "教えて", "教え",
]

FUZZY_THRESHOLD = 85   # toleransi typo untuk nama bahasa
MAX_CONTENT_WORDS = 1  # sisa kata bermakna maksimal untuk dianggap perintah
SHORT_MESSAGE_WORDS = 3    # pesan sependek ini cukup nama bahasa saja
SHORT_NATIVE_CHARS = 12    # idem untuk pesan CJK (tanpa spasi)

_NATIVE_REMOVALS = sorted(
    {
        *[p for names in NATIVE_LANGUAGE_NAMES.values() for p in names],
        *NATIVE_TRIGGERS,
        *NATIVE_FILLERS,
    },
    key=len,
    reverse=True,
)
_NON_LATIN = re.compile(r"[Ѐ-ӿ぀-ヿ㐀-鿿가-힯]")


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has_trigger(text_norm: str) -> bool:
    tokens = text_norm.split()
    token_set = set(tokens)
    for word in SWITCH_TRIGGER_WORDS:
        if " " in word:
            if word in text_norm:
                return True
        elif word in token_set:
            return True
        elif len(word) >= 6 and any(word in token for token in tokens):
            # mis. "diterjemahkan" mengandung "terjemahkan"
            return True
    return any(trigger in text_norm for trigger in NATIVE_TRIGGERS)


def _match_native_language(text_norm: str):
    """Nama bahasa beraksara non-Latin; yang paling panjang cocok menang."""
    best_lang, best_name = None, ""
    for lang_code, names in NATIVE_LANGUAGE_NAMES.items():
        for name in names:
            if name in text_norm and len(name) > len(best_name):
                best_lang, best_name = lang_code, name
    return best_lang, best_name


def _match_latin_language(text_norm: str):
    """Nama bahasa Latin per kata dan per pasangan kata (bukan substring)."""
    tokens = text_norm.split()
    candidates = list(tokens) + [
        f"{a} {b}" for a, b in zip(tokens, tokens[1:])
    ]

    best_lang, best_score, matched_phrase = None, 0.0, None
    for lang_code, phrases in LANGUAGE_NAME_TARGETS.items():
        for phrase in phrases:
            phrase_norm = _normalize(phrase)
            for candidate in candidates:
                # Nama pendek (<=4 huruf) harus persis, supaya "thai" tidak
                # cocok dengan "that".
                if len(phrase_norm) <= 4:
                    score = 100.0 if candidate == phrase_norm else 0.0
                else:
                    score = _similarity(phrase_norm, candidate)
                if score >= FUZZY_THRESHOLD and score > best_score:
                    best_lang, best_score, matched_phrase = (
                        lang_code, score, phrase_norm
                    )
    return best_lang, matched_phrase


def _content_word_count(text_norm: str, language_phrase: str) -> int:
    """Jumlah kata bermakna yang tersisa setelah pemicu/pengisi/bahasa dibuang."""
    language_tokens = set((language_phrase or "").split())
    count = 0
    for token in text_norm.split():
        if token in FILLER_WORDS or token in language_tokens:
            continue
        if _NON_LATIN.search(token):
            remainder = token
            for piece in _NATIVE_REMOVALS:
                remainder = remainder.replace(piece, "")
            # Kalimat CJK tidak dipisah spasi, jadi hitung dari jumlah huruf:
            # sisa 3 huruf atau lebih dianggap isi pertanyaan sungguhan.
            if len(remainder) >= 3:
                count += len(remainder) // 2 + 1
            continue
        count += 1
    return count


def detect_language_switch_request(question: str) -> "str | None":
    """Kode bahasa bila kalimat adalah perintah ganti bahasa, selain itu None."""
    text_norm = _normalize(question)
    if not text_norm:
        return None

    lang_code, matched = _match_native_language(text_norm)
    if lang_code is None:
        lang_code, matched = _match_latin_language(text_norm)
    if lang_code is None:
        return None

    # Tanpa kata pemicu, nama bahasa harus berdiri hampir sendirian,
    # mis. "bahasa inggris" atau "in english".
    if not _has_trigger(text_norm):
        is_native = bool(_NON_LATIN.search(text_norm))
        short = (
            len(text_norm.replace(" ", "")) <= SHORT_NATIVE_CHARS
            if is_native else len(text_norm.split()) <= SHORT_MESSAGE_WORDS
        )
        if not short:
            return None

    # Pertanyaan sungguhan (masih ada kata bermakna) bukan perintah ganti bahasa.
    if _content_word_count(text_norm, matched) > MAX_CONTENT_WORDS:
        return None

    return lang_code