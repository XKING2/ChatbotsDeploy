import re


def clean_text(text: str) -> str:

    if not text:
        return ""

    # Buang karakter non-breaking space & artefak umum hasil parsing PDF
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")

    # Normalisasi spasi berlebih
    text = re.sub(r"[ \t]+", " ", text)

    text = text.strip()

    return text


def preprocess_blocks(blocks: list) -> list:


    cleaned = []

    for block in blocks:

        text = clean_text(block["text"])

        if not text:
            continue

        new_block = dict(block)
        new_block["text"] = text

        cleaned.append(new_block)

    return cleaned