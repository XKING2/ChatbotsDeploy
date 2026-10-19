
import os
from google import genai

from config import (
    GEMINI_API_KEY,
    GEMINI_API_KEY_1,
    GEMINI_API_KEY_2,
    EMBEDDING_MODEL_1,
    EMBEDDING_MODEL_2,
)

# Urutan key/model yang akan dicoba.
ROUTES = [
    ("Primary", GEMINI_API_KEY_1 or GEMINI_API_KEY, EMBEDDING_MODEL_1),
    ("Backup", GEMINI_API_KEY_2, EMBEDDING_MODEL_2 or EMBEDDING_MODEL_1),
]

def create_embedding(text: str):
    if not text or not text.strip():
        return None

    seen = set()

    for name, api_key, model in ROUTES:
        if not api_key or not model:
            print(f"[EMBEDDING] {name}: key/model tidak tersedia")
            continue

        # Jangan mengulang key yang sama.
        if api_key in seen:
            continue
        seen.add(api_key)

        try:
            print(f"[EMBEDDING] Mencoba {name}: {model}")
            client = genai.Client(api_key=api_key)

            response = client.models.embed_content(
                model=model,
                contents=text,
            )

            if not response.embeddings:
                print(f"[EMBEDDING] {name}: respons kosong")
                continue

            vector = response.embeddings[0].values
            if not vector:
                continue

            print(f"[EMBEDDING] {name}: berhasil")
            return vector

        except Exception as exc:
            print(
                f"[EMBEDDING] {name} gagal: "
                f"{type(exc).__name__}: {exc}"
            )

            # 401/403 umumnya tidak akan selesai dengan retry.
            # Lanjutkan ke key cadangan.

    print("[EMBEDDING] Semua jalur gagal.")
    return None
