import os
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("EMBEDDING_MODEL_1")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY tidak terbaca dari .env")

if not model:
    raise RuntimeError("EMBEDDING_MODEL_1 tidak terbaca dari .env")

client = genai.Client(api_key=api_key)

try:
    result = client.models.embed_content(
        model=model,
        contents="Apa itu Golden Visa Indonesia?",
    )

    if not result.embeddings:
        raise RuntimeError("API tidak mengembalikan embedding")

    vector = result.embeddings[0].values

    print("Embedding berhasil.")
    print("Model:", model)
    print("Dimensi vektor:", len(vector))

except Exception as error:
    print("Embedding gagal.")
    print("Jenis error:", type(error).__name__)
    print("Detail:", error)