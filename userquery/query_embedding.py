from userquery.query_normalizer import normalize_query
from LLM.embedding_service import create_embedding


# ============================================================
# KONFIGURASI QUERY EMBEDDING
# ============================================================

QUERY_TASK_PREFIX = "task: question answering | query:"


# ============================================================
# PREPROCESS QUERY
# ============================================================

def preprocess_query(query: str) -> str:
    """
    Melakukan normalisasi query sebelum embedding.
    Bahasa asli pengguna tetap dipertahankan.
    """

    return normalize_query(query)


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_query_embedding(query: str):
    """
    Membuat embedding untuk pertanyaan pengguna.

    Menggunakan format query Gemini Embedding 2:
    task: question answering | query: {query}

    Return:
        list[float] : jika berhasil
        None        : jika query kosong atau embedding gagal
    """

    normalized_query = preprocess_query(query)

    if not normalized_query:
        return None

    embedding_text = (
        f"{QUERY_TASK_PREFIX} {normalized_query}"
    )

    try:
        embedding = create_embedding(embedding_text)

        if embedding is None:
            print("[QUERY EMBEDDING] Gagal membuat embedding.")
            return None

        return embedding

    except Exception as error:
        print(
            f"[QUERY EMBEDDING] Error: "
            f"{type(error).__name__}: {error}"
        )
        return None