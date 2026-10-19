
from userquery.query_normalizer import normalize_query
from LLM.embedding_service import create_embedding


def preprocess_query(query: str) -> str:
    return normalize_query(query)


def create_query_embedding(query: str):
    query = preprocess_query(query)

    if not query:
        return None

    return create_embedding(query)
