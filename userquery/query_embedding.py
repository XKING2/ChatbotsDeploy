
from google import genai

from config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL_1
)

from userquery.query_normalizer import normalize_query


client = genai.Client(
    api_key=GEMINI_API_KEY
)


def preprocess_query(query: str) -> str:
    return normalize_query(query)


def create_query_embedding(query: str):

    query = preprocess_query(query)

    if not query:
        return None

    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_1,
            contents=query
        )

        return response.embeddings[0].values

    except Exception as e:
        print("=" * 60)
        print("QUERY EMBEDDING ERROR")
        print(e)
        print("=" * 60)

        return None
