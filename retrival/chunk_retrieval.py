from database.chroma_db import chunk_collection
from retrival.semantic_search import semantic_search

CHUNK_TOP_K = 5


class ChunkRetrieval:

    def __init__(self):
        self.chunk_collection = chunk_collection

        if self.chunk_collection.count() == 0:
            raise ValueError(
                "Knowledge chunk tidak ditemukan di ChromaDB."
            )

    def retrieve(
        self,
        query_embedding,
        entity_id=None,
        subtopic_id=None,
        top_k=CHUNK_TOP_K
    ):
        conditions = []

        if entity_id:
            conditions.append({
                "entity_id": entity_id
            })

        if subtopic_id:
            conditions.append({
                "subtopic_id": subtopic_id
            })

        if len(conditions) == 0:
            where = None
        elif len(conditions) == 1:
            where = conditions[0]
        else:
            where = {
                "$and": conditions
            }

        return semantic_search(
            query_embedding=query_embedding,
            collection=self.chunk_collection,
            where=where,
            top_k=top_k,
            result_type="chunk"
        )