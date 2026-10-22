from database.chroma_db import chunk_collection
from retrival.semantic_search import semantic_search

CHUNK_TOP_K = 5


def _empty_result():
    return {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }


class ChunkRetrieval:

    def __init__(self):
        self.chunk_collection = chunk_collection

        if self.chunk_collection.count() == 0:
            raise ValueError(
                "Knowledge chunk tidak ditemukan di ChromaDB."
            )

    @staticmethod
    def _id_condition(field, value):
        """Satu id -> equality, beberapa id -> $in, kosong -> None."""
        if isinstance(value, (list, tuple, set)):
            values = [v for v in value if v]
            if not values:
                return None
            if len(values) == 1:
                return {field: values[0]}
            return {field: {"$in": values}}

        if value:
            return {field: value}
        return None

    def build_where(self, entity_id=None, subtopic_id=None):
        conditions = [
            condition
            for condition in (
                self._id_condition("entity_id", entity_id),
                self._id_condition("subtopic_id", subtopic_id),
            )
            if condition
        ]

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def count(self, entity_id=None, subtopic_id=None):
        where = self.build_where(entity_id, subtopic_id)

        if where is None:
            return self.chunk_collection.count()

        existing = self.chunk_collection.get(
            where=where,
            include=["metadatas"]
        )
        return len(existing.get("ids") or [])

    def retrieve(
        self,
        query_embedding,
        entity_id=None,
        subtopic_id=None,
        top_k=CHUNK_TOP_K,
        min_score=None
    ):
        """Cari chunk. entity_id/subtopic_id boleh berupa list."""
        return semantic_search(
            query_embedding=query_embedding,
            collection=self.chunk_collection,
            where=self.build_where(entity_id, subtopic_id),
            top_k=top_k,
            result_type="chunk",
            min_score=min_score
        )

    def retrieve_all(
        self,
        query_embedding,
        entity_id=None,
        subtopic_id=None
    ):
        """Semua chunk dalam entity/subtopic, urut skor, tanpa filter skor.

        Dipakai untuk menyusun konteks LLM: begitu entity/subtopic sudah
        ditentukan, seluruh isinya harus ikut, bukan hanya yang skornya tinggi.
        """
        total = self.count(entity_id, subtopic_id)

        if total == 0:
            return _empty_result()

        return self.retrieve(
            query_embedding=query_embedding,
            entity_id=entity_id,
            subtopic_id=subtopic_id,
            top_k=total,
            min_score=0.0
        )