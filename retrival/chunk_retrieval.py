from database.chroma_db import chunk_collection
from retrival.semantic_search import semantic_search

CHUNK_TOP_K = 5


class ChunkRetrieval:

    def __init__(self):
        self.collection = chunk_collection

    @staticmethod
    def build_where(heading_id=None, subheading_id=None):
        conditions = []

        for field, value in (("heading_id", heading_id),
                             ("subheading_id", subheading_id)):
            if isinstance(value, (list, tuple, set)):
                values = list(dict.fromkeys(v for v in value if v))
                if len(values) == 1:
                    conditions.append({field: values[0]})
                elif values:
                    conditions.append({field: {"$in": values}})
            elif value:
                conditions.append({field: value})

        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def count(self, heading_id=None, subheading_id=None):
        where = self.build_where(heading_id, subheading_id)
        if where is None:
            return self.collection.count()
        return len(self.collection.get(where=where, include=[]).get("ids") or [])

    def retrieve(self, query_embedding, heading_id=None, subheading_id=None,
                 top_k=CHUNK_TOP_K, min_score=None):
        return semantic_search(
            query_embedding=query_embedding,
            collection=self.collection,
            where=self.build_where(heading_id, subheading_id),
            top_k=top_k,
            result_type="chunk",
            min_score=min_score,
        )

    def get_all(self, heading_id=None, subheading_id=None):
        """
        Semua chunk satu heading/subheading, urut sesuai dokumen
        (chunk_index). Tanpa pencarian vektor, jadi lebih ringan dari
        retrieve_all(). Mengembalikan [(id, dokumen, metadata)].
        """
        where = self.build_where(heading_id, subheading_id)
        if where is None:
            return []

        data = self.collection.get(where=where, include=["documents", "metadatas"])
        rows = list(zip(data["ids"], data["documents"], data["metadatas"]))
        rows.sort(key=lambda r: ((r[2] or {}).get("chunk_index") is None,
                                 (r[2] or {}).get("chunk_index") or 0))
        return rows

    def get_all_scored(self, query_embedding, heading_id=None, subheading_id=None):
        """
        Seperti get_all(), tetapi tiap chunk diberi skor cosine terhadap
        pertanyaan (dihitung lokal dari embedding tersimpan), sehingga semua
        chunk subheading punya skor yang sebanding. [(id, dokumen, metadata, skor)]
        """
        where = self.build_where(heading_id, subheading_id)
        if where is None:
            return []

        data = self.collection.get(
            where=where, include=["documents", "metadatas", "embeddings"])
        embeddings = data.get("embeddings")
        embeddings = [] if embeddings is None else embeddings

        def cosine(vector):
            if query_embedding is None or vector is None or len(vector) == 0:
                return None
            dot = sum(float(a) * float(b) for a, b in zip(query_embedding, vector))
            na = sum(float(a) * float(a) for a in query_embedding) ** 0.5
            nb = sum(float(b) * float(b) for b in vector) ** 0.5
            return dot / (na * nb) if na and nb else None

        rows = []
        for index, (rid, doc, meta) in enumerate(
                zip(data["ids"], data["documents"], data["metadatas"])):
            vector = embeddings[index] if index < len(embeddings) else None
            rows.append((rid, doc, meta, cosine(vector)))
        rows.sort(key=lambda r: ((r[2] or {}).get("chunk_index") is None,
                                 (r[2] or {}).get("chunk_index") or 0))
        return rows

    def retrieve_all(self, query_embedding, heading_id=None, subheading_id=None):
        """Semua chunk beserta skor (dipertahankan untuk kompatibilitas)."""
        total = self.count(heading_id, subheading_id)
        if total == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]],
                    "distances": [[]]}
        return self.retrieve(query_embedding, heading_id, subheading_id,
                             top_k=total, min_score=0.0)