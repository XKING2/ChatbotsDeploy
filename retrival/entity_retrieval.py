from database.chroma_db import entity_collection

ENTITY_MAX_CANDIDATES = 3
SUBTOPIC_TOP_K = 5


class EntityRetrieval:
    def __init__(self):
        self.entity_collection = entity_collection

        entity_data = self.entity_collection.get(
            where={"type": "entity"},
            include=[
                "documents",
                "metadatas",
                "embeddings"
            ]
        )

        self.entity_ids = entity_data.get("ids", [])
        self.entity_documents = entity_data.get("documents", [])
        self.entity_metadatas = entity_data.get("metadatas", [])
        self.entity_embeddings = entity_data.get("embeddings", [])

        if (
            self.entity_ids is None
            or len(self.entity_ids) == 0
        ):
            raise ValueError(
                "Knowledge entity tidak ditemukan di ChromaDB."
            )

        self.entity_id_by_name = {}
        self.entities = []

        for entity_id, document, metadata in zip(
            self.entity_ids,
            self.entity_documents,
            self.entity_metadatas
        ):
            metadata = metadata or {}

            entity_name = (
                metadata.get("entity")
                or document
                or ""
            ).strip()

            if not entity_name:
                continue

            self.entity_id_by_name[entity_name] = entity_id

            self.entities.append({
                "entity_id": entity_id,
                "entity": entity_name,
                "name": entity_name
            })

    def get_entities(self):
        return self.entities

    def get_entity_by_id(self, entity_id):
        if not entity_id:
            return None

        for entity in self.entities:
            if entity.get("entity_id") == entity_id:
                return entity

        return None

    def get_aliases(self):
        result = self.entity_collection.get(
            where={"type": "alias"},
            include=[
                "documents",
                "metadatas"
            ]
        )

        aliases = []

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        for alias_id, document, metadata in zip(
            ids,
            documents,
            metadatas
        ):
            metadata = metadata or {}

            alias = (
                metadata.get("alias")
                or document
                or ""
            ).strip()

            entity_id = metadata.get("entity_id")

            if not alias or not entity_id:
                continue

            aliases.append({
                "alias_id": alias_id,
                "alias": alias,
                "entity_id": entity_id
            })

        return aliases

    def get_subtopics(self, entity_id=None):
        if entity_id:
            where = {
                "$and": [
                    {"type": "subtopic"},
                    {"entity_id": entity_id}
                ]
            }
        else:
            where = {
                "type": "subtopic"
            }

        result = self.entity_collection.get(
            where=where,
            include=[
                "documents",
                "metadatas"
            ]
        )

        subtopics = []

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        for subtopic_id, document, metadata in zip(
            ids,
            documents,
            metadatas
        ):
            metadata = metadata or {}

            subtopic = (
                metadata.get("subtopic")
                or document
                or ""
            ).strip()

            if not subtopic:
                continue

            subtopics.append({
                "subtopic_id": (
                    metadata.get("subtopic_id")
                    or subtopic_id
                ),
                "entity_id": metadata.get("entity_id"),
                "subtopic": subtopic
            })

        return subtopics

    def retrieve(
        self,
        query_embedding,
        previous_entity=None
    ):
        if query_embedding is None:
            return {
                "found": False,
                "status": "not_found",
                "results": [],
                "top3": []
            }

        results = []

        for entity_id, document, metadata, embedding in zip(
            self.entity_ids,
            self.entity_documents,
            self.entity_metadatas,
            self.entity_embeddings
        ):
            if embedding is None:
                continue

            score = self._cosine_similarity(
                query_embedding,
                embedding
            )

            metadata = metadata or {}

            entity_name = (
                metadata.get("entity")
                or document
                or ""
            ).strip()

            if not entity_name:
                continue

            results.append({
                "name": entity_name,
                "entity_id": entity_id,
                "score": score
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        top_results = results[:ENTITY_MAX_CANDIDATES]

        return {
            "found": bool(top_results),
            "status": "found" if top_results else "not_found",
            "results": top_results,
            "top3": top_results
        }

    def retrieve_subtopic(
        self,
        query_embedding,
        entity_id=None,
        top_k=SUBTOPIC_TOP_K
    ):
        if query_embedding is None:
            return {
                "found": False,
                "results": []
            }

        where = {
            "type": "subtopic"
        }

        if entity_id:
            where = {
                "$and": [
                    {"type": "subtopic"},
                    {"entity_id": entity_id}
                ]
            }

        try:
            result = self.entity_collection.query(
                query_embeddings=[query_embedding],
                where=where,
                n_results=top_k,
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )

        except Exception as exc:
            print(
                f"Subtopic Retrieval Error: {exc}"
            )

            return {
                "found": False,
                "results": []
            }

        ids = result.get("ids", [[]])
        documents = result.get("documents", [[]])
        metadatas = result.get("metadatas", [[]])
        distances = result.get("distances", [[]])

        ids = ids[0] if ids else []
        documents = documents[0] if documents else []
        metadatas = metadatas[0] if metadatas else []
        distances = distances[0] if distances else []

        results = []

        for item_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances
        ):
            metadata = metadata or {}

            score = (
                1.0 - float(distance)
                if distance is not None
                else 0.0
            )

            results.append({
                "subtopic_id": (
                    metadata.get("subtopic_id")
                    or item_id
                ),
                "subtopic": (
                    metadata.get("subtopic")
                    or document
                    or ""
                ).strip(),
                "entity_id": metadata.get("entity_id"),
                "score": score
            })

        return {
            "found": bool(results),
            "results": results
        }

    @staticmethod
    def _cosine_similarity(
        vector_a,
        vector_b
    ):
        dot_product = sum(
            a * b
            for a, b in zip(
                vector_a,
                vector_b
            )
        )

        norm_a = sum(
            a * a
            for a in vector_a
        ) ** 0.5

        norm_b = sum(
            b * b
            for b in vector_b
        ) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (
            norm_a * norm_b
        )