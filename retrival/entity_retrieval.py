from database.chroma_db import entity_collection
from retrival.semantic_search import semantic_search

ENTITY_MAX_CANDIDATES = 3
SUBTOPIC_TOP_K = 5


class EntityRetrieval:
    """Akses data entity, alias, dan subtopic di entity_collection.

    Semua pencarian semantik lewat semantic_search (ChromaDB), tanpa
    menghitung cosine sendiri dan tanpa memuat seluruh embedding ke memori.
    Entity, alias, dan subtopic dimuat sekali saat inisialisasi.
    """

    def __init__(self):
        self.entity_collection = entity_collection

        entity_data = self.entity_collection.get(
            where={"type": "entity"},
            include=["documents", "metadatas"]
        )

        self.entity_ids = entity_data.get("ids") or []
        self.entity_documents = entity_data.get("documents") or []
        self.entity_metadatas = entity_data.get("metadatas") or []

        if len(self.entity_ids) == 0:
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

        self._aliases = self._load_aliases()
        self._subtopics = self._load_subtopics()

    # ------------------------------------------------------------ loading
    def _load_aliases(self):
        result = self.entity_collection.get(
            where={"type": "alias"},
            include=["documents", "metadatas"]
        )

        aliases = []

        for alias_id, document, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or []
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
                "entity_id": entity_id,
                "subtopic_id": metadata.get("subtopic_id")
            })

        return aliases

    def _load_subtopics(self):
        result = self.entity_collection.get(
            where={"type": "subtopic"},
            include=["documents", "metadatas"]
        )

        subtopics = []

        for item_id, document, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or []
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
                "subtopic_id": metadata.get("subtopic_id") or item_id,
                "entity_id": metadata.get("entity_id"),
                "subtopic": subtopic
            })

        return subtopics

    # ------------------------------------------------------------ getters
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
        return list(self._aliases)

    def get_subtopics(self, entity_id=None):
        if not entity_id:
            return list(self._subtopics)

        return [
            item for item in self._subtopics
            if item.get("entity_id") == entity_id
        ]

    # ------------------------------------------------------ semantic search
    def retrieve(
        self,
        query_embedding,
        previous_entity=None,
        top_k=ENTITY_MAX_CANDIDATES
    ):
        if query_embedding is None:
            return {
                "found": False,
                "status": "not_found",
                "results": [],
                "top3": []
            }

        raw = semantic_search(
            query_embedding=query_embedding,
            collection=self.entity_collection,
            where={"type": "entity"},
            top_k=top_k,
            result_type="entity"
        )

        results = []

        for item_id, document, metadata, distance in self._rows(raw):
            entity_name = (
                metadata.get("entity")
                or document
                or ""
            ).strip()

            if not entity_name:
                continue

            results.append({
                "name": entity_name,
                "entity_id": metadata.get("entity_id") or item_id,
                "score": self._score(distance)
            })

        return {
            "found": bool(results),
            "status": "found" if results else "not_found",
            "results": results,
            "top3": results[:3]
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

        where = {"type": "subtopic"}

        if entity_id:
            where = {
                "$and": [
                    {"type": "subtopic"},
                    {"entity_id": entity_id}
                ]
            }

        raw = semantic_search(
            query_embedding=query_embedding,
            collection=self.entity_collection,
            where=where,
            top_k=top_k,
            result_type="subtopic"
        )

        results = []

        for item_id, document, metadata, distance in self._rows(raw):
            results.append({
                "subtopic_id": metadata.get("subtopic_id") or item_id,
                "subtopic": (
                    metadata.get("subtopic")
                    or document
                    or ""
                ).strip(),
                "entity_id": metadata.get("entity_id"),
                "score": self._score(distance)
            })

        return {
            "found": bool(results),
            "results": results
        }

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _rows(raw):
        """Ubah hasil Chroma (nested list) menjadi baris datar."""
        def first(key):
            value = (raw or {}).get(key)
            if value is None or len(value) == 0:
                return []
            return value[0]

        for item_id, document, metadata, distance in zip(
            first("ids"),
            first("documents"),
            first("metadatas"),
            first("distances")
        ):
            yield item_id, document, (metadata or {}), distance

    @staticmethod
    def _score(distance):
        return 1.0 - float(distance) if distance is not None else 0.0