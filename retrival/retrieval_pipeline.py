from database.chroma_db import entity_collection, chunk_collection
from retrival.context_understanding import ContextUnderstanding
from retrival.semantic_search import semantic_search
from userquery.query_embedding import create_query_embedding
from userquery.query_normalizer import normalize_query


ENTITY_TOP_K = 5
SUBTOPIC_TOP_K = 5
CHUNK_TOP_K = 5


class RetrievalPipeline:
    def __init__(self, context_engine=None):
        self.context_engine = context_engine or ContextUnderstanding()
        self.entity_collection = entity_collection
        self.chunk_collection = chunk_collection
        self.entity_map = self._build_entity_map()

    def _build_entity_map(self):
        result = self.entity_collection.get(
            where={"type": "entity"},
            include=["documents", "metadatas"],
        )
        entity_map = {}
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        for entity_id, document, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            entity_name = (metadata.get("entity") or document or "").strip()
            if entity_id and entity_name:
                entity_map[entity_id] = entity_name
        return entity_map

    def _get_entity_name(self, entity_id):
        return self.entity_map.get(entity_id) if entity_id else None

    def _convert_results(self, result, result_type):
        if not result:
            return []

        ids = result.get("ids") or [[]]
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        ids = ids[0] if ids else []
        documents = documents[0] if documents else []
        metadatas = metadatas[0] if metadatas else []
        distances = distances[0] if distances else []

        results = []
        for item_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            score = 1.0 - float(distance) if distance is not None else 0.0
            entity_id = metadata.get("entity_id")
            entity_name = metadata.get("entity") or self._get_entity_name(entity_id)
            subtopic_id = metadata.get("subtopic_id")
            subtopic = metadata.get("subtopic")

            results.append({
                "type": result_type,
                "id": item_id,
                "score": score,
                "content": document or "",
                "entity_id": entity_id,
                "entity": entity_name,
                "subtopic_id": subtopic_id,
                "subtopic": subtopic,
                "content_chunk_id": (
                    metadata.get("content_chunk_id") or item_id
                    if result_type == "chunk" else None
                ),
                "document": metadata.get("document"),
                "category": metadata.get("category"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "chunk_index": metadata.get("chunk_index"),
                "total_chunks": metadata.get("total_chunks"),
                "description": metadata.get("description"),
                "examples": metadata.get("examples"),
            })
        return results

    @staticmethod
    def _rank_candidates(entity_results, subtopic_results, chunk_results):
        candidates = entity_results + subtopic_results + chunk_results
        candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return candidates

    @staticmethod
    def _empty_result(status="not_found", context=None):
        return {
            "found": False,
            "status": status,
            "context": context,
            "entity_results": [],
            "subtopic_results": [],
            "chunk_results": [],
            "results": [],
            "top3": [],
            "best_candidate": None,
        }

    def retrieve(self, question, previous_entity=None, top_k=CHUNK_TOP_K):
        if not question or not question.strip():
            return self._empty_result()

        # One normalization and one query embedding for all three searches.
        normalized_question = normalize_query(question)
        query_embedding = create_query_embedding(normalized_question)
        if query_embedding is None:
            return self._empty_result(status="embedding_failed")

        entity_raw = semantic_search(
            query_embedding=query_embedding,
            collection=self.entity_collection,
            where={"type": "entity"},
            top_k=ENTITY_TOP_K,
            result_type="entity",
        )
        subtopic_raw = semantic_search(
            query_embedding=query_embedding,
            collection=self.entity_collection,
            where={"type": "subtopic"},
            top_k=SUBTOPIC_TOP_K,
            result_type="subtopic",
        )
        chunk_raw = semantic_search(
            query_embedding=query_embedding,
            collection=self.chunk_collection,
            top_k=top_k,
            result_type="chunk",
        )

        entity_results = self._convert_results(entity_raw, "entity")
        subtopic_results = self._convert_results(subtopic_raw, "subtopic")
        chunk_results = self._convert_results(chunk_raw, "chunk")

        # ContextUnderstanding resolves context from these results; it does not
        # run EntityRetrieval semantic search or create another embedding.
        context = self.context_engine.analyze(
            question=question,
            previous_entity=previous_entity,
            normalized_question=normalized_question,
            query_embedding=query_embedding,
            entity_results=entity_results,
            subtopic_results=subtopic_results,
            chunk_results=chunk_results,
        )

        # Keep the raw semantic matches for debugging, but never treat a raw
        # ChromaDB match as proof that the question belongs to the domain.
        raw_candidates = self._rank_candidates(
            entity_results, subtopic_results, chunk_results
        )
        domain_relevant = bool(context.get("domain_relevant"))
        resolved_entity_id = context.get("entity_id")

        # Once an entity is resolved, only expose content belonging to that
        # entity to the answer-generation stage. This prevents a high-scoring
        # chunk from a different service from hijacking the answer.
        if domain_relevant and resolved_entity_id:
            scoped_entity_results = [
                item for item in entity_results
                if item.get("entity_id") == resolved_entity_id
            ]
            scoped_subtopic_results = [
                item for item in subtopic_results
                if item.get("entity_id") == resolved_entity_id
            ]
            scoped_chunk_results = [
                item for item in chunk_results
                if item.get("entity_id") == resolved_entity_id
            ]
            candidates = self._rank_candidates(
                scoped_entity_results,
                scoped_subtopic_results,
                scoped_chunk_results,
            )
        else:
            candidates = []

        best_candidate = candidates[0] if candidates else None
        found = bool(domain_relevant and candidates)
        if found:
            status = "found"
        elif not domain_relevant:
            status = "out_of_domain"
        else:
            status = "not_found"

        return {
            # found is the only flag the production answer stage should trust.
            "found": found,
            "status": status,
            "domain_relevant": domain_relevant,
            "retrieval_found": bool(raw_candidates),
            "context": context,
            "entity_results": entity_results,
            "subtopic_results": subtopic_results,
            "chunk_results": chunk_results,
            # Relevant, entity-scoped candidates for answer generation.
            "results": candidates,
            "top3": candidates[:3],
            "best_candidate": best_candidate,
            # Unfiltered matches retained for debugging only.
            "raw_results": raw_candidates,
        }
