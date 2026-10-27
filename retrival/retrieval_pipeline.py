import time

from database.chroma_db import entity_collection
from retrival.chunk_retrieval import ChunkRetrieval
from retrival.context_understanding import ContextUnderstanding
from retrival.semantic_search import semantic_search
from userquery.query_embedding import create_query_embedding
from userquery.query_normalizer import normalize_query


ENTITY_TOP_K = 5
SUBTOPIC_TOP_K = 5
CHUNK_TOP_K = 5

# Alias retrieval: aliases (multilingual) are embedded in entity_collection with
# type="alias". They are searched as extra evidence for entity and subtopic.
ALIAS_TOP_K = 30                # banyak alias dibandingkan sekaligus (multi-vektor)
ALIAS_RANK_K = 10               # konstanta reciprocal-rank; hanya untuk info debug alias_support
ALIAS_SCOPED_ENTITY_LIMIT = 3   # chunk search tambahan dibatasi ke N entity teratas
MAX_ENTITY_CANDIDATES = 8
MAX_SUBTOPIC_CANDIDATES = 10

# Context assembly for the LLM (tidak mengubah pemilihan entity/subtopic).
ENTITY_CHUNK_TOP_K = 6      # tanpa subtopic: ambil N chunk terbaik dalam entity
EXTRA_CHUNK_TOP_K = 2       # dengan subtopic: tambahan chunk terbaik dari subtopic lain
MAX_RELEVANT_CHUNKS = 12    # batas aman panjang prompt

# Catatan: TIDAK ada angka minimum skor di file ini. Penyaringan skor hanya
# dilakukan di retrival/semantic_search.py (SEMANTIC_MIN_SCORES).


class RetrievalPipeline:
    def __init__(self, context_engine=None, chunk_retrieval=None):
        self.context_engine = context_engine or ContextUnderstanding()
        self.chunk_retrieval = chunk_retrieval or ChunkRetrieval()
        self.entity_collection = entity_collection
        self.chunk_collection = self.chunk_retrieval.chunk_collection
        self.entity_map = self._build_entity_map()
        self.subtopic_map = self._build_subtopic_map()

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

    def _build_subtopic_map(self):
        result = self.entity_collection.get(
            where={"type": "subtopic"},
            include=["documents", "metadatas"],
        )
        subtopic_map = {}
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        for item_id, document, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            subtopic_id = metadata.get("subtopic_id") or item_id
            name = (metadata.get("subtopic") or document or "").strip()
            if subtopic_id and name:
                subtopic_map[subtopic_id] = {
                    "name": name,
                    "entity_id": metadata.get("entity_id"),
                }
        return subtopic_map

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
    def _blank_item(result_type, item_id, score, entity_id, entity,
                    subtopic_id=None, subtopic=None):
        return {
            "type": result_type,
            "id": item_id,
            "score": score,
            "content": "",
            "entity_id": entity_id,
            "entity": entity,
            "subtopic_id": subtopic_id,
            "subtopic": subtopic,
            "content_chunk_id": None,
            "document": None,
            "category": None,
            "page_start": None,
            "page_end": None,
            "chunk_index": None,
            "total_chunks": None,
            "description": None,
            "examples": None,
        }

    def _alias_subtopic_id(self, item_id, metadata):
        """Find the subtopic an alias belongs to (None for entity-level aliases)."""
        subtopic_id = metadata.get("subtopic_id")
        if subtopic_id:
            return subtopic_id
        if isinstance(item_id, str) and "::alias::" in item_id:
            prefix = item_id.split("::alias::", 1)[0]
            if prefix in self.subtopic_map:
                return prefix
        return None

    def _convert_alias_results(self, result):
        """Turn alias hits into entity-level and subtopic-level candidates.

        Every alias is evidence for its entity. An alias that belongs to a
        subtopic is also evidence for that subtopic.
        """
        entity_items, subtopic_items, hits = [], [], []
        if not result:
            return entity_items, subtopic_items, hits

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for item_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = metadata or {}
            entity_id = metadata.get("entity_id")
            if not entity_id:
                continue
            score = 1.0 - float(distance) if distance is not None else 0.0
            alias = metadata.get("alias") or document or ""
            entity_name = self._get_entity_name(entity_id)
            subtopic_id = self._alias_subtopic_id(item_id, metadata)
            subtopic_info = self.subtopic_map.get(subtopic_id) if subtopic_id else None

            entity_item = self._blank_item(
                "entity", entity_id, score, entity_id, entity_name
            )
            entity_item["source"] = "alias"
            entity_item["matched_alias"] = alias
            entity_items.append(entity_item)

            subtopic_name = subtopic_info["name"] if subtopic_info else None
            if subtopic_id and subtopic_name:
                subtopic_item = self._blank_item(
                    "subtopic", subtopic_id, score, entity_id, entity_name,
                    subtopic_id, subtopic_name,
                )
                subtopic_item["source"] = "alias"
                subtopic_item["matched_alias"] = alias
                subtopic_items.append(subtopic_item)

            hits.append({
                "alias": alias,
                "entity": entity_name,
                "subtopic": subtopic_name,
                "score": score,
            })
        return (
            self._aggregate_alias_support(entity_items, "entity_id"),
            self._aggregate_alias_support(subtopic_items, "subtopic_id"),
            hits,
        )

    @staticmethod
    def _aggregate_alias_support(items, key):
        """One candidate per entity/subtopic, keeping its best alias score.

        `alias_support` / `alias_hit_count` are diagnostic only (shown by the
        debug test). They do not influence which entity or subtopic is chosen.
        """
        groups = {}
        for rank, item in enumerate(items, start=1):
            # items are in ranking order (best alias first).
            k = item.get(key)
            if not k:
                continue
            groups.setdefault(k, []).append((rank, item))

        aggregated = []
        for entries in groups.values():
            best_item = max(entries, key=lambda e: e[1].get("score", 0.0))[1]
            best_item = dict(best_item)
            best_item["alias_support"] = sum(
                1.0 / (ALIAS_RANK_K + rank) for rank, _ in entries
            )
            best_item["alias_hit_count"] = len(entries)
            aggregated.append(best_item)
        return aggregated

    @staticmethod
    def _merge_best(primary, extra, key, limit):
        """Merge two candidate lists, keep the best score per key.

        Evidence fields from the losing item (alias_support) are kept.
        """
        best = {}
        support = {}
        counts = {}
        for item in list(primary) + list(extra):
            k = item.get(key)
            if not k:
                continue
            if item.get("alias_support"):
                support[k] = max(support.get(k, 0.0), item["alias_support"])
                counts[k] = max(counts.get(k, 0), item.get("alias_hit_count", 0))
            if k not in best or item.get("score", 0.0) > best[k].get("score", 0.0):
                best[k] = item
        merged = []
        for k, item in best.items():
            if k in support:
                item = dict(item)
                item["alias_support"] = support[k]
                item["alias_hit_count"] = counts[k]
            merged.append(item)
        merged.sort(key=lambda it: it.get("score", 0.0), reverse=True)
        return merged[:limit]

    @staticmethod
    def _rank_candidates(entity_results, subtopic_results, chunk_results):
        candidates = entity_results + subtopic_results + chunk_results
        candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return candidates

    @staticmethod
    def _empty_result(status="not_found", context=None, search_errors=None):
        return {
            "found": False,
            "status": status,
            "domain_relevant": False,
            "context": context,
            "entity_results": [],
            "subtopic_results": [],
            "chunk_results": [],
            "relevant_chunks": [],
            "search_errors": search_errors or [],
            "results": [],
            "top3": [],
            "best_candidate": None,
        }

    @staticmethod
    def _scoped_entity_ids(entity_results, subtopic_results, chunk_results, limit):
        """Entity yang paling kuat, berurutan: entity/alias, subtopic, lalu chunk.

        Dasarnya semua bukti, bukan hanya daftar entity, supaya chunk tambahan
        tetap dicari walau penyaringan skor mengosongkan salah satu daftar.
        """
        entity_ids = []
        for group in (entity_results, subtopic_results, chunk_results):
            for item in group:
                entity_id = item.get("entity_id")
                if entity_id and entity_id not in entity_ids:
                    entity_ids.append(entity_id)
                if len(entity_ids) >= limit:
                    return entity_ids
        return entity_ids

    def _collect_relevant_chunks(self, query_embedding, entity_id, subtopic_id):

        if not entity_id:
            return []

        raw = self.chunk_retrieval.retrieve_all(
            query_embedding=query_embedding,
            entity_id=entity_id,
        )
        scored = self._convert_results(raw, "chunk")  # sorted by score (best first)
        if not scored:
            return []

        if subtopic_id:
            primary = [c for c in scored if c.get("subtopic_id") == subtopic_id]
            primary.sort(key=lambda c: (c.get("chunk_index") is None, c.get("chunk_index") or 0))
            primary_ids = {c["id"] for c in primary}
            extras = [c for c in scored if c["id"] not in primary_ids][:EXTRA_CHUNK_TOP_K]
            chunks = primary + extras
        else:
            chunks = scored[:ENTITY_CHUNK_TOP_K]

        return chunks[:MAX_RELEVANT_CHUNKS]

    def retrieve(self, question, previous_entity=None, top_k=CHUNK_TOP_K):
        if not question or not question.strip():
            return self._empty_result()

        timings = {}
        clock = time.perf_counter
        started = clock()

        # One normalization and one query embedding for all searches.
        normalized_question = normalize_query(question)
        mark = clock()
        query_embedding = create_query_embedding(normalized_question)
        timings["embedding"] = clock() - mark
        if query_embedding is None:
            result = self._empty_result(status="embedding_failed")
            result["timings"] = timings
            return result
        mark = clock()

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
        chunk_raw = self.chunk_retrieval.retrieve(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        # Alias evidence: multilingual aliases of entities and subtopics.
        alias_raw = semantic_search(
            query_embedding=query_embedding,
            collection=self.entity_collection,
            where={"type": "alias"},
            top_k=ALIAS_TOP_K,
            result_type="alias",
        )

        timings["chroma_search"] = clock() - mark

        search_errors = [
            raw["error"]
            for raw in (entity_raw, subtopic_raw, chunk_raw, alias_raw)
            if raw and raw.get("error")
        ]

        entity_results = self._convert_results(entity_raw, "entity")
        subtopic_results = self._convert_results(subtopic_raw, "subtopic")
        chunk_results = self._convert_results(chunk_raw, "chunk")

        alias_entities, alias_subtopics, alias_hits = (
            self._convert_alias_results(alias_raw)
        )
        entity_results = self._merge_best(
            entity_results, alias_entities, "entity_id", MAX_ENTITY_CANDIDATES
        )
        subtopic_results = self._merge_best(
            subtopic_results, alias_subtopics, "subtopic_id",
            MAX_SUBTOPIC_CANDIDATES,
        )

        # Extra chunk evidence inside the strongest entities, so a subtopic
        # found through an alias can be confirmed by an aligned chunk.
        top_entity_ids = self._scoped_entity_ids(
            entity_results, subtopic_results, chunk_results,
            ALIAS_SCOPED_ENTITY_LIMIT,
        )
        if top_entity_ids:
            scoped_raw = self.chunk_retrieval.retrieve(
                query_embedding=query_embedding,
                entity_id=top_entity_ids,
                top_k=top_k,
            )
            if scoped_raw.get("error"):
                search_errors.append(scoped_raw["error"])
            chunk_results = self._merge_best(
                chunk_results,
                self._convert_results(scoped_raw, "chunk"),
                "id",
                top_k * 2,
            )

        # ContextUnderstanding resolves context from these results; it does not
        # run EntityRetrieval semantic search or create another embedding.
        mark = clock()
        context = self.context_engine.analyze(
            question=question,
            previous_entity=previous_entity,
            normalized_question=normalized_question,
            query_embedding=query_embedding,
            entity_results=entity_results,
            subtopic_results=subtopic_results,
            chunk_results=chunk_results,
        )
        timings["context_analysis"] = clock() - mark

        # Keep the raw semantic matches for debugging, but never treat a raw
        # ChromaDB match as proof that the question belongs to the domain.
        raw_candidates = self._rank_candidates(
            entity_results, subtopic_results, chunk_results
        )
        domain_relevant = bool(context.get("domain_relevant"))
        resolved_entity_id = context.get("entity_id")

        candidates = []
        relevant_chunks = []

        if domain_relevant and resolved_entity_id:
            # Only content of the resolved entity reaches answer generation.
            # This prevents a high-scoring chunk from a different service from
            # hijacking the answer.
            candidates = self._rank_candidates(
                [i for i in entity_results if i.get("entity_id") == resolved_entity_id],
                [i for i in subtopic_results if i.get("entity_id") == resolved_entity_id],
                [i for i in chunk_results if i.get("entity_id") == resolved_entity_id],
            )

            mark = clock()
            relevant_chunks = self._collect_relevant_chunks(
                query_embedding,
                resolved_entity_id,
                context.get("subtopic_id"),
            )
            timings["collect_chunks"] = clock() - mark

            # Follow-up: entity berasal dari konteks sebelumnya sehingga bisa
            # tanpa kandidat; chunk entity itu tetap menjadi bahannya.
            if not candidates and relevant_chunks:
                candidates = sorted(
                    relevant_chunks,
                    key=lambda item: item.get("score", 0.0),
                    reverse=True,
                )

        # found: ada konteks nyata untuk dijawab oleh LLM.
        found = bool(domain_relevant and relevant_chunks)

        # best_candidate is meant to be usable content, so prefer the best
        # chunk; an entity/subtopic row has no text of its own.
        best_candidate = None
        if candidates:
            chunk_candidates = [c for c in candidates if c.get("type") == "chunk"]
            best_candidate = chunk_candidates[0] if chunk_candidates else candidates[0]

        if found:
            status = "found"
        elif not domain_relevant:
            # Hasil kosong karena error teknis bukan berarti di luar domain.
            status = "search_error" if search_errors else "out_of_domain"
        else:
            status = "not_found"

        timings["retrieval_total"] = clock() - started

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
            # Raw alias matches, kept for debugging.
            "alias_hits": alias_hits,
            # Kegagalan teknis ChromaDB (kosong bila semua pencarian berhasil).
            "search_errors": search_errors,
            # Full, ordered context for answer generation (used by format_context).
            "relevant_chunks": relevant_chunks,
            # Relevant, entity-scoped candidates for answer generation.
            "results": candidates,
            "top3": candidates[:3],
            "best_candidate": best_candidate,
            # Unfiltered matches retained for debugging only.
            "raw_results": raw_candidates,
            # Waktu per tahap (detik), untuk mencari bagian yang lambat.
            "timings": timings,
        }