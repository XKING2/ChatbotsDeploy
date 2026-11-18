import time

from database.chroma_db import heading_collection
from retrival.chunk_retrieval import ChunkRetrieval
from retrival.context_understanding import ContextUnderstanding
from retrival.semantic_search import semantic_search
from userquery.query_embedding import create_query_embedding
from userquery.query_normalizer import normalize_query

# ============================================================
# KONFIGURASI
# ============================================================

HEADING_TOP_K = 10
ALIAS_TOP_K = 30
SUBHEADING_TOP_K = 15
CHUNK_TOP_K = 10

HEADING_CHUNK_TOP_K = 10   # chunk berperingkat di dalam heading terpilih
EXTRA_CHUNK_TOP_K = 2      # chunk tambahan di luar subheading terpilih
MAX_RELEVANT_CHUNKS = 20   # batas konteks untuk LLM
RUNNER_UP_RATIO = 0.8      # subheading kedua ikut jika skor RRF >= 80% juara


def _score(distance):
    try:
        return None if distance is None else 1.0 - float(distance)
    except (TypeError, ValueError):
        return None


class RetrievalPipeline:

    def __init__(self, context_engine=None, chunk_retrieval=None):
        self.context_engine = context_engine or ContextUnderstanding()
        self.chunk_retrieval = chunk_retrieval or ChunkRetrieval()
        self.heading_collection = heading_collection
        self.chunk_collection = self.chunk_retrieval.collection

    # ---------------------------------------------------------
    # KONVERSI HASIL CHROMA
    # ---------------------------------------------------------

    @staticmethod
    def _rows(raw):
        raw = raw or {}
        columns = [(raw.get(k) or [[]])[0]
                   for k in ("ids", "documents", "metadatas", "distances")]
        for item_id, doc, meta, dist in zip(*columns):
            yield item_id, doc or "", meta or {}, _score(dist)

    def _heading_rows(self, raw, catalog, kind):
        """Hasil pencarian heading atau alias -> daftar dict."""
        results = []
        for rid, doc, meta, score in self._rows(raw):
            hid = meta.get("heading_id") or (rid if kind == "heading" else None)
            if not hid:
                continue
            hid = str(hid)
            sid = meta.get("subheading_id")
            sub = catalog.subheading(sid)
            item = {
                **meta,
                "id": rid, "type": kind, "score": score, "document": doc,
                "heading_id": hid,
                "heading": meta.get("heading") or catalog.heading_name(hid)
                           or (doc if kind == "heading" else None),
                "subheading_id": str(sid) if sid else None,
                "subheading": meta.get("subheading")
                              or (sub["name"] if sub else None),
            }
            if kind == "alias":
                item["alias"] = meta.get("alias") or doc
                item["alias_scope"] = "subheading" if sid else "heading"
            results.append(item)
        return results

    def _subheading_rows(self, raw, catalog):
        results = []
        for rid, doc, meta, score in self._rows(raw):
            hid = meta.get("heading_id")
            sid = meta.get("subheading_id") or rid
            if not hid or not sid:
                continue
            sub = catalog.subheading(sid)
            results.append({
                **meta,
                "id": rid, "type": "subheading", "score": score,
                "document": doc,
                "heading_id": str(hid),
                "heading": meta.get("heading") or catalog.heading_name(hid),
                "subheading_id": str(sid),
                "subheading": meta.get("subheading")
                              or (sub["name"] if sub else doc),
            })
        return results

    @staticmethod
    def _chunk_item(rid, content, meta, score, catalog):
        hid = meta.get("heading_id")
        sid = meta.get("subheading_id")
        sub = catalog.subheading(sid)
        return {
            **meta,
            "id": rid,
            "type": "chunk",
            "heading_id": str(hid) if hid else None,
            "heading": meta.get("heading") or catalog.heading_name(hid),
            "subheading_id": str(sid) if sid else None,
            "subheading": meta.get("subheading") or (sub["name"] if sub else None),
            "score": score,
            "content": content,
            "document": meta.get("document"),
            "content_chunk_id": meta.get("content_chunk_id") or rid,
            "category": meta.get("category"),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "chunk_index": meta.get("chunk_index"),
            "total_chunks": meta.get("total_chunks"),
        }

    def _chunk_rows(self, raw, catalog):
        return [self._chunk_item(rid, doc, meta, score, catalog)
                for rid, doc, meta, score in self._rows(raw)
                if meta.get("heading_id")]

    # ---------------------------------------------------------
    # PENCARIAN
    # ---------------------------------------------------------

    @staticmethod
    def _order(item):
        index = item.get("chunk_index")
        return (index is None, index if index is not None else 0)

    def _search(self, embedding, where, top_k):
        return semantic_search(embedding, self.heading_collection, where=where,
                               top_k=top_k, min_score=0.0)

    def _assemble(self, embedding, context, scoped_chunks, catalog):
        """
        Susun konteks LLM: seluruh chunk subheading terpilih (urut dokumen),
        subheading pesaing yang skornya berdekatan, dan beberapa chunk
        terbaik lain dalam heading.
        """
        heading_id = context.get("heading_id")
        subheading_id = context.get("subheading_id")
        scored = {c["id"]: c for c in scoped_chunks}

        if not subheading_id:
            return list(scoped_chunks)[:MAX_RELEVANT_CHUNKS]

        def full(sid):
            items = []
            for rid, doc, meta, score in self.chunk_retrieval.get_all_scored(
                    embedding, heading_id, sid):
                known = scored.get(rid)
                if score is None and known:
                    score = known["score"]
                items.append(self._chunk_item(rid, doc, meta, score, catalog))
            return items

        primary = full(subheading_id)
        if len(primary) > MAX_RELEVANT_CHUNKS:
            primary = sorted(primary, key=lambda c: -(c["score"] or -1))
            primary = sorted(primary[:MAX_RELEVANT_CHUNKS], key=self._order)

        selected, seen = list(primary), {c["id"] for c in primary}

        candidates = context.get("subheading_candidates") or []
        if len(candidates) > 1 and candidates[0]["rrf_score"] > 0:
            runner = candidates[1]
            if runner["rrf_score"] >= RUNNER_UP_RATIO * candidates[0]["rrf_score"]:
                extra = [c for c in scoped_chunks
                         if c.get("subheading_id") == runner["subheading_id"]
                         and c["id"] not in seen][:EXTRA_CHUNK_TOP_K]
                selected += extra
                seen.update(c["id"] for c in extra)

        extras = [c for c in scoped_chunks if c["id"] not in seen][:EXTRA_CHUNK_TOP_K]
        return (selected + extras)[:MAX_RELEVANT_CHUNKS]

    # ---------------------------------------------------------
    # HASIL
    # ---------------------------------------------------------

    @staticmethod
    def _empty_result(status="not_found", context=None, search_errors=None,
                      timings=None):
        return {
            "found": False, "status": status, "domain_relevant": False,
            "retrieval_found": False, "context": context,
            "heading_results": [], "subheading_results": [],
            "chunk_results": [], "alias_hits": [],
            "relevant_chunks": [], "results": [], "top3": [],
            "best_candidate": None, "raw_results": [],
            "search_errors": search_errors or [], "timings": timings or {},
        }

    @staticmethod
    def _unique(*groups):
        seen, out = set(), []
        for group in groups:
            for item in group:
                key = (item.get("type"), item.get("id"))
                if key not in seen:
                    seen.add(key)
                    out.append(item)
        return out

    # ---------------------------------------------------------
    # UTAMA
    # ---------------------------------------------------------

    def retrieve(self, question, previous_heading=None,
                 previous_heading_id=None, top_k=CHUNK_TOP_K):
        if not isinstance(question, str) or not question.strip():
            return self._empty_result()
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k harus berupa integer minimal 1.")

        clock = time.perf_counter
        timings, started = {}, clock()

        normalized = normalize_query(question)
        if not normalized:
            return self._empty_result("empty_query")

        mark = clock()
        embedding = create_query_embedding(normalized)
        timings["embedding"] = clock() - mark
        if embedding is None:
            timings["retrieval_total"] = clock() - started
            return self._empty_result("embedding_failed", timings=timings)

        try:
            catalog = self.context_engine.catalog()
        except Exception as exc:
            return self._empty_result(
                "search_error", search_errors=[f"catalog: {exc}"], timings=timings)

        # --- empat kanal pencarian, satu embedding ---
        mark = clock()
        raws = {
            "heading": self._search(embedding, {"type": "heading"}, HEADING_TOP_K),
            "alias": self._search(embedding, {"type": "alias"}, ALIAS_TOP_K),
            "subheading": self._search(
                embedding, {"type": "subheading"}, SUBHEADING_TOP_K),
            "chunk": self.chunk_retrieval.retrieve(
                query_embedding=embedding, top_k=top_k, min_score=0.0),
        }
        timings["chroma_search"] = clock() - mark
        search_errors = [str(r["error"]) for r in raws.values()
                         if r and r.get("error")]

        heading_results = self._heading_rows(raws["heading"], catalog, "heading")
        alias_results = self._heading_rows(raws["alias"], catalog, "alias")
        subheading_results = self._subheading_rows(raws["subheading"], catalog)
        chunk_results = self._chunk_rows(raws["chunk"], catalog)

        alias_hits = [{
            "alias": a["alias"], "heading": a["heading"],
            "subheading": a["subheading"], "heading_id": a["heading_id"],
            "subheading_id": a["subheading_id"], "score": a["score"],
        } for a in alias_results]

        # --- pemahaman konteks; chunk dalam heading diambil sesuai kebutuhan ---
        scoped_cache = {}

        def scoped_chunks(heading_id):
            if heading_id not in scoped_cache:
                raw = self.chunk_retrieval.retrieve(
                    query_embedding=embedding, heading_id=heading_id,
                    top_k=HEADING_CHUNK_TOP_K, min_score=0.0)
                if raw.get("error"):
                    search_errors.append(str(raw["error"]))
                scoped_cache[heading_id] = self._chunk_rows(raw, catalog)
            return scoped_cache[heading_id]

        mark = clock()
        context = self.context_engine.analyze(
            question=question,
            previous_heading=previous_heading,
            previous_heading_id=previous_heading_id,
            normalized_question=normalized,
            query_embedding=embedding,
            heading_results=heading_results,
            alias_results=alias_results,
            subheading_results=subheading_results,
            chunk_results=chunk_results,
            scoped_chunks=scoped_chunks,
        )
        timings["context_analysis"] = clock() - mark

        domain_relevant = bool(context.get("domain_relevant"))
        heading_id = context.get("heading_id")

        relevant = []
        if domain_relevant and heading_id:
            mark = clock()
            relevant = self._assemble(
                embedding, context, scoped_chunks(heading_id), catalog)
            timings["collect_chunks"] = clock() - mark

        found = bool(domain_relevant and relevant)
        if found:
            status = "found"
        elif search_errors:
            status = "search_error"
        elif not domain_relevant:
            status = "out_of_domain"
        else:
            status = "not_found"

        ranked = sorted(relevant, key=lambda c: -(c["score"] if c["score"] is not None else -1))

        raw_results = self._unique(
            heading_results, alias_results, subheading_results, chunk_results)
        timings["retrieval_total"] = clock() - started

        return {
            "found": found,
            "status": status,
            "domain_relevant": domain_relevant,
            "retrieval_found": bool(raw_results),
            "context": context,
            "heading_results": heading_results,
            "alias_results": alias_results,
            "subheading_results": subheading_results,
            "chunk_results": chunk_results,
            "alias_hits": alias_hits,
            "relevant_chunks": relevant,
            "results": relevant,
            "top3": ranked[:3],
            "best_candidate": ranked[0] if ranked else None,
            "raw_results": raw_results,
            "search_errors": search_errors,
            "timings": timings,
        }