"""Serialize relevant RetrievalPipeline chunks as JSON for the LLM."""

import json


def _as_list(value):
    """Accept a list or Chroma's nested list format."""
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return value if isinstance(value, list) else []


def _page_value(metadata):
    page_start = metadata.get("page_start", metadata.get("page"))
    page_end = metadata.get("page_end")
    if page_start is None:
        return None
    if page_end is not None and str(page_start) != str(page_end):
        return f"{page_start}-{page_end}"
    return page_start


def format_context(retrieval_result):
    """
    Return a JSON string, not a prose-formatted context block.

    New RetrievalPipeline contract:
      {
        "found": bool,
        "context": {"entity_id": ...},
        "chunk_results": [
          {"content": ..., "entity": ..., "entity_id": ...,
           "subtopic": ..., "subtopic_id": ..., "score": ..., ...}
        ]
      }

    Legacy raw Chroma results remain supported for migration/testing.
    """
    if not retrieval_result:
        return json.dumps(
            {"found": False, "contexts": [], "message": "Tidak ada informasi yang relevan."},
            ensure_ascii=False,
            indent=2,
        )

    if "found" in retrieval_result or "chunk_results" in retrieval_result:
        if not retrieval_result.get("found", False):
            return json.dumps(
                {
                    "found": False,
                    "contexts": [],
                    "message": "Tidak ada informasi yang relevan ditemukan dalam basis pengetahuan.",
                },
                ensure_ascii=False,
                indent=2,
            )

        context_meta = retrieval_result.get("context") or {}
        resolved_entity_id = context_meta.get("entity_id")
        chunks = retrieval_result.get("relevant_chunks")
        keep_reading_order = chunks is not None
        if chunks is None:
            chunks = retrieval_result.get("chunk_results") or []

        # Only include chunks belonging to the resolved entity. Never fall
        # back to raw semantic candidates when relevance validation failed.
        if resolved_entity_id:
            chunks = [
                item for item in chunks
                if item.get("entity_id") == resolved_entity_id
            ]

        normalized = []
        for item in chunks:
            content = item.get("content") or item.get("document")
            if not content:
                continue
            score = item.get("score")
            try:
                score = float(score) if score is not None else None
            except (TypeError, ValueError):
                score = None

            # Hanya field yang berguna bagi LLM. id, skor, dan chunk_index
            # tidak membantu menjawab dan hanya memperbesar prompt.
            entry = {
                "subtopic": item.get("subtopic"),
                "source": item.get("document") or item.get("source"),
                "page": _page_value(item),
                "content": content,
            }
            normalized.append({k: v for k, v in entry.items() if v not in (None, "")})

        # relevant_chunks is already in reading order (subtopic chunks by
        # chunk_index); re-sorting by score would scramble multi-part content.
        if not keep_reading_order:
            normalized.sort(
                key=lambda item: item["similarity_score"]
                if item["similarity_score"] is not None else -1,
                reverse=True,
            )
        payload = {
            "found": bool(normalized),
            "resolved_entity": context_meta.get("entity"),
            "contexts": normalized,
        }
        if not normalized:
            payload["message"] = (
                "Tidak ada informasi yang relevan ditemukan dalam basis pengetahuan."
            )
        # Tanpa indent: spasi dan baris baru hanya menambah token.
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    # Legacy Chroma result shape.
    documents = _as_list(retrieval_result.get("documents"))
    metadatas = _as_list(retrieval_result.get("metadatas"))
    distances = _as_list(retrieval_result.get("distances"))
    contexts = []
    for i, doc in enumerate(documents):
        meta = metadatas[i] or {} if i < len(metadatas) else {}
        score = None
        if i < len(distances):
            try:
                score = 1.0 - float(distances[i])
            except (TypeError, ValueError):
                pass
        contexts.append({
            "entity": meta.get("entity"),
            "entity_id": meta.get("entity_id"),
            "subtopic": meta.get("subtopic"),
            "subtopic_id": meta.get("subtopic_id"),
            "category": meta.get("category"),
            "source": meta.get("document") or meta.get("source"),
            "page": _page_value(meta),
            "similarity_score": score,
            "content": doc,
        })
    return json.dumps(
        {"found": bool(contexts), "resolved_entity": None,
         "resolved_entity_id": None, "contexts": contexts},
        ensure_ascii=False,
        indent=2,
    )