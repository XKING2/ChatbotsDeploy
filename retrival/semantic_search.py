from config import TOP_K


def empty_result(error=None):
    result = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    if error:
        result["error"] = str(error)
    return result


def semantic_search(query_embedding, collection, where=None, top_k=TOP_K,
                    result_type=None, min_score=None):
    """
    Pencarian cosine ChromaDB. Tanpa min_score, semua kandidat dikembalikan
    sesuai peringkat (penyaringan dilakukan lewat penggabungan peringkat).
    """
    if query_embedding is None:
        return empty_result()
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k harus berupa integer minimal 1.")

    try:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        print(f"Semantic Search Error ({result_type}): {type(exc).__name__}: {exc}")
        return empty_result(exc)

    if min_score is None or min_score <= 0:
        return result

    keep = [i for i, d in enumerate((result.get("distances") or [[]])[0])
            if d is not None and 1.0 - float(d) >= min_score]
    return {key: [[(result.get(key) or [[]])[0][i] for i in keep]]
            for key in ("ids", "documents", "metadatas", "distances")}