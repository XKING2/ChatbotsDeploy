from config import TOP_K

SEMANTIC_MIN_SCORES = {
    "entity": 0.60,
    "alias": 0.60,
    "subtopic": 0.60,
    "chunk": 0.55,
}


def _empty_result(error=None):
    result = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    if error:
        # Kegagalan teknis dibedakan dari "tidak ada yang cocok".
        result["error"] = str(error)
    return result


def _first(result, key):
    value = result.get(key)
    if value is None or len(value) == 0:
        return []
    return value[0]


def semantic_search(
    query_embedding,
    collection,
    where=None,
    top_k=TOP_K,
    result_type=None,
    min_score=None,
    include=None
):

    if query_embedding is None:
        return _empty_result()

    if include is None:
        include = ["documents", "metadatas", "distances"]
    elif "distances" not in include:
        include = list(include) + ["distances"]

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": include,
    }
    if where:
        query_args["where"] = where

    try:
        result = collection.query(**query_args)
    except Exception as exc:
        print(f"Semantic Search Error ({result_type}): {exc}")
        return _empty_result(error=exc)

    if min_score is None:
        min_score = SEMANTIC_MIN_SCORES.get(result_type, 0.0)

    if min_score <= 0:
        return result

    ids = _first(result, "ids")
    documents = _first(result, "documents")
    metadatas = _first(result, "metadatas")
    distances = _first(result, "distances")

    keep = []
    for index, distance in enumerate(distances):
        score = 1.0 - float(distance) if distance is not None else 0.0
        if score >= min_score:
            keep.append(index)

    def pick(values):
        return [values[i] for i in keep if i < len(values)]

    return {
        "ids": [pick(ids)],
        "documents": [pick(documents)],
        "metadatas": [pick(metadatas)],
        "distances": [pick(distances)],
    }