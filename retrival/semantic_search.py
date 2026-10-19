from config import TOP_K


SEMANTIC_MIN_SCORES = {
    "entity": 0.0,
    "subtopic": 0.0,
    "chunk": 0.0
}


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

        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    if include is None:

        include = [
            "documents",
            "metadatas",
            "distances"
        ]

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": include
    }

    if where is not None:
        query_args["where"] = where

    try:

        result = collection.query(
            **query_args
        )

    except Exception as exc:

        print(
            f"Semantic Search Error: {exc}"
        )

        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    if min_score is None:

        min_score = SEMANTIC_MIN_SCORES.get(
            result_type,
            0.0
        )

    if min_score <= 0:

        return result

    ids = result.get(
        "ids",
        [[]]
    )

    documents = result.get(
        "documents",
        [[]]
    )

    metadatas = result.get(
        "metadatas",
        [[]]
    )

    distances = result.get(
        "distances",
        [[]]
    )

    ids = ids[0] if ids else []
    documents = (
        documents[0]
        if documents
        else []
    )

    metadatas = (
        metadatas[0]
        if metadatas
        else []
    )

    distances = (
        distances[0]
        if distances
        else []
    )

    filtered_ids = []
    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []

    for (
        item_id,
        document,
        metadata,
        distance
    ) in zip(
        ids,
        documents,
        metadatas,
        distances
    ):

        score = (
            1.0 - float(distance)
            if distance is not None
            else 0.0
        )

        if score < min_score:
            continue

        filtered_ids.append(item_id)
        filtered_documents.append(document)
        filtered_metadatas.append(metadata)
        filtered_distances.append(distance)

    return {
        "ids": [filtered_ids],
        "documents": [filtered_documents],
        "metadatas": [filtered_metadatas],
        "distances": [filtered_distances]
    }