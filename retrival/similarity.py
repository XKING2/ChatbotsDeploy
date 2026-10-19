import numpy as np


def build_embedding_matrix(items, embedding_key="embedding"):
    matrix = np.array(
        [item[embedding_key] for item in items],
        dtype=np.float32
    )
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-10
    return matrix, norms


def cosine_batch(query_embedding, matrix, norms):
    query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        query_norm = 1e-10

    dot_products = matrix @ query
    return dot_products / (norms * query_norm)


def rank_candidates(names, scores, extra_fields=None):
    candidates = []

    for i, name in enumerate(names):
        candidate = {
            "name": name,
            "score": float(scores[i])
        }

        if extra_fields:
            for field_name, values in extra_fields.items():
                candidate[field_name] = values[i]

        candidates.append(candidate)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def determine_specificity(
    candidates,
    threshold,
    specific_gap,
    max_candidates
):
    relevant = [
        c for c in candidates
        if c["score"] >= threshold
    ]

    if not relevant:
        return {
            "found": False,
            "status": "not_found",
            "results": [],
            "top3": candidates[:3]
        }

    if len(relevant) == 1:
        return {
            "found": True,
            "status": "specific",
            "results": [relevant[0]],
            "top3": candidates[:3]
        }

    gap = relevant[0]["score"] - relevant[1]["score"]

    if gap >= specific_gap:
        return {
            "found": True,
            "status": "specific",
            "results": [relevant[0]],
            "top3": candidates[:3]
        }

    return {
        "found": True,
        "status": "ambiguous",
        "results": relevant[:max_candidates],
        "top3": candidates[:3]
    }


def build_multi_embedding_matrix(
    entities,
    embeddings_key="embeddings"
):
    vectors = []
    owner_names = []

    for entity in entities:
        for embedding in entity[embeddings_key]:
            vectors.append(embedding)
            owner_names.append(
                entity.get("name") or entity.get("entity")
            )

    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-10

    return matrix, norms, owner_names


def aggregate_max_per_entity(owner_names, scores):
    best_scores = {}

    for name, score in zip(owner_names, scores):
        score = float(score)

        if name not in best_scores or score > best_scores[name]:
            best_scores[name] = score

    names = list(best_scores.keys())
    score_values = [best_scores[name] for name in names]

    return names, score_values
