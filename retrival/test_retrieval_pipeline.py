import sys

from database.chroma_db import chunk_collection, entity_collection
from retrival.retrieval_pipeline import RetrievalPipeline
from retrival.semantic_search import SEMANTIC_MIN_SCORES, semantic_search
from userquery.query_embedding import create_query_embedding
from userquery.query_normalizer import normalize_query

DEFAULT_QUESTION = "How long is a visa on arrival valid for?"


def print_separator(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _clip(text, limit):
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def print_candidates(title, candidates, show_support=False):
    print_separator(title)

    if not candidates:
        print("Tidak ada kandidat.")
        return

    for index, item in enumerate(candidates, start=1):
        line = f"{index}. type={item.get('type')} | score={item.get('score', 0.0):.6f}"
        if show_support and item.get("alias_support"):
            line += (
                f" | alias_support={item['alias_support']:.3f}"
                f" ({item.get('alias_hit_count', 0)} alias)"
            )
        print(line)
        print(f"   entity={_clip(item.get('entity'), 60)}")
        print(f"   subtopic={_clip(item.get('subtopic'), 70)}")
        if item.get("type") == "chunk":
            print(
                f"   chunk_id={item.get('content_chunk_id')} "
                f"| page={item.get('page_start')}-{item.get('page_end')}"
            )
            print(f"   content={_clip(item.get('content'), 150)}")


def print_score_tuning(question):
    """Skor mentah TANPA filter minimum, untuk menyetel SEMANTIC_MIN_SCORES.

    Jalankan dengan query valid dan query di luar domain, lalu atur minimum di
    antara skor teratas keduanya.
    """
    print_separator("SCORE TUNING (tanpa filter, hanya untuk menyetel minimum)")
    embedding = create_query_embedding(normalize_query(question))
    if embedding is None:
        print("Embedding gagal.")
        return

    searches = (
        ("entity", entity_collection, {"type": "entity"}),
        ("alias", entity_collection, {"type": "alias"}),
        ("subtopic", entity_collection, {"type": "subtopic"}),
        ("chunk", chunk_collection, None),
    )
    for name, collection, where in searches:
        raw = semantic_search(
            query_embedding=embedding,
            collection=collection,
            where=where,
            top_k=3,
            result_type=name,
            min_score=0.0,
        )
        distances = (raw.get("distances") or [[]])[0]
        scores = [1.0 - float(d) for d in distances if d is not None]
        top = f"{scores[0]:.4f}" if scores else "-"
        minimum = SEMANTIC_MIN_SCORES.get(name)
        verdict = ""
        if scores and minimum is not None:
            verdict = "lolos" if scores[0] >= minimum else "TERFILTER"
        print(
            f"{name:<9} skor teratas={top} | minimum={minimum} | {verdict} "
            f"| top3={[round(v, 4) for v in scores]}"
        )


def main():
    args = [a for a in sys.argv[1:] if a != "--tune"]
    tune = "--tune" in sys.argv[1:]
    question = " ".join(args).strip() or DEFAULT_QUESTION

    print_separator("RETRIEVAL PIPELINE DEBUG")
    print(f"\nQuestion : {question}")

    result = RetrievalPipeline().retrieve(question=question)
    context = result.get("context") or {}

    print_separator("CONTEXT RESULT")
    detected_score = context.get("detected_score")
    subtopic_score = context.get("subtopic_score")
    print("Entity          :", context.get("entity"))
    print("Entity ID       :", context.get("entity_id"))
    print("Detected Entity :", context.get("detected_entity"))
    print("Detected Score  :", f"{detected_score:.6f}" if detected_score is not None else None)
    print("Explicit Entity :", context.get("explicit_entity"))
    print("Resolution      :", context.get("resolution_reason"))
    margin = context.get("entity_score_margin")
    print("Score margin    :", f"{margin:.6f}" if margin is not None else None)
    print("Subtopic        :", context.get("subtopic"))
    print("Subtopic ID     :", context.get("subtopic_id"))
    print("Subtopic Score  :", f"{subtopic_score:.6f}" if subtopic_score is not None else None)
    print("Follow-up       :", context.get("is_followup"))

    # Retrieved contexts untuk dataset RAGAS
    print_separator("RETRIEVED CONTEXTS (DATASET RAGAS)")

    relevant = result.get("relevant_chunks", [])

    if not relevant:
        print("Retrieved contexts kosong.")
    else:
        retrieved_contexts = []

        for index, item in enumerate(relevant, start=1):
            content = (item.get("content") or "").strip()

            if not content:
                continue

            retrieved_contexts.append(content)

            print(f"\n--- Context {len(retrieved_contexts)} ---")
            print(f"Chunk ID  : {item.get('content_chunk_id')}")
            print(f"Entity    : {item.get('entity')}")
            print(f"Subtopic  : {item.get('subtopic')}")
            print(f"Score     : {item.get('score', 0.0):.6f}")
            print(f"Content   :\n{content}")

        print("\n" + "=" * 70)
        print(f"Total retrieved contexts: {len(retrieved_contexts)}")

    print_separator("ENTITY RANKING (keputusan ContextUnderstanding)")
    ranking = (context.get("entity_result") or {}).get("results", [])
    if not ranking:
        print("Tidak ada kandidat.")
    for index, item in enumerate(ranking, start=1):
        print(
            f"{index}. score={item.get('score', 0.0):.6f} "
            f"| sumber={','.join(item.get('sources') or [])} "
            f"| {_clip(item.get('entity'), 60)}"
        )

    print_candidates("ENTITY CANDIDATES", result.get("entity_results", []), show_support=True)
    print_candidates("SUBTOPIC CANDIDATES", result.get("subtopic_results", []), show_support=True)
    print_candidates("CHUNK CANDIDATES (bukti retrieval)", result.get("chunk_results", []))

    # Alias yang paling cocok dengan pertanyaan (multi-vektor).
    print_separator("ALIAS HITS (urut skor)")
    hits = result.get("alias_hits", [])
    if not hits:
        print("Tidak ada alias yang cocok.")
    for index, hit in enumerate(hits[:10], start=1):
        print(f"{index}. {hit['score']:.4f} | {hit['alias']!r}")
        print(f"   -> {_clip(hit.get('entity'), 50)} | {_clip(hit.get('subtopic'), 50)}")

    # Persis apa yang dikirim ke LLM.
    print_separator("RELEVANT CHUNKS (konteks yang dikirim ke LLM, urutan baca)")
    relevant = result.get("relevant_chunks", [])
    if not relevant:
        print("Tidak ada.")
    total_chars = 0
    for index, item in enumerate(relevant, start=1):
        content = item.get("content") or ""
        total_chars += len(content)
        print(
            f"{index}. {_clip(item.get('entity'), 40)} | {_clip(item.get('subtopic'), 40)} "
            f"| chunk_index={item.get('chunk_index')} | skor={item.get('score', 0.0):.4f} "
            f"| {len(content)} karakter"
        )
    print(f"\nTotal chunk: {len(relevant)} | total karakter konteks: {total_chars}")

    print_separator("ALL CANDIDATES - RANKING (di dalam entity terpilih)")
    ranked = result.get("results", [])
    if not ranked:
        print("Tidak ada kandidat.")
    for index, item in enumerate(ranked[:10], start=1):
        print(
            f"{index}. type={item.get('type')} | score={item.get('score', 0.0):.6f} "
            f"| {_clip(item.get('subtopic'), 60)}"
        )

    best = result.get("best_candidate")
    print_separator("BEST CANDIDATE")
    if best:
        print("Type        :", best.get("type"))
        print("ID          :", best.get("id"))
        print("Score       :", f"{best.get('score', 0.0):.6f}")
        print("Entity      :", best.get("entity"))
        print("Subtopic    :", best.get("subtopic"))
    else:
        print("Tidak ada best candidate.")

    print_separator("FINAL STATUS")
    print("Found  :", result.get("found"))
    print("Status :", result.get("status"))
    errors = result.get("search_errors") or []
    if errors:
        print("Search errors:")
        for error in errors:
            print("  -", error)

    if tune or not result.get("found"):
        print_score_tuning(question)


if __name__ == "__main__":
    main()