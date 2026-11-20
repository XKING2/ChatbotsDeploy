import sys

from database.chroma_db import chunk_collection, heading_collection
from retrival.retrieval_pipeline import RetrievalPipeline
from retrival.semantic_search import semantic_search
from userquery.query_embedding import create_query_embedding
from userquery.query_normalizer import normalize_query

DEFAULT_QUESTION = "Apa Itu GV"


def print_separator(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _clip(text, limit):
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _fmt(value):
    return f"{value:.6f}" if isinstance(value, (int, float)) else str(value)


def print_candidates(title, candidates):
    print_separator(title)
    if not candidates:
        print("Tidak ada kandidat.")
        return

    for index, item in enumerate(candidates, start=1):
        print(f"{index}. type={item.get('type')} | score={_fmt(item.get('score'))}")
        print(f"   heading={_clip(item.get('heading'), 60)}")
        print(f"   subheading={_clip(item.get('subheading'), 70)}")
        if item.get("type") == "alias":
            print(f"   alias={_clip(item.get('alias'), 70)} "
                  f"| scope={item.get('alias_scope')}")
        if item.get("type") == "chunk":
            print(f"   chunk_id={item.get('content_chunk_id')} "
                  f"| page={item.get('page_start')}-{item.get('page_end')}")
            print(f"   content={_clip(item.get('content'), 150)}")


def print_score_tuning(question):

    print_separator("SCORE TUNING (skor mentah per kanal, tanpa ambang)")
    embedding = create_query_embedding(normalize_query(question))
    if embedding is None:
        print("Embedding gagal.")
        return

    searches = (
        ("heading", heading_collection, {"type": "heading"}),
        ("alias", heading_collection, {"type": "alias"}),
        ("subheading", heading_collection, {"type": "subheading"}),
        ("chunk", chunk_collection, None),
    )
    for name, collection, where in searches:
        raw = semantic_search(embedding, collection, where=where, top_k=3,
                              result_type=name, min_score=0.0)
        distances = (raw.get("distances") or [[]])[0]
        scores = [1.0 - float(d) for d in distances if d is not None]
        top = f"{scores[0]:.4f}" if scores else "-"
        print(f"{name:<11} skor teratas={top} | top3={[round(v, 4) for v in scores]}")


def main():
    args = [a for a in sys.argv[1:] if a != "--tune"]
    tune = "--tune" in sys.argv[1:]
    question = " ".join(args).strip() or DEFAULT_QUESTION

    print_separator("RETRIEVAL PIPELINE DEBUG")
    print(f"\nQuestion : {question}")

    result = RetrievalPipeline().retrieve(question=question)
    context = result.get("context") or {}

    print_separator("CONTEXT RESULT")
    print("Heading            :", context.get("heading"))
    print("Heading ID         :", context.get("heading_id"))
    print("Detected Heading   :", context.get("detected_heading"))
    print("Detected Score     :", _fmt(context.get("detected_score")))
    print("Explicit Heading   :", context.get("explicit_heading"))
    print("Ambiguous          :", context.get("ambiguous_candidates"))
    print("Resolution         :", context.get("resolution_reason"))
    print("Domain relevant    :", context.get("domain_relevant"),
          f"(kanal setuju: {context.get('agreeing_channels')})")
    print("Subheading         :", context.get("subheading"))
    print("Subheading ID      :", context.get("subheading_id"))
    print("Subheading Match   :", context.get("subheading_match_type"))
    print("Subheading Score   :", _fmt(context.get("subheading_score")))
    print("Follow-up          :", context.get("is_followup"))

    # Retrieved contexts untuk dataset RAGAS
    print_separator("RETRIEVED CONTEXTS (DATASET RAGAS)")
    relevant = result.get("relevant_chunks", [])
    retrieved_contexts = []
    for item in relevant:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        retrieved_contexts.append(content)
        print(f"\n--- Context {len(retrieved_contexts)} ---")
        print(f"Chunk ID   : {item.get('content_chunk_id')}")
        print(f"Heading    : {item.get('heading')}")
        print(f"Subheading : {item.get('subheading')}")
        print(f"Score      : {_fmt(item.get('score'))}")
        print(f"Content    :\n{content}")
    if not retrieved_contexts:
        print("Retrieved contexts kosong.")
    else:
        print("\n" + "=" * 70)
        print(f"Total retrieved contexts: {len(retrieved_contexts)}")

    print_separator("HEADING RANKING (RRF di ContextUnderstanding)")
    ranking = (context.get("heading_result") or {}).get("results", [])
    if not ranking:
        print("Tidak ada kandidat.")
    for index, item in enumerate(ranking[:10], start=1):
        ranks = ", ".join(f"{c}#{r}" for c, r in
                          (item.get("channel_ranks") or {}).items())
        print(f"{index}. rrf={item.get('rrf_score', 0.0):.6f} "
              f"| cosine={_fmt(item.get('best_cosine'))} "
              f"| {ranks} | {_clip(item.get('heading'), 60)}")

    print_separator("SUBHEADING RANKING (di dalam heading terpilih)")
    subs = context.get("subheading_candidates") or []
    if not subs:
        print("Tidak ada kandidat.")
    for index, item in enumerate(subs, start=1):
        ranks = ", ".join(f"{c}#{r}" for c, r in
                          (item.get("channel_ranks") or {}).items())
        print(f"{index}. rrf={item.get('rrf_score', 0.0):.6f} | {ranks} "
              f"| {_clip(item.get('subheading'), 60)}")

    print_candidates("HEADING CANDIDATES", result.get("heading_results", []))
    print_candidates("ALIAS CANDIDATES", result.get("alias_results", []))
    print_candidates("SUBHEADING CANDIDATES", result.get("subheading_results", []))
    print_candidates("CHUNK CANDIDATES (bukti retrieval)", result.get("chunk_results", []))

    print_separator("ALIAS HITS (urut skor)")
    hits = result.get("alias_hits", [])
    if not hits:
        print("Tidak ada alias yang cocok.")
    for index, hit in enumerate(hits[:10], start=1):
        print(f"{index}. {_fmt(hit.get('score'))} | {hit.get('alias')!r}")
        print(f"   -> {_clip(hit.get('heading'), 50)} | {_clip(hit.get('subheading'), 50)}")

    print_separator("RELEVANT CHUNKS (konteks yang dikirim ke LLM, urutan baca)")
    if not relevant:
        print("Tidak ada.")
    total_chars = 0
    for index, item in enumerate(relevant, start=1):
        content = item.get("content") or ""
        total_chars += len(content)
        print(f"{index}. {_clip(item.get('heading'), 40)} | {_clip(item.get('subheading'), 40)} "
              f"| chunk_index={item.get('chunk_index')} | skor={_fmt(item.get('score'))} "
              f"| {len(content)} karakter")
    print(f"\nTotal chunk: {len(relevant)} | total karakter konteks: {total_chars}")

    print_separator("TOP 3 CHUNK (skor tertinggi)")
    for index, item in enumerate(result.get("top3", []), start=1):
        print(f"{index}. score={_fmt(item.get('score'))} "
              f"| {_clip(item.get('subheading'), 60)} | chunk_index={item.get('chunk_index')}")

    best = result.get("best_candidate")
    print_separator("BEST CANDIDATE")
    if best:
        print("Type       :", best.get("type"))
        print("ID         :", best.get("id"))
        print("Score      :", _fmt(best.get("score")))
        print("Heading    :", best.get("heading"))
        print("Subheading :", best.get("subheading"))
    else:
        print("Tidak ada best candidate.")

    print_separator("FINAL STATUS")
    print("Found  :", result.get("found"))
    print("Status :", result.get("status"))
    for error in result.get("search_errors") or []:
        print("  - search error:", error)
    print("Timings:", {k: round(v, 3) for k, v in (result.get("timings") or {}).items()})

    if tune or not result.get("found"):
        print_score_tuning(question)


if __name__ == "__main__":
    main()