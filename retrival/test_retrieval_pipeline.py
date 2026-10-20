from retrival.retrieval_pipeline import RetrievalPipeline


def print_separator(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_candidates(title, candidates):
    print_separator(title)

    if not candidates:
        print("Tidak ada kandidat.")
        return

    for index, item in enumerate(candidates, start=1):
        print(
            f"{index}. "
            f"type={item.get('type')} "
            f"| score={item.get('score', 0.0):.6f}"
        )

        print(
            f"   id={item.get('id')}"
        )

        print(
            f"   entity={item.get('entity')}"
        )

        print(
            f"   entity_id={item.get('entity_id')}"
        )

        print(
            f"   subtopic={item.get('subtopic')}"
        )

        print(
            f"   subtopic_id={item.get('subtopic_id')}"
        )

        if item.get("type") == "chunk":
            print(
                f"   chunk_id={item.get('content_chunk_id')}"
            )

            print(
                f"   document={item.get('document')}"
            )

            print(
                f"   page={item.get('page_start')}"
                f"-{item.get('page_end')}"
            )

            content = (
                item.get("content")
                or ""
            ).replace("\n", " ").strip()

            if len(content) > 150:
                content = content[:150] + "..."

            print(
                f"   content={content}"
            )


def main():

    print_separator(
        "RETRIEVAL PIPELINE DEBUG"
    )

    question = "how do i can extend my kitas?" 

    print(
        f"\nQuestion : {question}"
    )

    pipeline = RetrievalPipeline()

    result = pipeline.retrieve(
        question=question
    )

    # ============================================================
    # CONTEXT
    # ============================================================

    context = result.get(
        "context",
        {}
    )

    print_separator(
        "CONTEXT RESULT"
    )

    print(
        "Entity          :",
        context.get("entity")
    )

    print(
        "Entity ID       :",
        context.get("entity_id")
    )

    print(
        "Detected Entity :",
        context.get("detected_entity")
    )

    detected_score = context.get(
        "detected_score"
    )

    print(
        "Detected Score  :",
        (
            f"{detected_score:.6f}"
            if detected_score is not None
            else None
        )
    )

    print(
        "Explicit Entity :",
        context.get("explicit_entity")
    )

    print(
        "Subtopic        :",
        context.get("subtopic")
    )

    print(
        "Subtopic ID     :",
        context.get("subtopic_id")
    )

    subtopic_score = context.get(
        "subtopic_score"
    )

    print(
        "Subtopic Score  :",
        (
            f"{subtopic_score:.6f}"
            if subtopic_score is not None
            else None
        )
    )

    print(
        "Follow-up       :",
        context.get("is_followup")
    )

    # ============================================================
    # ENTITY
    # ============================================================

    print_candidates(
        "ENTITY CANDIDATES",
        result.get(
            "entity_results",
            []
        )
    )

    # ============================================================
    # SUBTOPIC
    # ============================================================

    print_candidates(
        "SUBTOPIC CANDIDATES",
        result.get(
            "subtopic_results",
            []
        )
    )

    # ============================================================
    # CHUNK
    # ============================================================

    print_candidates(
        "CHUNK CANDIDATES",
        result.get(
            "chunk_results",
            []
        )
    )

    # ============================================================
    # ALL CANDIDATES
    # ============================================================

    print_candidates(
        "ALL CANDIDATES - RANKING",
        result.get(
            "candidates",
            []
        )
    )

    # ============================================================
    # BEST CANDIDATE
    # ============================================================

    best = result.get(
        "best_candidate"
    )

    print_separator(
        "BEST CANDIDATE"
    )

    if best:
        print(
            "Type        :",
            best.get("type")
        )

        print(
            "ID          :",
            best.get("id")
        )

        print(
            "Score       :",
            f"{best.get('score', 0.0):.6f}"
        )

        print(
            "Entity      :",
            best.get("entity")
        )

        print(
            "Entity ID   :",
            best.get("entity_id")
        )

        print(
            "Subtopic    :",
            best.get("subtopic")
        )

        print(
            "Subtopic ID :",
            best.get("subtopic_id")
        )

    else:
        print(
            "Tidak ada best candidate."
        )

    # ============================================================
    # FINAL STATUS
    # ============================================================

    print_separator(
        "FINAL STATUS"
    )

    print(
        "Found :",
        result.get("found")
    )

    if result.get("error"):
        print(
            "Error :",
            result.get("error")
        )


if __name__ == "__main__":
    main()