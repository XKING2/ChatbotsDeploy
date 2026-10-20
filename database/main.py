import os
import sys
import time

from config import PDF_FOLDER

from database.parser import parse_pdf
from database.preprocessing import preprocess_blocks
from database.section_builder import build_sections
from database.chunker import chunk_sections
from database.embedding import create_embedding
from database.hashing import compute_hash

from database.entity_builder import (
    collect_entities,
    build_entity_embedding_text,
    build_subtopic_embedding_text
)

from database.alias_generator import generate_aliases_batch

from database.heading_prompt import (
    build_heading_description,
    build_heading_examples
)

from database.index_builder import build_knowledge_index

from database.chroma_db import (
    get_existing_chunk_hashes,
    add_chunks,
    get_existing_entity_hashes,
    add_entity_item,
    get_aliases,
    alias_id,
    database_stats
)


REQUEST_LIMIT = 50
COOLDOWN_SECONDS = 120

request_count = 0


def embed(text):
    global request_count

    if request_count >= REQUEST_LIMIT:
        print(
            f"\nSudah {REQUEST_LIMIT} request, "
            f"jeda {COOLDOWN_SECONDS} detik..."
        )

        time.sleep(COOLDOWN_SECONDS)
        request_count = 0

    result = create_embedding(text)

    request_count += 1

    return result


def prepare_chunks(chunks):
    existing = get_existing_chunk_hashes()

    pending = []
    unchanged = 0

    for chunk in chunks:
        chunk_id = chunk["content_chunk_id"]
        chunk_hash = chunk["content_hash"]

        if existing.get(chunk_id) != chunk_hash:
            pending.append(chunk)
        else:
            unchanged += 1

    print(
        f"Chunk baru/berubah : {len(pending)}"
    )

    print(
        f"Chunk tidak berubah: {unchanged}"
    )

    return pending


def process_chunks(chunks):
    pending = prepare_chunks(chunks)

    valid = []

    for i, chunk in enumerate(pending, 1):

        print(
            f"\n[{i}/{len(pending)}] "
            f"{chunk['entity']}"
        )

        print(
            f"{chunk['subtopic']}"
        )

        embedding = embed(
            chunk["retrieval_text"]
        )

        if embedding is None:
            print("Embedding gagal.")
            continue

        chunk["embedding"] = embedding

        valid.append(chunk)

        print("OK")

    add_chunks(valid)


def load_aliases(entities):
    aliases_by_entity = {}

    new_entities = []

    for item in entities:

        entity_id = item["entity_id"]
        entity = item["entity"]

        data = get_aliases(entity_id)

        aliases = [
            metadata.get("alias")
            for metadata in data["metadatas"]
            if metadata.get("alias")
        ]

        if aliases:
            aliases_by_entity[entity] = aliases

        else:
            new_entities.append(entity)

    print(
        f"Entity sudah memiliki alias : "
        f"{len(entities) - len(new_entities)}"
    )

    print(
        f"Entity perlu generate alias  : "
        f"{len(new_entities)}"
    )

    if new_entities:

        generated = generate_aliases_batch(
            new_entities
        ) or {}

        for entity in new_entities:

            aliases = generated.get(
                entity,
                []
            )

            if not isinstance(aliases, list):
                aliases = []

            aliases_by_entity[entity] = aliases

    return aliases_by_entity


def process_entities(entities):

    existing = get_existing_entity_hashes()

    aliases_by_entity = load_aliases(
        entities
    )

    entity_count = 0
    subtopic_count = 0
    alias_count = 0

    for item in entities:

        entity_id = item["entity_id"]
        entity = item["entity"]

        aliases = aliases_by_entity.get(
            entity,
            []
        )

        subtopics = list(
            item["subtopics"].values()
        )

        subtopic_names = [
            x["subtopic"]
            for x in subtopics
        ]

        # ====================================================
        # ENTITY EMBEDDING
        # ====================================================

        entity_text = build_entity_embedding_text(
            entity,
            aliases,
            subtopic_names
        )

        entity_hash = compute_hash(
            entity_text
        )

        if existing.get(entity_id) != entity_hash:

            print(
                f"\nENTITY: {entity}"
            )

            embedding = embed(
                entity_text
            )

            if embedding is not None:

                add_entity_item(
                    "entity",
                    entity_id,
                    entity_id,
                    entity_text,
                    embedding,
                    entity_hash,
                    subtopic_count=len(subtopics),
                    alias_count=len(aliases)
                )

                existing[entity_id] = entity_hash

                entity_count += 1

        # ====================================================
        # SUBTOPIC EMBEDDING
        # ====================================================

        for subtopic_data in subtopics:

            subtopic_id = (
                subtopic_data["subtopic_id"]
            )

            subtopic = (
                subtopic_data["subtopic"]
            )

            # ------------------------------------------------
            # DESCRIPTION
            # ------------------------------------------------

            description = build_heading_description(
                subtopic,
                entity
            )

            # ------------------------------------------------
            # EXAMPLES
            # ------------------------------------------------

            examples = build_heading_examples(
                subtopic,
                entity
            )

            # ------------------------------------------------
            # EMBEDDING TEXT
            # ------------------------------------------------

            subtopic_text = build_subtopic_embedding_text(
                entity=entity,
                subtopic=subtopic,
                description=description,
                examples=examples
            )

            # ------------------------------------------------
            # HASH
            # ------------------------------------------------

            subtopic_hash = compute_hash(
                subtopic_text
            )

            # ------------------------------------------------
            # CHECK EXISTING
            # ------------------------------------------------

            if existing.get(subtopic_id) == subtopic_hash:

                continue

            print(
                f"\nSUBTOPIC: {subtopic}"
            )

            print(
                f"ENTITY  : {entity}"
            )

            print(
                "\n--- DESCRIPTION ---"
            )

            print(
                description
            )

            print(
                "\n--- EXAMPLES ---"
            )

            for example in examples:
                print(
                    f"- {example}"
                )

            print(
                "\n--- EMBEDDING TEXT ---"
            )

            print(
                subtopic_text
            )

            print(
                "--- END EMBEDDING TEXT ---"
            )

            # ------------------------------------------------
            # CREATE EMBEDDING
            # ------------------------------------------------

            embedding = embed(
                subtopic_text
            )

            if embedding is None:

                print(
                    "Embedding gagal."
                )

                continue

            # ------------------------------------------------
            # SAVE TO CHROMADB
            # ------------------------------------------------

            add_entity_item(
                "subtopic",
                subtopic_id,
                entity_id,
                subtopic_text,
                embedding,
                subtopic_hash,
                description=description,
                examples=" | ".join(examples)
            )

            existing[subtopic_id] = (
                subtopic_hash
            )

            subtopic_count += 1

            print(
                "Subtopic berhasil disimpan."
            )

        # ====================================================
        # ALIAS EMBEDDING
        # ====================================================

        for alias in aliases:

            alias = alias.strip()

            if not alias:
                continue

            current_id = alias_id(
                entity_id,
                alias
            )

            alias_text = (
                f"Layanan\n"
                f"{entity}\n\n"
                f"Nama lain\n"
                f"{alias}"
            )

            alias_hash = compute_hash(
                alias_text
            )

            if existing.get(current_id) == alias_hash:
                continue

            print(
                f"\nALIAS: {alias}"
            )

            embedding = embed(
                alias_text
            )

            if embedding is None:

                print(
                    "Embedding gagal."
                )

                continue

            add_entity_item(
                "alias",
                current_id,
                entity_id,
                alias,
                embedding,
                alias_hash
            )

            existing[current_id] = (
                alias_hash
            )

            alias_count += 1

    return (
        entity_count,
        subtopic_count,
        alias_count
    )


def main():

    pdf_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            PDF_FOLDER,
            "Dataset_SOP_Kanim.pdf"
        )
    )

    print(
        "=" * 70
    )

    print(
        "DATABASE KNOWLEDGE BASE"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # PARSE PDF
    # ========================================================

    parsed = parse_pdf(
        pdf_path
    )

    print(
        f"Baris mentah : {len(parsed)}"
    )

    # ========================================================
    # PREPROCESS
    # ========================================================

    parsed = preprocess_blocks(
        parsed
    )

    print(
        f"Baris bersih : {len(parsed)}"
    )

    # ========================================================
    # BUILD SECTIONS
    # ========================================================

    sections = build_sections(
        parsed
    )

    print(
        f"Section      : {len(sections)}"
    )

    # ========================================================
    # BUILD CHUNKS
    # ========================================================

    chunks = chunk_sections(
        sections
    )

    print(
        f"Chunk        : {len(chunks)}"
    )

    if not chunks:

        print(
            "\nTidak ada chunk. "
            "Database tidak diubah."
        )

        sys.exit(1)

    # ========================================================
    # CHUNK EMBEDDING
    # ========================================================

    print(
        "\n--- CHUNK EMBEDDING ---"
    )

    process_chunks(
        chunks
    )

    # ========================================================
    # ENTITY / ALIAS / SUBTOPIC
    # ========================================================

    print(
        "\n--- ENTITY / ALIAS / SUBTOPIC ---"
    )

    entities = collect_entities(
        sections
    )

    print(
        f"Entity       : {len(entities)}"
    )

    counts = process_entities(
        entities
    )

    # ========================================================
    # DATABASE STATS
    # ========================================================

    print(
        "\n--- DATABASE ---"
    )

    for name, count in database_stats().items():

        print(
            f"{name}: {count}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\nData baru/berubah:"
    )

    print(
        f"Entity   : {counts[0]}"
    )

    print(
        f"Subtopic : {counts[1]}"
    )

    print(
        f"Alias    : {counts[2]}"
    )

    # ========================================================
    # KNOWLEDGE INDEX
    # ========================================================

    print(
        "\n--- KNOWLEDGE INDEX ---"
    )

    build_knowledge_index(
        sections,
        chunks
    )

    print(
        "\nPIPELINE DATABASE SELESAI"
    )


if __name__ == "__main__":
    main()