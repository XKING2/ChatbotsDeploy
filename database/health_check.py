import json
import os

from config import (
    CHROMA_DB_PATH,
    CHUNK_COLLECTION_NAME,
    ENTITY_COLLECTION_NAME
)

from database.chroma_db import (
    client,
    chunk_collection,
    entity_collection
)


EXPECTED_EMBEDDING_DIMENSION = 3072
from config import DATA_DIR

KNOWLEDGE_INDEX_PATH = os.path.join(
    DATA_DIR,
    "knowledge_index.json"
)

errors = []
warnings = []


def ok(message):
    print(f"✓ {message}")


def fail(message):
    print(f"✗ {message}")
    errors.append(message)


def warn(message):
    print(f"! {message}")
    warnings.append(message)


def check_collections():
    print("\n[1] COLLECTIONS")

    try:
        chunk_count = chunk_collection.count()
        entity_count = entity_collection.count()

        ok(f"{CHUNK_COLLECTION_NAME}: {chunk_count}")
        ok(f"{ENTITY_COLLECTION_NAME}: {entity_count}")

        return chunk_count, entity_count

    except Exception as e:
        fail(f"Gagal membaca collection: {e}")
        return 0, 0


def load_entity_data():
    return entity_collection.get(
        include=["documents", "metadatas", "embeddings"]
    )


def load_chunk_data():
    return chunk_collection.get(
        include=["documents", "metadatas", "embeddings"]
    )


def check_entity_records(entity_data):
    print("\n[2] ENTITY RECORDS")

    ids = entity_data.get("ids", [])
    documents = entity_data.get("documents", [])
    metadatas = entity_data.get("metadatas", [])
    embeddings = entity_data.get("embeddings", [])

    if not ids:
        fail("Tidak ada data pada entity collection.")
        return

    entity_ids = set()
    actual_entity_ids = set()
    subtopic_ids = set()
    alias_ids = set()

    type_counts = {
        "entity": 0,
        "subtopic": 0,
        "alias": 0
    }

    for index, item_id in enumerate(ids):
        metadata = metadatas[index] or {}
        item_type = metadata.get("type")

        if item_type not in type_counts:
            fail(
                f"ID {item_id} memiliki type tidak valid: "
                f"{item_type}"
            )
            continue

        type_counts[item_type] += 1

        entity_id = metadata.get("entity_id")

        if item_type == "entity":
            actual_entity_ids.add(item_id)

            if not metadata.get("entity"):
                fail(f"Entity {item_id} tidak memiliki nama entity.")

            if not metadata.get("content_hash"):
                fail(f"Entity {item_id} tidak memiliki content_hash.")

        elif item_type == "subtopic":
            subtopic_ids.add(item_id)

            if not entity_id:
                fail(f"Subtopic {item_id} tidak memiliki entity_id.")

            if not metadata.get("subtopic"):
                fail(f"Subtopic {item_id} tidak memiliki nama subtopic.")

            if not metadata.get("content_hash"):
                fail(f"Subtopic {item_id} tidak memiliki content_hash.")

        elif item_type == "alias":
            alias_ids.add(item_id)

            if not entity_id:
                fail(f"Alias {item_id} tidak memiliki entity_id.")

            if not metadata.get("alias"):
                fail(f"Alias {item_id} tidak memiliki nilai alias.")

            if not metadata.get("content_hash"):
                fail(f"Alias {item_id} tidak memiliki content_hash.")

        if item_id in entity_ids:
            fail(f"Duplicate ID ditemukan: {item_id}")

        entity_ids.add(item_id)

    ok(f"Entity      : {type_counts['entity']}")
    ok(f"Subtopic    : {type_counts['subtopic']}")
    ok(f"Alias       : {type_counts['alias']}")
    ok(f"Total record: {len(ids)}")

    return {
        "all_ids": entity_ids,
        "entity_ids": actual_entity_ids,
        "subtopic_ids": subtopic_ids,
        "alias_ids": alias_ids
    }


def check_entity_embeddings(entity_data):
    print("\n[3] ENTITY EMBEDDINGS")

    embeddings = entity_data.get("embeddings")
    ids = entity_data.get("ids", [])

    if embeddings is None or len(embeddings) == 0:
        fail("Tidak ditemukan embedding pada entity collection.")
        return

    valid = 0

    for item_id, embedding in zip(ids, embeddings):
        if embedding is None:
            fail(f"Embedding kosong: {item_id}")
            continue

        dimension = len(embedding)

        if dimension != EXPECTED_EMBEDDING_DIMENSION:
            fail(
                f"Dimensi embedding salah: {item_id} "
                f"(ditemukan {dimension}, "
                f"expected {EXPECTED_EMBEDDING_DIMENSION})"
            )
            continue

        valid += 1

    if valid == len(ids):
        ok(
            f"Semua embedding valid: "
            f"{valid}/{len(ids)} "
            f"({EXPECTED_EMBEDDING_DIMENSION} dimensi)"
        )


def check_chunk_records(chunk_data, entity_info):
    print("\n[4] CHUNK RECORDS")

    ids = chunk_data.get("ids", [])
    documents = chunk_data.get("documents", [])
    metadatas = chunk_data.get("metadatas", [])
    embeddings = chunk_data.get("embeddings", [])

    if not ids:
        fail("Tidak ada chunk pada chunk collection.")
        return

    required_fields = [
        "entity_id",
        "entity",
        "subtopic_id",
        "subtopic",
        "content_chunk_id",
        "content_hash",
        "chunk_index",
        "total_chunks"
    ]

    valid_chunks = 0
    referenced_entities = set()
    referenced_subtopics = set()

    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}

        if metadata.get("type") != "chunk":
            fail(
                f"Chunk {chunk_id} memiliki type "
                f"tidak valid: {metadata.get('type')}"
            )

        missing = [
            field
            for field in required_fields
            if metadata.get(field) is None
        ]

        if missing:
            fail(
                f"Chunk {chunk_id} kehilangan metadata: "
                f"{', '.join(missing)}"
            )
            continue

        if not documents[index]:
            fail(f"Document/retrieval_text kosong: {chunk_id}")
            continue

        referenced_entities.add(metadata["entity_id"])
        referenced_subtopics.add(metadata["subtopic_id"])

        if metadata["content_chunk_id"] != chunk_id:
            fail(
                f"content_chunk_id tidak cocok dengan ID "
                f"Chroma: {chunk_id}"
            )
            continue

        valid_chunks += 1

    ok(f"Chunk valid: {valid_chunks}/{len(ids)}")

    missing_entities = (
        referenced_entities - entity_info["entity_ids"]
    )

    if missing_entities:
        for entity_id in sorted(missing_entities):
            fail(
                f"Chunk mereferensikan entity yang tidak ada: "
                f"{entity_id}"
            )
    else:
        ok(
            f"Entity relation valid: "
            f"{len(referenced_entities)} entity"
        )

    missing_subtopics = (
        referenced_subtopics - entity_info["subtopic_ids"]
    )

    if missing_subtopics:
        for subtopic_id in sorted(missing_subtopics):
            fail(
                f"Chunk mereferensikan subtopic yang tidak ada: "
                f"{subtopic_id}"
            )
    else:
        ok(
            f"Subtopic relation valid: "
            f"{len(referenced_subtopics)} subtopic"
        )

    return {
        "chunk_ids": set(ids),
        "subtopic_ids": referenced_subtopics
    }


def check_chunk_embeddings(chunk_data):
    print("\n[5] CHUNK EMBEDDINGS")

    ids = chunk_data.get("ids", [])
    embeddings = chunk_data.get("embeddings")

    if embeddings is None or len(embeddings) == 0:
        fail("Tidak ditemukan embedding pada chunk collection.")
        return

    valid = 0

    for item_id, embedding in zip(ids, embeddings):
        if embedding is None:
            fail(f"Embedding kosong: {item_id}")
            continue

        dimension = len(embedding)

        if dimension != EXPECTED_EMBEDDING_DIMENSION:
            fail(
                f"Dimensi embedding salah: {item_id} "
                f"(ditemukan {dimension}, "
                f"expected {EXPECTED_EMBEDDING_DIMENSION})"
            )
            continue

        valid += 1

    if valid == len(ids):
        ok(
            f"Semua embedding chunk valid: "
            f"{valid}/{len(ids)} "
            f"({EXPECTED_EMBEDDING_DIMENSION} dimensi)"
        )

def check_subtopic_relations(entity_data, chunk_data):
    print("\n[6] SUBTOPIC RELATIONS")

    entity_ids = entity_data.get("ids", [])
    entity_metadatas = entity_data.get("metadatas", [])

    chunk_metadatas = chunk_data.get("metadatas", [])

    entity_subtopics = {}

    for item_id, metadata in zip(entity_ids, entity_metadatas):
        metadata = metadata or {}

        if metadata.get("type") != "subtopic":
            continue

        subtopic_id = metadata.get("subtopic_id", item_id)
        entity_id = metadata.get("entity_id")

        entity_subtopics[subtopic_id] = entity_id

    chunk_subtopics = set()

    for metadata in chunk_metadatas:
        metadata = metadata or {}

        if metadata.get("type") == "chunk":
            subtopic_id = metadata.get("subtopic_id")

            if subtopic_id:
                chunk_subtopics.add(subtopic_id)

    orphan_subtopics = (
        set(entity_subtopics.keys()) - chunk_subtopics
    )

    if orphan_subtopics:
        for subtopic_id in sorted(orphan_subtopics):
            warn(
                f"Subtopic tidak memiliki chunk: "
                f"{subtopic_id}"
            )
    else:
        ok(
            f"Semua subtopic memiliki chunk: "
            f"{len(entity_subtopics)}"
        )


def check_alias_relations(entity_data):
    print("\n[7] ALIAS RELATIONS")

    ids = entity_data.get("ids", [])
    metadatas = entity_data.get("metadatas", [])

    entity_ids = set()

    for item_id, metadata in zip(ids, metadatas):
        metadata = metadata or {}

        if metadata.get("type") == "entity":
            entity_ids.add(item_id)

    aliases_by_entity = {}

    for item_id, metadata in zip(ids, metadatas):
        metadata = metadata or {}

        if metadata.get("type") != "alias":
            continue

        entity_id = metadata.get("entity_id")

        if entity_id not in entity_ids:
            fail(
                f"Alias {item_id} memiliki parent entity "
                f"yang tidak ada: {entity_id}"
            )
            continue

        aliases_by_entity.setdefault(entity_id, []).append(
            metadata.get("alias")
        )

    orphan_count = 0
    duplicate_count = 0

    for entity_id, aliases in aliases_by_entity.items():
        clean_aliases = [
            alias.strip().lower()
            for alias in aliases
            if isinstance(alias, str) and alias.strip()
        ]

        duplicates = (
            len(clean_aliases) -
            len(set(clean_aliases))
        )

        if duplicates:
            duplicate_count += duplicates
            warn(
                f"Duplicate alias pada {entity_id}: "
                f"{duplicates}"
            )

    if not orphan_count:
        ok(
            f"Semua alias memiliki parent entity: "
            f"{sum(len(v) for v in aliases_by_entity.values())}"
        )

    if duplicate_count == 0:
        ok("Tidak ditemukan duplicate alias dalam entity yang sama.")


def load_knowledge_index():
    print("\n[8] KNOWLEDGE INDEX")

    if not os.path.exists(KNOWLEDGE_INDEX_PATH):
        warn(
            f"knowledge_index.json tidak ditemukan: "
            f"{KNOWLEDGE_INDEX_PATH}"
        )
        return None

    try:
        with open(
            KNOWLEDGE_INDEX_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            fail("knowledge_index.json bukan object JSON.")
            return None

        entities = data.get("entities")

        if not isinstance(entities, list):
            fail("Field 'entities' tidak berupa list.")
            return None

        ok(
            f"knowledge_index.json valid: "
            f"{len(entities)} entity"
        )

        return data

    except json.JSONDecodeError as e:
        fail(
            f"knowledge_index.json tidak valid: {e}"
        )
    except Exception as e:
        fail(
            f"Gagal membaca knowledge_index.json: {e}"
        )

    return None


def check_index_against_chroma(index_data, entity_data, chunk_data):
    print("\n[9] INDEX ↔ CHROMADB")

    if not index_data:
        warn("Cross-check dilewati karena index tidak tersedia.")
        return

    chroma_entity_ids = set()

    for item_id, metadata in zip(
        entity_data.get("ids", []),
        entity_data.get("metadatas", [])
    ):
        metadata = metadata or {}

        if metadata.get("type") == "entity":
            chroma_entity_ids.add(item_id)

    chroma_chunk_ids = set(chunk_data.get("ids", []))

    index_entity_ids = set()
    index_chunk_ids = set()

    for entity in index_data.get("entities", []):
        entity_id = entity.get("entity_id")

        if entity_id:
            index_entity_ids.add(entity_id)

        for subtopic in entity.get("subtopics", []):
            for chunk in subtopic.get("content_chunks", []):
                chunk_id = chunk.get("content_chunk_id")

                if chunk_id:
                    index_chunk_ids.add(chunk_id)

    missing_entities = index_entity_ids - chroma_entity_ids
    missing_chunks = index_chunk_ids - chroma_chunk_ids

    extra_entities = chroma_entity_ids - index_entity_ids
    extra_chunks = chroma_chunk_ids - index_chunk_ids

    if missing_entities:
        fail(
            "Entity di index tidak ditemukan di ChromaDB: "
            f"{len(missing_entities)}"
        )
    else:
        ok(
            f"Entity index ↔ ChromaDB cocok: "
            f"{len(index_entity_ids)}"
        )

    if missing_chunks:
        fail(
            "Chunk di index tidak ditemukan di ChromaDB: "
            f"{len(missing_chunks)}"
        )
    else:
        ok(
            f"Chunk index ↔ ChromaDB cocok: "
            f"{len(index_chunk_ids)}"
        )

    if extra_entities:
        warn(
            f"ChromaDB memiliki entity yang belum ada di index: "
            f"{len(extra_entities)}"
        )

    if extra_chunks:
        warn(
            f"ChromaDB memiliki chunk yang belum ada di index: "
            f"{len(extra_chunks)}"
        )


def print_result():
    print("\n" + "=" * 70)

    if errors:
        print("RESULT: FAIL")
        print("=" * 70)
        print(f"Error ditemukan: {len(errors)}")

        for error in errors:
            print(f"  - {error}")

        if warnings:
            print(f"Warning: {len(warnings)}")

        return 1

    if warnings:
        print("RESULT: PASS WITH WARNING")
        print("=" * 70)
        print(f"Warning ditemukan: {len(warnings)}")
        return 0

    print("RESULT: PASS")
    print("=" * 70)
    print("Database siap untuk tahap retrieval.")
    return 0


def main():
    print("=" * 70)
    print("KNOWLEDGE BASE HEALTH CHECK")
    print("=" * 70)

    print(f"ChromaDB: {CHROMA_DB_PATH}")

    chunk_count, entity_count = check_collections()

    entity_data = load_entity_data()
    chunk_data = load_chunk_data()

    entity_info = check_entity_records(entity_data)

    if entity_info:
        check_entity_embeddings(entity_data)

        chunk_info = check_chunk_records(
            chunk_data,
            entity_info
        )

        check_chunk_embeddings(chunk_data)
        check_subtopic_relations(
            entity_data,
            chunk_data
        )
        check_alias_relations(entity_data)

        index_data = load_knowledge_index()

        check_index_against_chroma(
            index_data,
            entity_data,
            chunk_data
        )

    return print_result()


if __name__ == "__main__":
    raise SystemExit(main())