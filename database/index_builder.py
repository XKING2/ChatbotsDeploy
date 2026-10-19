import json
import os

from config import KNOWLEDGE_INDEX_PATH
from database.hashing import compute_hash


def load_existing_index():
    if not os.path.exists(KNOWLEDGE_INDEX_PATH):
        return {"entities": []}

    try:
        with open(KNOWLEDGE_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {"entities": []}

        entities = data.get("entities")

        if not isinstance(entities, list):
            data["entities"] = []

        return data

    except json.JSONDecodeError as exc:
        print(
            f"knowledge_index.json rusak/tidak valid ({exc}), "
            "akan dibangun ulang dari nol."
        )

        backup_path = KNOWLEDGE_INDEX_PATH + ".corrupted"

        try:
            os.replace(
                KNOWLEDGE_INDEX_PATH,
                backup_path
            )
            print(
                f"File rusak dipindahkan ke: "
                f"{backup_path}"
            )
        except OSError:
            pass

        return {"entities": []}


def _index_chunks_by_subtopic(chunks):
    by_subtopic = {}

    for chunk in chunks:
        subtopic_id = (
            chunk.get("subtopic_id") or ""
        ).strip()

        if not subtopic_id:
            continue

        content_chunk_id = (
            chunk.get("content_chunk_id") or ""
        ).strip()

        if not content_chunk_id:
            continue

        by_subtopic.setdefault(
            subtopic_id,
            []
        ).append({
            "content_chunk_id": content_chunk_id,
            "content_hash": chunk.get(
                "content_hash"
            ),
            "chunk_index": chunk.get(
                "chunk_index"
            ),
            "total_chunks": chunk.get(
                "total_chunks"
            )
        })

    for subtopic_id in by_subtopic:
        by_subtopic[subtopic_id].sort(
            key=lambda item: (
                item.get("chunk_index") or 0
            )
        )

    return by_subtopic


def _build_subtopics_for_entity(
    entity,
    sections,
    chunks_by_subtopic,
    old_entry=None
):
    subtopics = {}

    for section in sections:
        if section.get("entity") != entity:
            continue

        subtopic = (
            section.get("subtopic") or ""
        ).strip()

        subtopic_id = (
            section.get("subtopic_id") or ""
        ).strip()

        if not subtopic or not subtopic_id:
            continue

        if subtopic_id not in subtopics:
            subtopics[subtopic_id] = {
                "subtopic_id": subtopic_id,
                "subtopic": subtopic,
                "_content_parts": []
            }

        content = (
            section.get("content") or ""
        ).strip()

        if content:
            subtopics[
                subtopic_id
            ]["_content_parts"].append(content)

    old_subtopics_by_id = {}

    if old_entry:
        old_subtopics = old_entry.get(
            "subtopics",
            []
        )

        if isinstance(old_subtopics, list):
            for subtopic in old_subtopics:
                if not isinstance(subtopic, dict):
                    continue

                subtopic_id = (
                    subtopic.get(
                        "subtopic_id"
                    ) or ""
                ).strip()

                if subtopic_id:
                    old_subtopics_by_id[
                        subtopic_id
                    ] = subtopic

    result = []

    for subtopic_id, entry in subtopics.items():
        content = " ".join(
            entry["_content_parts"]
        ).strip()

        subtopic_hash = compute_hash(
            content
        )

        result.append({
            "subtopic_id": subtopic_id,
            "subtopic": entry["subtopic"],
            "subtopic_hash": subtopic_hash,
            "content_chunks": chunks_by_subtopic.get(
                subtopic_id,
                []
            )
        })

    current_subtopic_ids = set(
        subtopics.keys()
    )

    for subtopic_id, old_subtopic in (
        old_subtopics_by_id.items()
    ):
        if subtopic_id in current_subtopic_ids:
            continue

        result.append(old_subtopic)

    result.sort(
        key=lambda item: (
            item.get("subtopic") or ""
        ).lower()
    )

    return result


def _build_aliases_for_entity(
    entity,
    entity_id,
    old_entry=None
):
    if not old_entry:
        return []

    aliases = old_entry.get(
        "aliases",
        []
    )

    if not isinstance(aliases, list):
        return []

    return aliases


def _build_entities_from_sections(sections):
    entities = {}

    for section in sections:
        entity = (
            section.get("entity") or ""
        ).strip()

        entity_id = (
            section.get("entity_id") or ""
        ).strip()

        if not entity or not entity_id:
            continue

        entities[entity_id] = {
            "entity_id": entity_id,
            "entity": entity
        }

    return entities


def _build_entity_hash(entity, subtopics):
    entity_hash_source = "\n".join([
        entity,
        *[
            subtopic.get(
                "subtopic",
                ""
            )
            for subtopic in subtopics
        ]
    ])

    return compute_hash(
        entity_hash_source
    )


def build_knowledge_index(
    sections,
    chunks
):
    if not sections:
        print(
            "Tidak ada section. "
            "Knowledge index tidak diubah."
        )
        return

    existing = load_existing_index()

    existing_entities_by_id = {
        entity.get("entity_id"): entity
        for entity in existing.get(
            "entities",
            []
        )
        if isinstance(entity, dict)
        and entity.get("entity_id")
    }

    entities = _build_entities_from_sections(
        sections
    )

    chunks_by_subtopic = (
        _index_chunks_by_subtopic(chunks)
    )

    print("\n" + "=" * 70)
    print("MEMBANGUN KNOWLEDGE INDEX")
    print("=" * 70)

    entity_index = []

    current_entity_ids = set(
        entities.keys()
    )

    for entity_id, entity_data in (
        entities.items()
    ):
        entity = entity_data["entity"]

        old_entry = (
            existing_entities_by_id.get(
                entity_id
            )
        )

        subtopics = (
            _build_subtopics_for_entity(
                entity=entity,
                sections=sections,
                chunks_by_subtopic=(
                    chunks_by_subtopic
                ),
                old_entry=old_entry
            )
        )

        aliases = (
            _build_aliases_for_entity(
                entity=entity,
                entity_id=entity_id,
                old_entry=old_entry
            )
        )

        entity_hash = _build_entity_hash(
            entity,
            subtopics
        )

        entry = {
            "entity_id": entity_id,
            "entity": entity,
            "entity_hash": entity_hash,
            "aliases": aliases,
            "subtopics": subtopics
        }

        entity_index.append(entry)

        print(
            f"\nENTITY: {entity}"
        )
        print(
            f"  ID       : {entity_id}"
        )
        print(
            f"  Alias    : {len(aliases)}"
        )
        print(
            f"  Subtopic : {len(subtopics)}"
        )

        for subtopic in subtopics:
            print(
                f"    - {subtopic['subtopic']} "
                f"({len(subtopic.get('content_chunks', []))} chunk)"
            )

    untouched_entity_ids = (
        set(existing_entities_by_id.keys())
        - current_entity_ids
    )

    for entity_id in untouched_entity_ids:
        old_entry = (
            existing_entities_by_id[
                entity_id
            ]
        )

        entity_index.append(
            old_entry
        )

        print(
            f"\nENTITY LAMA DIPERTAHANKAN: "
            f"{old_entry.get('entity', entity_id)}"
        )

    entity_index.sort(
        key=lambda item: (
            item.get("entity", "")
        ).lower()
    )

    knowledge = {
        "entities": entity_index
    }

    tmp_path = (
        KNOWLEDGE_INDEX_PATH
        + ".tmp"
    )

    with open(
        tmp_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            knowledge,
            f,
            ensure_ascii=False,
            indent=4
        )

    os.replace(
        tmp_path,
        KNOWLEDGE_INDEX_PATH
    )

    print("\n" + "=" * 70)
    print(
        f"Jumlah Entity total : "
        f"{len(entity_index)}"
    )
    print(
        "Knowledge Index berhasil "
        "dibuat/diperbarui"
    )
    print(
        KNOWLEDGE_INDEX_PATH
    )
    print("=" * 70)