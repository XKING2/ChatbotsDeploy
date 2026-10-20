
import json
import os

from config import KNOWLEDGE_INDEX_PATH
from database.hashing import compute_hash
from database.entity_builder import (
    build_entity_embedding_text,
    build_subtopic_embedding_text,
)
from database.heading_prompt import (
    build_heading_description,
    build_heading_examples,
)

try:
    from database.chroma_db import get_aliases
except ImportError:
    get_aliases = None

try:
    from database.alias_generator import generate_subtopic_aliases_batch
except ImportError:
    generate_subtopic_aliases_batch = None


def load_existing_index():
    if not os.path.exists(KNOWLEDGE_INDEX_PATH):
        return {"entities": []}

    try:
        with open(KNOWLEDGE_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {"entities": []}

        if not isinstance(data.get("entities"), list):
            data["entities"] = []

        return data

    except (json.JSONDecodeError, OSError) as exc:
        print(f"knowledge_index.json tidak dapat dibaca ({exc}).")

        backup_path = KNOWLEDGE_INDEX_PATH + ".corrupted"

        try:
            os.replace(KNOWLEDGE_INDEX_PATH, backup_path)
            print(f"File lama dipindahkan ke: {backup_path}")
        except OSError:
            pass

        return {"entities": []}


def _index_chunks_by_subtopic(chunks):
    result = {}

    for chunk in chunks or []:
        sid = (chunk.get("subtopic_id") or "").strip()
        cid = (chunk.get("content_chunk_id") or "").strip()

        if not sid or not cid:
            continue

        result.setdefault(sid, []).append({
            "content_chunk_id": cid,
            "content_hash": chunk.get("content_hash"),
            "chunk_index": chunk.get("chunk_index"),
            "total_chunks": chunk.get("total_chunks"),
        })

    for values in result.values():
        values.sort(key=lambda x: x.get("chunk_index") or 0)

    return result


def _clean_aliases(aliases, original_name=None):
    """Bersihkan alias, hilangkan duplikat, dan kecualikan nama aslinya."""
    cleaned = []
    seen = set()

    if original_name:
        seen.add(original_name.strip().casefold())

    if not isinstance(aliases, list):
        return cleaned

    for alias in aliases:
        if not isinstance(alias, str):
            continue

        alias = alias.strip()
        key = alias.casefold()

        if alias and key not in seen:
            cleaned.append(alias)
            seen.add(key)

    return cleaned


def _aliases_from_chroma(entity_id):
    if get_aliases is None:
        return []

    try:
        data = get_aliases(entity_id) or {}
        metadatas = data.get("metadatas") or []

        aliases = []

        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue

            value = (metadata.get("alias") or "").strip()

            if value:
                aliases.append(value)

        return _clean_aliases(aliases)

    except Exception as exc:
        print(
            f"Peringatan: gagal membaca alias Chroma "
            f"untuk {entity_id}: {exc}"
        )
        return []


def _merge_aliases(entity_id, old_entry=None):
    """Prioritaskan alias Chroma; gunakan JSON lama jika tidak tersedia."""
    aliases = _aliases_from_chroma(entity_id)

    if not aliases and isinstance(old_entry, dict):
        aliases = old_entry.get("aliases", [])

    entity_name = (
        old_entry.get("entity")
        if isinstance(old_entry, dict)
        else None
    )

    return _clean_aliases(aliases, entity_name)


def _build_entities_from_sections(sections):
    entities = {}

    for section in sections:
        entity = (section.get("entity") or "").strip()
        entity_id = (section.get("entity_id") or "").strip()

        if entity and entity_id:
            entities[entity_id] = {
                "entity_id": entity_id,
                "entity": entity,
            }

    return entities


def _generate_subtopic_aliases(entity_id, entity, grouped):
    """
    Hasil:
    {
        "entity_id::subtopic_id": ["alias 1", "alias 2"]
    }
    """
    if not grouped or generate_subtopic_aliases_batch is None:
        return {}

    alias_inputs = [
        {
            "entity_id": entity_id,
            "entity": entity,
            "subtopic_id": sid,
            "subtopic": item["subtopic"],
        }
        for sid, item in grouped.items()
    ]

    try:
        result = generate_subtopic_aliases_batch(alias_inputs)

        return result if isinstance(result, dict) else {}

    except Exception as exc:
        print(
            f"[IndexBuilder] Gagal generate alias subtopic "
            f"untuk entity '{entity}': {exc}"
        )
        return {}


def _build_subtopics_for_entity(
    entity_id,
    entity,
    sections,
    chunks_by_subtopic,
    old_entry=None,
):
    old_by_id = {}

    if isinstance(old_entry, dict):
        for item in old_entry.get("subtopics", []) or []:
            if isinstance(item, dict) and item.get("subtopic_id"):
                old_by_id[item["subtopic_id"]] = item

    grouped = {}

    for section in sections:
        if (section.get("entity") or "").strip() != entity:
            continue

        name = (section.get("subtopic") or "").strip()
        sid = (section.get("subtopic_id") or "").strip()

        if not name or not sid:
            continue

        item = grouped.setdefault(sid, {
            "subtopic_id": sid,
            "subtopic": name,
            "_content_parts": [],
        })

        content = (section.get("content") or "").strip()

        if content:
            item["_content_parts"].append(content)

    # Generate alias untuk subtopic entity ini.
    generated_aliases = _generate_subtopic_aliases(
        entity_id=entity_id,
        entity=entity,
        grouped=grouped,
    )

    results = []

    for sid, item in grouped.items():
        old = old_by_id.get(sid, {})
        subtopic_name = item["subtopic"]

        alias_key = f"{entity_id}::{sid}"
        new_aliases = generated_aliases.get(alias_key, [])

        # Jika alias baru kosong, pertahankan alias lama.
        if not new_aliases:
            new_aliases = old.get("aliases", [])

        clean_aliases = _clean_aliases(
            new_aliases,
            original_name=subtopic_name,
        )

        description = build_heading_description(
            subtopic_name,
            entity,
        )

        examples = build_heading_examples(
            subtopic_name,
            entity,
        )

        subtopic_text = build_subtopic_embedding_text(
            entity=entity,
            subtopic=subtopic_name,
            description=description,
            examples=examples,
        )

        content = " ".join(item["_content_parts"]).strip()

        results.append({
            "subtopic_id": sid,
            "subtopic": subtopic_name,
            "aliases": clean_aliases,
            "subtopic_hash": compute_hash(content),
            "description": description,
            "examples": examples,
            "embedding_text": subtopic_text,
            "content_chunks": chunks_by_subtopic.get(sid, []),
        })

        print(
            f"  SUBTOPIC: {subtopic_name} "
            f"| aliases={len(clean_aliases)}"
        )

    # Pertahankan subtopic lama yang tidak ada di section saat ini.
    current_ids = {item["subtopic_id"] for item in results}

    for sid, old in old_by_id.items():
        if sid not in current_ids:
            results.append(old)
            print(
                f"  SUBTOPIC LAMA DIPERTAHANKAN: "
                f"{old.get('subtopic', sid)}"
            )

    results.sort(
        key=lambda x: (x.get("subtopic") or "").casefold()
    )

    return results


def build_knowledge_index(sections, chunks):
    if not sections:
        print("Tidak ada section. Knowledge index tidak diubah.")
        return

    existing = load_existing_index()

    old_by_id = {
        item.get("entity_id"): item
        for item in existing.get("entities", [])
        if isinstance(item, dict) and item.get("entity_id")
    }

    entities = _build_entities_from_sections(sections)
    chunks_by_subtopic = _index_chunks_by_subtopic(chunks)

    entity_index = []

    for entity_id, data in entities.items():
        entity = data["entity"]
        old = old_by_id.get(entity_id)

        aliases = _merge_aliases(entity_id, old)

        subtopics = _build_subtopics_for_entity(
            entity_id=entity_id,
            entity=entity,
            sections=sections,
            chunks_by_subtopic=chunks_by_subtopic,
            old_entry=old,
        )

        subtopic_names = [
            item.get("subtopic", "")
            for item in subtopics
        ]

        entity_text = build_entity_embedding_text(
            entity,
            aliases=aliases,
            subtopics=subtopic_names,
        )

        entry = {
            "entity_id": entity_id,
            "entity": entity,
            "entity_hash": compute_hash(entity_text),
            "aliases": aliases,
            "aliases_generated": bool(aliases),
            "embedding_text": entity_text,
            "subtopics": subtopics,
        }

        entity_index.append(entry)

        alias_total = sum(
            len(item.get("aliases", []))
            for item in subtopics
        )

        print(
            f"ENTITY: {entity} "
            f"| aliases={len(aliases)} "
            f"| subtopics={len(subtopics)} "
            f"| subtopic_aliases={alias_total}"
        )

    # Pertahankan entity lama yang tidak ada dalam batch section.
    current_ids = set(entities)

    for entity_id, old in old_by_id.items():
        if entity_id not in current_ids:
            entity_index.append(old)
            print(
                f"ENTITY LAMA DIPERTAHANKAN: "
                f"{old.get('entity', entity_id)}"
            )

    entity_index.sort(
        key=lambda x: (x.get("entity") or "").casefold()
    )

    knowledge = {"entities": entity_index}

    directory = os.path.dirname(KNOWLEDGE_INDEX_PATH)

    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = KNOWLEDGE_INDEX_PATH + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            knowledge,
            f,
            ensure_ascii=False,
            indent=4,
        )

    os.replace(tmp_path, KNOWLEDGE_INDEX_PATH)

    total_subtopics = sum(
        len(item.get("subtopics", []))
        for item in entity_index
    )

    total_subtopic_aliases = sum(
        len(subtopic.get("aliases", []))
        for item in entity_index
        for subtopic in item.get("subtopics", [])
    )

    print("=" * 70)
    print(f"Jumlah entity: {len(entity_index)}")
    print(f"Jumlah subtopic: {total_subtopics}")
    print(f"Jumlah alias subtopic: {total_subtopic_aliases}")
    print(f"Knowledge index tersimpan: {KNOWLEDGE_INDEX_PATH}")
    print("=" * 70)
