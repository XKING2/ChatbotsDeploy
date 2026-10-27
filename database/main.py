import json
import os
import sys
import time

from config import PDF_FOLDER

from database.parser import parse_pdf
from database.preprocessing import preprocess_blocks
from database.section_builder import build_sections
from database.chunker import chunk_sections
from database.embedding import create_embedding, create_embeddings_batch
from database.hashing import compute_hash

from database.entity_builder import (
    collect_entities,
    build_entity_embedding_text,
    build_subtopic_embedding_text
)

from database.alias_generator import (
    generate_aliases_batch,
    generate_subtopic_aliases_batch
)

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


# ============================================================
# ALIAS: PENGECEKAN SEBELUM GENERATE
# ============================================================
# Aturan:
#   - Alias hanya di-generate untuk heading (entity) atau
#     subheading (subtopic) BARU yang belum punya alias.
#   - Heading/subheading yang sudah punya alias TIDAK pernah
#     di-generate ulang, dan alias lamanya tidak diubah.
#   - Update isi data (chunk) saja tidak memicu generate alias.
#   - Jika generate gagal (API error), tidak ada yang disimpan
#     dan akan dicoba lagi pada run berikutnya.
#   - Jika Gemini sengaja mengembalikan daftar kosong, item
#     dicatat di file state agar tidak ditanya terus-menerus.

def _norm(text):
    return " ".join(str(text or "").casefold().split())


def _state_path():
    """
    File penanda alias yang sudah dicek. Letaknya di folder data
    Chroma jika config menyediakan path-nya (agar ikut volume),
    atau bisa diatur lewat ALIAS_STATE_PATH di config.py.
    """
    import config

    explicit = getattr(config, "ALIAS_STATE_PATH", None)
    if explicit:
        return explicit

    for name in ("CHROMA_PATH", "CHROMA_DIR", "CHROMA_DB_PATH", "DB_PATH"):
        value = getattr(config, name, None)
        if isinstance(value, str) and os.path.isdir(value):
            return os.path.join(value, "alias_checked.json")

    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "alias_checked.json"
    )


def _load_checked():
    try:
        with open(_state_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("checked", []))
    except (OSError, ValueError, AttributeError):
        return set()


def _save_checked(checked):
    try:
        path = _state_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"checked": sorted(checked)},
                f,
                ensure_ascii=False,
                indent=0
            )
        os.replace(tmp, path)
    except OSError as exc:
        print(f"Peringatan: state alias tidak tersimpan: {exc}")


def _clean_aliases(aliases):
    if not isinstance(aliases, list):
        return []
    return list(dict.fromkeys(
        alias.strip() for alias in aliases
        if isinstance(alias, str) and alias.strip()
    ))


def read_existing_aliases(entity_id):
    """
    Satu kali baca Chroma per entity. Mengembalikan:
      entity_aliases : alias tingkat entity
      by_sub_id      : {subtopic_id: [alias]}
      by_sub_name    : {nama_subtopic_ternormalisasi: [alias]}
    Alias dianggap alias subtopic jika alias_scope="subtopic",
    atau punya subtopic_id, atau id-nya memuat "::subtopic::".
    """
    data = get_aliases(entity_id) or {}
    metadatas = data.get("metadatas") or []
    ids = data.get("ids") or []

    entity_aliases = []
    by_sub_id = {}
    by_sub_name = {}

    for index, metadata in enumerate(metadatas):
        if not isinstance(metadata, dict):
            continue

        alias = metadata.get("alias")
        if not isinstance(alias, str) or not alias.strip():
            continue
        alias = alias.strip()

        record_id = ids[index] if index < len(ids) else ""
        sub_id = metadata.get("subtopic_id")

        is_subtopic = (
            metadata.get("alias_scope") == "subtopic"
            or bool(sub_id)
            or "::subtopic::" in str(record_id)
        )

        if not is_subtopic:
            entity_aliases.append(alias)
            continue

        if sub_id:
            by_sub_id.setdefault(str(sub_id), []).append(alias)

        sub_name = _norm(metadata.get("subtopic"))
        if sub_name:
            by_sub_name.setdefault(sub_name, []).append(alias)

    return (
        _clean_aliases(entity_aliases),
        {k: _clean_aliases(v) for k, v in by_sub_id.items()},
        {k: _clean_aliases(v) for k, v in by_sub_name.items()},
    )


def load_aliases(entities, existing_cache=None):
    """
    Alias entity. Hanya entity tanpa alias (heading baru) yang
    di-generate. Mengembalikan (aliases_by_entity, new_entities).
    """
    cache = existing_cache if existing_cache is not None else {}
    checked = _load_checked()

    aliases_by_entity = {}
    pending = []
    skipped_checked = 0

    for item in entities:
        entity_id = item["entity_id"]
        entity = item["entity"]

        if entity_id not in cache:
            cache[entity_id] = read_existing_aliases(entity_id)

        aliases = cache[entity_id][0]

        if aliases:
            aliases_by_entity[entity] = aliases
        elif f"entity::{entity_id}" in checked:
            aliases_by_entity[entity] = []
            skipped_checked += 1
        else:
            pending.append(item)

    print(
        f"Entity sudah memiliki alias : "
        f"{len(entities) - len(pending) - skipped_checked}"
    )
    print(
        f"Entity sudah dicek (kosong) : {skipped_checked}"
    )
    print(
        f"Entity perlu generate alias  : {len(pending)}"
    )

    new_entities = set()

    if pending:
        names = [item["entity"] for item in pending]
        generated = generate_aliases_batch(names) or {}

        for item in pending:
            entity = item["entity"]

            if entity not in generated:
                print(f"Generate gagal, dicoba lagi nanti: {entity}")
                continue

            aliases = _clean_aliases(generated[entity])
            aliases_by_entity[entity] = aliases
            new_entities.add(entity)

            if not aliases:
                checked.add(f"entity::{item['entity_id']}")

        _save_checked(checked)

    return aliases_by_entity, new_entities


def load_subtopic_aliases(entities, existing_cache=None):
    """
    Alias subtopic. Hanya subheading yang belum punya alias
    (subheading baru) yang di-generate. Pencocokan memakai
    subtopic_id, lalu nama subtopic sebagai cadangan.
    Mengembalikan (aliases_by_subtopic, new_keys).
    """
    cache = existing_cache if existing_cache is not None else {}
    checked = _load_checked()

    aliases_by_subtopic = {}
    pending = []
    existing_count = 0
    skipped_checked = 0

    for item in entities:
        entity_id = item["entity_id"]
        entity = item["entity"]

        if entity_id not in cache:
            cache[entity_id] = read_existing_aliases(entity_id)

        _, by_sub_id, by_sub_name = cache[entity_id]

        for subtopic_data in item["subtopics"].values():
            subtopic_id = subtopic_data["subtopic_id"]
            subtopic = subtopic_data["subtopic"]
            key = f"{entity_id}::{subtopic_id}"
            name_key = f"subtopic::{entity_id}::name::{_norm(subtopic)}"

            aliases = (
                by_sub_id.get(str(subtopic_id))
                or by_sub_name.get(_norm(subtopic))
                or []
            )

            if aliases:
                aliases_by_subtopic[key] = aliases
                existing_count += 1
            elif (
                f"subtopic::{key}" in checked
                or name_key in checked
            ):
                aliases_by_subtopic[key] = []
                skipped_checked += 1
            else:
                pending.append({
                    "entity_id": entity_id,
                    "entity": entity,
                    "subtopic_id": subtopic_id,
                    "subtopic": subtopic,
                    "_name_key": name_key,
                })

    print(f"Subtopic sudah memiliki alias : {existing_count}")
    print(f"Subtopic sudah dicek (kosong) : {skipped_checked}")
    print(f"Subtopic perlu generate alias  : {len(pending)}")

    new_keys = set()

    if pending:
        request_items = [
            {k: v for k, v in item.items() if k != "_name_key"}
            for item in pending
        ]
        generated = generate_subtopic_aliases_batch(request_items) or {}

        for item in pending:
            key = f'{item["entity_id"]}::{item["subtopic_id"]}'

            if key not in generated:
                print(
                    f"Generate gagal, dicoba lagi nanti: "
                    f"{item['entity']} / {item['subtopic']}"
                )
                continue

            aliases = _clean_aliases(generated[key])
            aliases_by_subtopic[key] = aliases
            new_keys.add(key)

            if not aliases:
                checked.add(f"subtopic::{key}")
                checked.add(item["_name_key"])

        _save_checked(checked)

    return aliases_by_subtopic, new_keys


def process_entities(entities):
    """Generate alias, siapkan dokumen yang berubah, lalu embed secara batch."""
    existing = get_existing_entity_hashes()
    existing_alias_cache = {}

    aliases_by_entity, new_entities = load_aliases(entities, existing_alias_cache)
    aliases_by_subtopic, new_subtopic_keys = load_subtopic_aliases(
        entities, existing_alias_cache
    )

    # Setiap job menyimpan metadata untuk ditulis ke Chroma setelah embedding.
    jobs = []
    entity_count = 0
    subtopic_count = 0
    alias_count = 0
    subtopic_alias_count = 0

    for item in entities:
        entity_id = item["entity_id"]
        entity = item["entity"]
        aliases = aliases_by_entity.get(entity, [])
        subtopics = list(item["subtopics"].values())
        subtopic_names = [x["subtopic"] for x in subtopics]

        entity_text = build_entity_embedding_text(entity, aliases, subtopic_names)
        entity_hash = compute_hash(entity_text)
        if existing.get(entity_id) != entity_hash:
            jobs.append({
                "kind": "entity", "text": entity_text, "id": entity_id,
                "entity_id": entity_id, "content_hash": entity_hash,
                "subtopic_count": len(subtopics), "alias_count": len(aliases),
            })

        for subtopic_data in subtopics:
            subtopic_id = subtopic_data["subtopic_id"]
            subtopic = subtopic_data["subtopic"]
            description = build_heading_description(subtopic, entity)
            examples = build_heading_examples(subtopic, entity)
            subtopic_text = build_subtopic_embedding_text(
                entity=entity,
                subtopic=subtopic,
                description=description,
                examples=examples,
            )
            subtopic_hash = compute_hash(subtopic_text)

            if existing.get(subtopic_id) != subtopic_hash:
                jobs.append({
                    "kind": "subtopic", "text": subtopic_text,
                    "id": subtopic_id, "entity_id": entity_id,
                    "content_hash": subtopic_hash,
                    "description": description,
                    "examples": " | ".join(examples),
                })

            # Alias subtopic hanya diproses jika alias baru saja dibuat.
            subtopic_key = f"{entity_id}::{subtopic_id}"
            if subtopic_key in new_subtopic_keys:
                for alias in aliases_by_subtopic.get(subtopic_key, []):
                    alias = alias.strip()
                    if not alias:
                        continue
                    current_id = (
                        f"{entity_id}::subtopic::{subtopic_id}"
                        f"::alias::{compute_hash(alias)[:16]}"
                    )
                    alias_text = (
                        f"Layanan\n{entity}\n\n"
                        f"Heading\n{subtopic}\n\nAlias heading\n{alias}"
                    )
                    alias_hash = compute_hash(alias_text)
                    if existing.get(current_id) == alias_hash:
                        continue
                    jobs.append({
                        "kind": "subtopic_alias", "text": alias_text,
                        "id": current_id, "entity_id": entity_id,
                        "content_hash": alias_hash, "alias": alias,
                        "subtopic_id": subtopic_id, "subtopic": subtopic,
                        "entity": entity,
                    })

        # Alias entity hanya diproses untuk entity yang baru dibuat alias-nya.
        if entity in new_entities:
            for alias in aliases:
                alias = alias.strip()
                if not alias:
                    continue
                current_id = alias_id(entity_id, alias)
                alias_text = f"Layanan\n{entity}\n\nNama lain\n{alias}"
                alias_hash = compute_hash(alias_text)
                if existing.get(current_id) == alias_hash:
                    continue
                jobs.append({
                    "kind": "entity_alias", "text": alias_text,
                    "id": current_id, "entity_id": entity_id,
                    "content_hash": alias_hash, "alias": alias,
                    "entity": entity,
                })

    if not jobs:
        print("[EMBED BATCH] Tidak ada entity/subtopic/alias baru atau berubah.")
        return (0, 0, 0, 0)

    print(f"\n[EMBED BATCH] Dokumen perlu embedding: {len(jobs)}")
    # Batasi ukuran request. Hasil tetap sesuai urutan jobs.
    BATCH_SIZE = 32
    for start in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[start:start + BATCH_SIZE]
        texts = [job["text"] for job in batch]
        vectors = create_embeddings_batch(texts, batch_size=BATCH_SIZE)

        if vectors is None or len(vectors) != len(batch):
            print(
                f"[EMBED BATCH] Batch {start + 1}-"
                f"{start + len(batch)} gagal; batch ini tidak disimpan."
            )
            continue

        for job, embedding in zip(batch, vectors):
            kind = job["kind"]
            if kind == "entity":
                add_entity_item(
                    "entity", job["id"], job["entity_id"], job["text"],
                    embedding, job["content_hash"],
                    subtopic_count=job["subtopic_count"],
                    alias_count=job["alias_count"],
                )
                entity_count += 1
            elif kind == "subtopic":
                add_entity_item(
                    "subtopic", job["id"], job["entity_id"], job["text"],
                    embedding, job["content_hash"],
                    description=job["description"],
                    examples=job["examples"],
                )
                subtopic_count += 1
                print(f"[OK] Subtopic disimpan: {job['id']}")
            elif kind == "subtopic_alias":
                add_entity_item(
                    "alias", job["id"], job["entity_id"], job["alias"],
                    embedding, job["content_hash"],
                    alias_scope="subtopic",
                    subtopic_id=job["subtopic_id"],
                    subtopic=job["subtopic"],
                    entity=job["entity"],
                    embedding_text=job["text"],
                )
                subtopic_alias_count += 1
            elif kind == "entity_alias":
                add_entity_item(
                    "alias", job["id"], job["entity_id"], job["alias"],
                    embedding, job["content_hash"],
                    alias_scope="entity",
                    entity=job["entity"],
                    embedding_text=job["text"],
                )
                alias_count += 1

            existing[job["id"]] = job["content_hash"]

        print(f"[EMBED BATCH] Selesai {min(start + len(batch), len(jobs))}/{len(jobs)}")

    return (entity_count, subtopic_count, alias_count, subtopic_alias_count)

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
        f"Alias entity   : {counts[2]}"
    )

    print(
        f"Alias subtopic : {counts[3]}"
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