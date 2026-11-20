import json
import math
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO

from config import CHROMA_DB_PATH, PDF_FOLDER

from database.parser import parse_pdf
from database.preprocessing import preprocess_blocks
from database.section_builder import build_sections
from database.chunker import chunk_sections
from database.heading_builder import (
    collect_headings,
    build_heading_embedding_text,
    build_subheading_embedding_text,
)
from database.alias_generator import (
    generate_heading_aliases_batch,
    generate_subheading_aliases_batch,
)
from database.heading_prompt import (
    build_heading_description,
    build_heading_examples,
)
from database.embedding import create_embeddings_batch
from database.hashing import compute_hash
from database.index_builder import build_knowledge_index
from database.chroma_db import (
    client,
    create_staging_generation,
    get_chunk_collection,
    get_heading_collection,
    get_active_generation,
    heading_alias_id,
    add_chunks,
    add_heading_item,
    validate_generation,
    activate_generation,
    delete_generation,
    discard_staging_generation,
    database_info,
)

# ============================================================
# CONFIG
# ============================================================

CHUNK_BATCH_SIZE = 200
EMBEDDING_BATCH_SIZE = 20
COPY_BATCH_SIZE = 500
DEFAULT_PDF_NAME = "Dataset_SOP_Kanim.pdf"

ALIAS_CACHE_PATH = os.path.join(CHROMA_DB_PATH, "alias_cache.json")
REPORT_DIR = os.path.join(CHROMA_DB_PATH, "ingestion_reports")

VERBOSE = "--ringkas" not in sys.argv
FORCE = "--force" in sys.argv

REPORT = []


# ============================================================
# HELPERS
# ============================================================

def log(message=""):
    print(message)
    REPORT.append(str(message))


def step(number, message):
    log(f"\n{'=' * 65}\n[{number}/8] {message}\n{'=' * 65}")


def save_report(generation):
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            REPORT_DIR,
            f"ingestion_{timestamp}_{generation or 'failed'}.txt"
        )
        with open(path, "w", encoding="utf-8") as file:
            file.write("\n".join(REPORT) + "\n")
        print(f"Laporan ingestion: {path}")
    except OSError as error:
        print(f"Gagal menyimpan laporan: {error}")


def normalize(value):
    return " ".join(str(value or "").casefold().split())


def clean_aliases(aliases):
    if not isinstance(aliases, list):
        return []

    result, seen = [], set()

    for alias in aliases:
        if not isinstance(alias, str):
            continue

        alias = " ".join(alias.split())

        if alias and alias.casefold() not in seen:
            seen.add(alias.casefold())
            result.append(alias)

    return result


def as_list(vector):
    return [float(value) for value in vector]


def batches(collection, include, where=None):
    """Membaca collection per batch sampai habis."""

    kwargs = {"where": where} if where else {}
    offset = 0

    while True:
        data = collection.get(
            limit=COPY_BATCH_SIZE,
            offset=offset,
            include=include,
            **kwargs
        )

        if not data["ids"]:
            return

        yield data
        offset += COPY_BATCH_SIZE


def make_source_identity(pdf_path):
    """source_id unik per PDF berdasarkan path relatif."""

    absolute_path = os.path.abspath(pdf_path)

    try:
        relative_path = os.path.relpath(
            absolute_path,
            os.path.abspath(PDF_FOLDER)
        )
    except ValueError:
        relative_path = os.path.basename(absolute_path)

    normalized = relative_path.replace("\\", "/").casefold()

    return "src_" + compute_hash(normalized)[:16], normalized


def validate_vectors(vectors, expected, label):
    if vectors is None:
        raise RuntimeError(f"Embedding {label} gagal dibuat.")

    if len(vectors) != expected:
        raise RuntimeError(
            f"Jumlah embedding {label} tidak sesuai. "
            f"Expected={expected}, actual={len(vectors)}"
        )

    dimension = None

    for index, vector in enumerate(vectors):
        if vector is None or len(vector) == 0:
            raise RuntimeError(
                f"Embedding {label} kosong pada index {index}."
            )

        if not all(math.isfinite(float(v)) for v in vector):
            raise RuntimeError(
                f"Embedding {label} mengandung nilai tidak valid."
            )

        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise RuntimeError(
                f"Dimensi embedding {label} tidak konsisten."
            )


# ============================================================
# SNAPSHOT & COMPARE (ID + HASH)
# ============================================================

def snapshot(collection, label, source_id=None):
    """Snapshot database; jika source_id diberikan, hanya PDF tersebut."""

    where = {"source_id": source_id} if source_id else None
    rows = {}

    try:
        for data in batches(collection, ["documents", "metadatas"], where):
            for rid, doc, meta in zip(
                data["ids"], data["documents"], data["metadatas"]
            ):
                meta = meta or {}
                rows[rid] = {
                    "hash": meta.get("content_hash"),
                    "meta": meta,
                    "doc": doc or ""
                }
    except Exception as error:
        raise RuntimeError(
            f"Gagal membaca snapshot {label}: {error}"
        ) from error

    return rows


def compare(items, old):
    """
    BARU    : ID tidak ada di database.
    BERUBAH : ID ada, hash berbeda.
    SAMA    : ID ada, hash sama.
    removed : ID lama yang tidak ada di data terbaru.
    """

    status = {}

    for item in items:
        previous = old.get(item["id"])

        if previous is None:
            status[item["id"]] = "BARU"
        elif previous["hash"] != item["hash"]:
            status[item["id"]] = "BERUBAH"
        else:
            status[item["id"]] = "SAMA"

    removed = sorted(set(old) - {item["id"] for item in items})

    return status, removed


def count_status(status):
    result = {"BARU": 0, "BERUBAH": 0, "SAMA": 0}

    for value in status.values():
        result[value] += 1

    return result


def print_comparison(label, status, removed):
    counts = count_status(status)

    log(f"\nPerbandingan {label}:")
    log(f"  BARU    : {counts['BARU']}")
    log(f"  BERUBAH : {counts['BERUBAH']}")
    log(f"  SAMA    : {counts['SAMA']}")
    log(f"  HAPUS   : {len(removed)}")


# ============================================================
# COPY ACTIVE -> STAGING
# ============================================================

def copy_collection(source, target, label):
    """Salin semua record (ID, dokumen, metadata, embedding) ke staging."""

    total = source.count()
    copied = 0

    for data in batches(
        source, ["documents", "metadatas", "embeddings"]
    ):
        if data["embeddings"] is None:
            raise RuntimeError(
                f"Embedding lama {label} tidak ditemukan."
            )

        target.upsert(
            ids=data["ids"],
            documents=data["documents"],
            metadatas=data["metadatas"],
            embeddings=[as_list(v) for v in data["embeddings"]]
        )

        copied += len(data["ids"])
        log(f"  Menyalin {label}: {copied}/{total}")

    if copied != total:
        raise RuntimeError(f"Penyalinan {label} tidak lengkap.")

    return copied


# ============================================================
# EMBEDDING REUSE
# ============================================================

def old_vectors(collection, wanted):
    """Ambil embedding lama hanya jika ID sama dan hash sama."""

    found = {}
    ids = list(wanted)

    for start in range(0, len(ids), COPY_BATCH_SIZE):
        data = collection.get(
            ids=ids[start:start + COPY_BATCH_SIZE],
            include=["metadatas", "embeddings"]
        )

        if data["embeddings"] is None:
            continue

        for rid, meta, vector in zip(
            data["ids"], data["metadatas"], data["embeddings"]
        ):
            if (
                (meta or {}).get("content_hash") == wanted[rid]
                and vector is not None
                and len(vector)
            ):
                found[rid] = as_list(vector)

    return found


def embed_items(items, active, label):
    """Pakai ulang embedding yang tidak berubah; embed hanya yang perlu."""

    reuse = {}

    if not FORCE:
        reuse = old_vectors(
            active,
            {item["id"]: item["hash"] for item in items}
        )

    vectors = [reuse.get(item["id"]) for item in items]
    todo = [i for i, vector in enumerate(vectors) if vector is None]

    log(
        f"\n{label}:"
        f"\n  Total item       : {len(items)}"
        f"\n  Vektor digunakan : {len(items) - len(todo)}"
        f"\n  Embedding baru   : {len(todo)}"
    )

    for start in range(0, len(todo), CHUNK_BATCH_SIZE):
        positions = todo[start:start + CHUNK_BATCH_SIZE]

        generated = create_embeddings_batch(
            [items[i]["text"] for i in positions],
            batch_size=EMBEDDING_BATCH_SIZE
        )

        validate_vectors(generated, len(positions), label)

        for index, vector in zip(positions, generated):
            vectors[index] = as_list(vector)

        log(
            f"  Progress embedding: "
            f"{min(start + len(positions), len(todo))}/{len(todo)}"
        )

    return vectors


# ============================================================
# ALIASES
# ============================================================

def load_alias_cache():
    try:
        with open(ALIAS_CACHE_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return {}


def save_alias_cache(cache):
    os.makedirs(os.path.dirname(ALIAS_CACHE_PATH), exist_ok=True)

    temp_path = ALIAS_CACHE_PATH + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)

    os.replace(temp_path, ALIAS_CACHE_PATH)


def read_existing_aliases():
    data = get_heading_collection().get(
        where={"type": "alias"},
        include=["metadatas"]
    )

    found = {}

    for metadata in data["metadatas"]:
        metadata = metadata or {}

        alias = metadata.get("alias")
        scope = metadata.get("alias_scope")
        heading_id = metadata.get("heading_id")

        if not alias or not heading_id:
            continue

        slot = found.setdefault(
            heading_id,
            {"heading": [], "by_id": {}, "by_name": {}}
        )

        if scope == "heading":
            slot["heading"].append(alias)

        elif scope == "subheading":
            sid = str(metadata.get("subheading_id") or "")
            slot["by_id"].setdefault(sid, []).append(alias)
            slot["by_name"].setdefault(
                normalize(metadata.get("subheading")), []
            ).append(alias)

    return found


def prepare_aliases(headings):
    existing = read_existing_aliases()
    cache = load_alias_cache()

    heading_aliases, subheading_aliases = {}, {}
    pending_headings, pending_subheadings = [], []

    for item in headings:
        hid = item["heading_id"]
        heading = item["heading"]
        slot = existing.get(hid, {})

        aliases = clean_aliases(slot.get("heading")) or clean_aliases(
            cache.get(f"{hid}::{normalize(heading)}")
        )

        heading_aliases[hid] = aliases

        if not aliases:
            pending_headings.append(item)

        for sub in item["subheadings"].values():
            sid = str(sub["subheading_id"])
            name = sub["subheading"]
            key = f"{hid}::{sid}"

            aliases = clean_aliases(
                slot.get("by_id", {}).get(sid)
                or slot.get("by_name", {}).get(normalize(name))
            ) or clean_aliases(cache.get(f"{key}::{normalize(name)}"))

            subheading_aliases[key] = aliases

            if not aliases:
                pending_subheadings.append({
                    "heading_id": hid,
                    "heading": heading,
                    "subheading_id": sub["subheading_id"],
                    "subheading": name
                })

    log(f"Alias heading yang perlu dibuat: {len(pending_headings)}")
    log(f"Alias subheading yang perlu dibuat: {len(pending_subheadings)}")

    if pending_headings:
        generated = generate_heading_aliases_batch(
            [item["heading"] for item in pending_headings]
        ) or {}

        for item in pending_headings:
            heading, hid = item["heading"], item["heading_id"]

            if heading not in generated:
                raise RuntimeError(
                    f"Gagal membuat alias heading: {heading}"
                )

            aliases = clean_aliases(generated[heading])
            heading_aliases[hid] = aliases
            cache[f"{hid}::{normalize(heading)}"] = aliases

    if pending_subheadings:
        generated = generate_subheading_aliases_batch(
            pending_subheadings
        ) or {}

        for item in pending_subheadings:
            hid, sid = item["heading_id"], item["subheading_id"]
            name = item["subheading"]
            key = f"{hid}::{sid}"

            if key not in generated:
                raise RuntimeError(
                    f"Gagal membuat alias subheading: {name}"
                )

            aliases = clean_aliases(generated[key])
            subheading_aliases[key] = aliases
            cache[f"{key}::{normalize(name)}"] = aliases

    if pending_headings or pending_subheadings:
        save_alias_cache(cache)

    return heading_aliases, subheading_aliases


# ============================================================
# CHUNKS
# ============================================================

def prepare_chunks(chunks, source_id, source_file):
    """Beri identitas dokumen dan ID chunk unik antar-PDF."""

    for chunk in chunks:
        chunk["source_id"] = source_id
        chunk["source_file"] = source_file
        chunk["content_chunk_id"] = (
            f"{source_id}::{chunk['content_chunk_id']}"
        )

        if not chunk.get("retrieval_text"):
            raise ValueError("Retrieval text chunk kosong.")

    return chunks


def process_chunks(chunks, collection, active):
    items = [
        {
            "id": chunk["content_chunk_id"],
            "text": chunk["retrieval_text"],
            "hash": chunk["content_hash"]
        }
        for chunk in chunks
    ]

    vectors = embed_items(items, active, "Chunk")

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    if add_chunks(chunks, target_collection=collection) != len(chunks):
        raise RuntimeError("Jumlah chunk yang disimpan tidak sesuai.")


def delete_stale_chunks(collection, source_id, current_ids):
    """Hapus hanya chunk milik source_id yang sedang diproses."""

    old_ids = set(
        collection.get(where={"source_id": source_id}, include=[])["ids"]
    )
    stale_ids = old_ids - set(current_ids)

    if stale_ids:
        collection.delete(ids=list(stale_ids))

    log(f"Chunk lama yang dihapus: {len(stale_ids)}")

    for record_id in sorted(stale_ids):
        log(f"  HAPUS: {record_id}")

    return len(stale_ids)


# ============================================================
# HEADINGS
# ============================================================

def subheading_alias_id(heading_id, subheading_id, alias):
    alias_hash = compute_hash(alias.strip().casefold())[:16]

    return (
        f"{heading_id}::subheading::"
        f"{subheading_id}::alias::{alias_hash}"
    )


def build_heading_jobs(headings, heading_aliases, subheading_aliases):
    jobs = []

    def add(kind, record_id, heading_id, text, name, **extra):
        jobs.append({
            "type": kind,
            "id": record_id,
            "heading_id": heading_id,
            "text": text,
            "hash": compute_hash(text),
            "name": name,
            "extra": extra
        })

    for item in headings:
        hid = item["heading_id"]
        heading = item["heading"]
        aliases = heading_aliases.get(hid, [])
        subs = list(item["subheadings"].values())

        add(
            "heading", hid, hid,
            build_heading_embedding_text(
                heading=heading,
                aliases=aliases,
                subheadings=[sub["subheading"] for sub in subs]
            ),
            heading,
            subheading_count=len(subs),
            alias_count=len(aliases)
        )

        for sub in subs:
            sid = sub["subheading_id"]
            name = sub["subheading"]
            description = build_heading_description(name, heading)
            examples = build_heading_examples(name, heading)

            add(
                "subheading", sid, hid,
                build_subheading_embedding_text(
                    heading=heading,
                    subheading=name,
                    description=description,
                    examples=examples
                ),
                name,
                subheading_id=sid,
                heading=heading,
                description=description,
                examples=" | ".join(examples)
            )

            for alias in subheading_aliases.get(f"{hid}::{sid}", []):
                add(
                    "alias",
                    subheading_alias_id(hid, sid, alias),
                    hid,
                    (
                        f"Layanan:\n{heading}\n\n"
                        f"Subheading:\n{name}\n\n"
                        f"Alias subheading:\n{alias}"
                    ),
                    alias,
                    alias_scope="subheading",
                    heading=heading,
                    subheading_id=sid,
                    subheading=name
                )

        for alias in aliases:
            add(
                "alias",
                heading_alias_id(hid, alias),
                hid,
                f"Layanan:\n{heading}\n\nNama lain:\n{alias}",
                alias,
                alias_scope="heading",
                heading=heading
            )

    return jobs


def process_headings(jobs, collection, active):
    vectors = embed_items(jobs, active, "Heading")

    for job, vector in zip(jobs, vectors):
        add_heading_item(
            item_type=job["type"],
            item_id=job["id"],
            heading_id=job["heading_id"],
            text=job["text"],
            embedding=vector,
            content_hash=job["hash"],
            target_collection=collection,
            name=job["name"],
            **job["extra"]
        )

    return len(jobs)


def delete_unused_heading_records(heading_collection, chunk_collection):
    """
    Heading/subheading/alias bersifat global, jadi hanya dihapus jika
    tidak ada chunk (dari PDF mana pun) yang masih memakainya.
    """

    used_headings, used_subheadings = set(), set()

    for data in batches(chunk_collection, ["metadatas"]):
        for metadata in data["metadatas"]:
            metadata = metadata or {}

            if metadata.get("heading_id"):
                used_headings.add(metadata["heading_id"])

            if metadata.get("subheading_id"):
                used_subheadings.add(metadata["subheading_id"])

    stale_ids = []

    for data in batches(heading_collection, ["metadatas"]):
        for record_id, metadata in zip(data["ids"], data["metadatas"]):
            metadata = metadata or {}

            kind = metadata.get("type")
            scope = metadata.get("alias_scope")
            heading_unused = metadata.get("heading_id") not in used_headings
            sub_unused = (
                metadata.get("subheading_id") not in used_subheadings
            )

            if (
                (kind == "heading" and heading_unused)
                or (kind == "subheading" and sub_unused)
                or (kind == "alias" and scope == "heading" and heading_unused)
                or (kind == "alias" and scope == "subheading" and sub_unused)
            ):
                stale_ids.append(record_id)

    if stale_ids:
        heading_collection.delete(ids=stale_ids)

    log(
        f"Heading/subheading/alias yang tidak digunakan lagi: "
        f"{len(stale_ids)}"
    )

    return len(stale_ids)


# ============================================================
# KNOWLEDGE INDEX (DARI SELURUH CHUNK DI STAGING)
# ============================================================

def rebuild_knowledge_index(collection, heading_collection):
    sections, chunks = [], []
    seen_sections = set()

    for data in batches(collection, ["documents", "metadatas"]):
        for record_id, document, metadata in zip(
            data["ids"], data["documents"], data["metadatas"]
        ):
            metadata = metadata or {}

            if metadata.get("type") != "chunk":
                continue

            heading_id = metadata.get("heading_id")
            subheading_id = metadata.get("subheading_id")

            if not heading_id or not subheading_id:
                continue

            document = document or ""

            common = {
                key: metadata.get(key)
                for key in (
                    "source_id", "source_file", "document", "category",
                    "section_id", "heading", "subheading",
                    "page_start", "page_end"
                )
            }
            common["heading_id"] = heading_id
            common["subheading_id"] = subheading_id
            common["subheading_occurrence"] = metadata.get(
                "subheading_occurrence", 1
            )

            section_key = (
                metadata.get("source_id"),
                metadata.get("section_id"),
                heading_id,
                subheading_id
            )

            if section_key not in seen_sections:
                seen_sections.add(section_key)
                sections.append({**common, "content": document})

            chunks.append({
                **common,
                "content_chunk_id": record_id,
                "content_hash": metadata.get("content_hash"),
                "chunk_index": metadata.get("chunk_index", 0),
                "total_chunks": metadata.get("total_chunks", 0),
                "retrieval_text": document
            })

    if not sections:
        raise RuntimeError(
            "Tidak ada section untuk membangun knowledge index."
        )

    with redirect_stdout(StringIO()):
        build_knowledge_index(sections, chunks, heading_collection)

    log(
        f"Knowledge index diperbarui dari "
        f"{len(chunks)} chunk seluruh dokumen."
    )


# ============================================================
# VALIDATION & CLEANUP
# ============================================================

def validate_collection(collection, expected_count, label):
    actual_count = collection.count()

    if actual_count != expected_count:
        raise RuntimeError(
            f"Jumlah {label} tidak sesuai. "
            f"Expected={expected_count}, actual={actual_count}"
        )

    dimension = None

    for data in batches(
        collection, ["documents", "metadatas", "embeddings"]
    ):
        if data["embeddings"] is None:
            raise RuntimeError(f"Embedding {label} tidak ditemukan.")

        if not (
            len(data["ids"])
            == len(data["documents"])
            == len(data["metadatas"])
            == len(data["embeddings"])
        ):
            raise RuntimeError(f"Data {label} tidak lengkap.")

        for record_id, document, metadata, vector in zip(
            data["ids"], data["documents"],
            data["metadatas"], data["embeddings"]
        ):
            metadata = metadata or {}

            if not document or not metadata or vector is None:
                raise RuntimeError(
                    f"Record {label} tidak lengkap: {record_id}"
                )

            if not metadata.get("content_hash"):
                raise RuntimeError(f"Hash tidak ditemukan: {record_id}")

            if not metadata.get("heading_id"):
                raise RuntimeError(
                    f"Heading ID tidak ditemukan: {record_id}"
                )

            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise RuntimeError(
                    f"Dimensi embedding berbeda: {record_id}"
                )

    return dimension


def validate_all(generation, chunk_collection, heading_collection):
    validate_generation(generation)

    chunk_count = chunk_collection.count()
    heading_count = heading_collection.count()

    chunk_dim = validate_collection(chunk_collection, chunk_count, "chunk")
    heading_dim = validate_collection(
        heading_collection, heading_count, "heading"
    )

    if (
        chunk_dim is not None
        and heading_dim is not None
        and chunk_dim != heading_dim
    ):
        raise RuntimeError("Dimensi embedding chunk dan heading berbeda.")

    return chunk_count, heading_count


def cleanup_old_generations(keep):
    names = {
        c if isinstance(c, str) else c.name
        for c in client.list_collections()
    }

    generations = {
        name.rpartition("__g_")[2] for name in names if "__g_" in name
    }

    for generation in sorted(generations - set(keep)):
        try:
            delete_generation(generation)
            log(f"Generation lama dihapus: {generation}")
        except Exception as error:
            log(f"Gagal menghapus generation {generation}: {error}")


# ============================================================
# MAIN
# ============================================================

def main():
    started = time.perf_counter()
    generation = None
    activated = False

    try:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        pdf_path = (
            args[0] if args
            else os.path.join(PDF_FOLDER, DEFAULT_PDF_NAME)
        )

        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF tidak ditemukan: {pdf_path}")

        source_id, source_file = make_source_identity(pdf_path)
        previous = get_active_generation()
        active_chunks = get_chunk_collection()
        active_headings = get_heading_collection()

        log("=" * 65)
        log("KNOWLEDGE BASE INGESTION")
        log("=" * 65)
        log(f"PDF          : {pdf_path}")
        log(f"Source ID    : {source_id}")
        log(f"Source file  : {source_file}")
        log(f"Force        : {FORCE}")
        log(f"Generation   : {previous}")
        log(f"Database     : {database_info()}")

        # 1. Parsing
        step(1, "Parsing dan preprocessing PDF")

        parsed = parse_pdf(pdf_path)

        if not parsed:
            raise RuntimeError(
                "Tidak ada teks yang berhasil diekstrak dari PDF."
            )

        log(f"Blok hasil parsing: {len(parsed)}")

        parsed = preprocess_blocks(parsed)

        if not parsed:
            raise RuntimeError("Tidak ada teks setelah preprocessing.")

        log(f"Blok setelah preprocessing: {len(parsed)}")

        # 2. Section, chunk, heading
        step(2, "Membentuk section, chunk, heading, dan subheading")

        sections = build_sections(parsed)

        if not sections:
            raise RuntimeError("Section tidak berhasil dibentuk.")

        for section in sections:
            section["source_id"] = source_id
            section["source_file"] = source_file

        chunks = prepare_chunks(
            chunk_sections(sections), source_id, source_file
        )
        headings = collect_headings(sections)

        if not chunks:
            raise RuntimeError("Chunk tidak berhasil dibentuk.")

        if not headings:
            raise RuntimeError("Heading tidak berhasil dikumpulkan.")

        log(f"Section   : {len(sections)}")
        log(f"Chunk     : {len(chunks)}")
        log(f"Heading   : {len(headings)}")

        if VERBOSE:
            for section in sections:
                log(
                    f"  {section.get('heading')} > "
                    f"{section.get('subheading')}"
                )

        # 3. Alias
        step(3, "Menyiapkan alias heading dan subheading")

        heading_aliases, subheading_aliases = prepare_aliases(headings)
        jobs = build_heading_jobs(
            headings, heading_aliases, subheading_aliases
        )

        log(f"Record heading yang diproses: {len(jobs)}")

        # 4. Compare
        step(4, "Membandingkan ID dan hash")

        # Chunk dibandingkan hanya dengan PDF yang sama;
        # heading ID bersifat global.
        old_chunks = snapshot(active_chunks, "chunk", source_id=source_id)
        old_heads = snapshot(active_headings, "heading")

        chunk_status, chunk_removed = compare(
            [
                {
                    "id": c["content_chunk_id"],
                    "hash": c["content_hash"],
                    "text": c["retrieval_text"]
                }
                for c in chunks
            ],
            old_chunks
        )
        head_status, _ = compare(jobs, old_heads)

        print_comparison("CHUNK", chunk_status, chunk_removed)
        print_comparison("HEADING", head_status, [])

        for record_id in chunk_removed:
            old_hash = old_chunks[record_id]["hash"]
            log(
                f"  Chunk lama dihapus: {record_id} "
                f"| hash={str(old_hash)[:12] if old_hash else '-'}"
            )

        # 5. Staging + salin semua data lama
        step(5, "Membuat staging dan mempertahankan seluruh data lama")

        staging = create_staging_generation()
        generation = staging["generation"]
        staging_chunks = staging["chunk_collection"]
        staging_headings = staging["heading_collection"]

        log(f"Generation staging: {generation}")

        copied_chunks = copy_collection(
            active_chunks, staging_chunks, "chunk"
        )
        copied_headings = copy_collection(
            active_headings, staging_headings, "heading"
        )

        log(f"Total chunk awal staging: {copied_chunks}")
        log(f"Total heading awal staging: {copied_headings}")

        # 6. Chunk
        step(6, "Menambahkan, memperbarui, dan menghapus chunk")

        process_chunks(chunks, staging_chunks, active_chunks)

        deleted_chunks = delete_stale_chunks(
            staging_chunks,
            source_id,
            [c["content_chunk_id"] for c in chunks]
        )

        chunk_counts = count_status(chunk_status)

        log(
            f"Chunk baru/berubah disimpan: "
            f"{chunk_counts['BARU'] + chunk_counts['BERUBAH']}"
        )
        log(f"Chunk lama dihapus: {deleted_chunks}")

        # 7. Heading
        step(7, "Memperbarui heading, subheading, dan alias")

        process_headings(jobs, staging_headings, active_headings)
        delete_unused_heading_records(staging_headings, staging_chunks)

        # 8. Validasi + aktivasi
        step(8, "Validasi dan aktivasi database")

        chunk_total, heading_total = validate_all(
            generation, staging_chunks, staging_headings
        )

        log(
            f"Validasi berhasil."
            f"\nTotal chunk: {chunk_total}"
            f"\nTotal heading: {heading_total}"
        )

        log("\nMembangun knowledge_index.json dari seluruh dokumen...")
        rebuild_knowledge_index(staging_chunks, staging_headings)

        activate_generation(generation)
        activated = True

        log(f"\nDatabase berhasil diaktifkan:\n{previous} -> {generation}")

        # Simpan satu generation sebelumnya untuk rollback.
        cleanup_old_generations(keep={generation, previous})

        log("\n" + "=" * 65)
        log("INGESTION BERHASIL")
        log("=" * 65)
        log(f"PDF             : {source_file}")
        log(f"Generation      : {generation}")
        log(f"Chunk BARU      : {chunk_counts['BARU']}")
        log(f"Chunk BERUBAH   : {chunk_counts['BERUBAH']}")
        log(f"Chunk SAMA      : {chunk_counts['SAMA']}")
        log(f"Chunk DIHAPUS   : {deleted_chunks}")
        log(f"Total chunk DB  : {chunk_total}")
        log(f"Total heading DB: {heading_total}")
        log(f"Waktu proses    : {time.perf_counter() - started:.2f} detik")
        log(f"Database        : {database_info()}")

        save_report(generation)

    except (Exception, KeyboardInterrupt) as error:
        log(
            "\nINGESTION DIBATALKAN."
            if isinstance(error, KeyboardInterrupt)
            else f"\nINGESTION GAGAL: {error}"
        )

        if generation and not activated:
            try:
                discard_staging_generation(generation)
                log("Staging yang gagal berhasil dibersihkan.")
            except Exception as cleanup_error:
                log(f"Gagal membersihkan staging: {cleanup_error}")

        log("Database aktif sebelumnya tidak diganti.")
        save_report(generation)

        raise


if __name__ == "__main__":
    main()