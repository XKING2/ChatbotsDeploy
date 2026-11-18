import json
import os
import re
import uuid

import chromadb

from config import CHROMA_DB_PATH, CHUNK_COLLECTION_NAME, HEADING_COLLECTION_NAME
from database.hashing import compute_hash

os.makedirs(CHROMA_DB_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

MANIFEST_PATH = os.path.join(CHROMA_DB_PATH, "active_generation.json")
COLLECTION_METADATA = {"hnsw:space": "cosine"}
BATCH = 500
BASES = {"chunk": CHUNK_COLLECTION_NAME, "heading": HEADING_COLLECTION_NAME}

_collections = {}
_manifest_cache = (None, None)  # (mtime_ns, manifest)


# ============================================================
# HELPER
# ============================================================

def _batches(items, size=BATCH):
    if size <= 0:
        raise ValueError("Ukuran batch harus lebih besar dari 0.")
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _clean(metadata):
    """Chroma menolak metadata bernilai None."""
    return {k: v for k, v in metadata.items() if v is not None}


def _collection(name):
    if name not in _collections:
        _collections[name] = client.get_or_create_collection(
            name=name, metadata=COLLECTION_METADATA
        )
    return _collections[name]


def _existing_names():
    return {c if isinstance(c, str) else c.name
            for c in client.list_collections()}


def _where(*conditions):
    return {"$and": list(conditions)}


def _delete_ids(collection, ids, batch_size=BATCH):
    for batch in _batches(list(ids), batch_size):
        collection.delete(ids=batch)


# ============================================================
# ACTIVE COLLECTION / MANIFEST
# ============================================================

def _load_manifest():
    """Manifest hanya dibaca ulang bila file berubah."""
    global _manifest_cache

    if not os.path.exists(MANIFEST_PATH):
        return {"generation": "legacy",
                "chunk_collection": CHUNK_COLLECTION_NAME,
                "heading_collection": HEADING_COLLECTION_NAME}

    mtime = os.stat(MANIFEST_PATH).st_mtime_ns
    if _manifest_cache[0] == mtime:
        return _manifest_cache[1]

    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not all(manifest.get(k) for k in
                   ("generation", "chunk_collection", "heading_collection")):
            raise ValueError("Isi manifest tidak lengkap.")
    except Exception as error:
        raise RuntimeError(
            f"Gagal membaca manifest database aktif: {error}"
        ) from error

    _manifest_cache = (mtime, manifest)
    return manifest


def _write_manifest(manifest):
    """Menulis manifest secara atomik."""
    global _manifest_cache

    tmp = f"{MANIFEST_PATH}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, MANIFEST_PATH)
        _manifest_cache = (os.stat(MANIFEST_PATH).st_mtime_ns, manifest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def get_active_generation():
    return _load_manifest()["generation"]


def get_active_collection_names():
    m = _load_manifest()
    return {"chunk": m["chunk_collection"], "heading": m["heading_collection"]}


def get_chunk_collection():
    return _collection(get_active_collection_names()["chunk"])


def get_heading_collection():
    return _collection(get_active_collection_names()["heading"])


class ActiveCollectionProxy:
    """Meneruskan method ke collection yang sedang aktif."""

    def __init__(self, kind):
        self.kind = kind

    def __getattr__(self, attribute):
        target = (get_chunk_collection() if self.kind == "chunk"
                  else get_heading_collection())
        return getattr(target, attribute)


chunk_collection = ActiveCollectionProxy("chunk")
heading_collection = ActiveCollectionProxy("heading")


def _resolve(target, kind):
    if target is not None:
        return target
    return get_chunk_collection() if kind == "chunk" else get_heading_collection()


# ============================================================
# GENERATION
# ============================================================

def _generation_names(generation):
    generation = str(generation)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", generation):
        raise ValueError("ID generation mengandung karakter tidak valid.")

    suffix = f"__g_{generation}"
    room = 63 - len(suffix)
    if room < 1:
        raise ValueError("ID generation terlalu panjang.")

    return generation, {kind: f"{str(base)[:room]}{suffix}"
                        for kind, base in BASES.items()}


def create_staging_generation(generation=None):
    generation, names = _generation_names(generation or uuid.uuid4().hex[:12])
    if set(names.values()) & _existing_names():
        raise ValueError(f"Generation {generation} sudah pernah dibuat.")

    created = []
    try:
        for kind in ("chunk", "heading"):
            client.create_collection(name=names[kind],
                                     metadata=COLLECTION_METADATA)
            created.append(names[kind])
    except Exception:
        for name in created:
            try:
                client.delete_collection(name)
            except Exception:
                pass
        raise

    return {
        "generation": generation,
        "chunk_collection_name": names["chunk"],
        "heading_collection_name": names["heading"],
        "chunk_collection": _collection(names["chunk"]),
        "heading_collection": _collection(names["heading"]),
    }


def get_staging_collection(generation, collection_type):
    if collection_type not in BASES:
        raise ValueError("collection_type harus 'chunk' atau 'heading'.")

    name = _generation_names(generation)[1][collection_type]
    if name not in _existing_names():
        raise ValueError(f"Collection staging tidak ditemukan: {name}")
    return _collection(name)


def validate_generation(generation):
    """Memastikan kedua collection staging tersedia."""
    generation, names = _generation_names(generation)

    missing = [n for n in names.values() if n not in _existing_names()]
    if missing:
        raise RuntimeError(f"Collection staging tidak lengkap: {missing}")

    return {
        "valid": True,
        "generation": generation,
        "chunk_collection": names["chunk"],
        "heading_collection": names["heading"],
        "chunk_count": _collection(names["chunk"]).count(),
        "heading_count": _collection(names["heading"]).count(),
    }


def activate_generation(generation):
    """Mengaktifkan generation melalui manifest."""
    v = validate_generation(generation)

    _write_manifest({"generation": v["generation"],
                     "chunk_collection": v["chunk_collection"],
                     "heading_collection": v["heading_collection"]})

    return {"success": True, "generation": v["generation"],
            "chunk_count": v["chunk_count"],
            "heading_count": v["heading_count"]}


def delete_generation(generation):
    """Menghapus generation yang tidak sedang aktif."""
    generation, names = _generation_names(generation)
    if generation == get_active_generation():
        raise ValueError("Generation aktif tidak boleh dihapus.")

    existing = _existing_names()
    deleted = []
    for name in names.values():
        if name in existing:
            client.delete_collection(name)
            _collections.pop(name, None)
            deleted.append(name)
    return deleted


discard_staging_generation = delete_generation


# ============================================================
# ID GENERATOR
# ============================================================

def heading_alias_id(heading_id, alias):
    return f"{heading_id}::alias::{compute_hash(alias.strip().casefold())[:12]}"


def subheading_record_id(heading_id, index):
    return f"{heading_id}::subheading::{index:03d}"


def chunk_record_id(heading_id, subheading_index, index):
    return (f"{subheading_record_id(heading_id, subheading_index)}"
            f"::chunk::{index:03d}")


# ============================================================
# CHUNK
# ============================================================

CHUNK_REQUIRED = (
    "category", "section_id", "heading_id", "heading", "subheading_id",
    "subheading", "subheading_occurrence", "content_chunk_id",
    "content_hash", "page_start", "page_end", "chunk_index", "total_chunks",
)
CHUNK_COPIED = CHUNK_REQUIRED + ("document",)


def _chunk_metadata(chunk):
    source_file = chunk.get("source_file", "")
    source_id = chunk.get("source_id")

    if not source_id:
        if not source_file:
            raise ValueError("Chunk wajib memiliki source_id atau source_file.")
        normalized = " ".join(str(source_file).casefold().split())
        source_id = "src_" + compute_hash(normalized)[:16]

    missing = [f for f in CHUNK_REQUIRED if chunk.get(f) is None]
    if missing:
        raise ValueError("Metadata chunk tidak lengkap: " + ", ".join(missing))

    return _clean({
        "type": "chunk",
        "source_id": source_id,
        "source_file": source_file,
        **{f: chunk.get(f, "" if f == "document" else None)
           for f in CHUNK_COPIED},
    })


def get_existing_chunk_hashes(source_id=None, target_collection=None):
    kwargs = {"where": {"source_id": source_id}} if source_id else {}
    data = _resolve(target_collection, "chunk").get(
        include=["metadatas"], **kwargs
    )
    return {i: (m or {}).get("content_hash")
            for i, m in zip(data["ids"], data["metadatas"])}


def add_chunks(chunks, target_collection=None):
    """Upsert chunk; ID yang sama diperbarui."""
    if not chunks:
        return 0

    for chunk in chunks:
        if not chunk.get("retrieval_text"):
            raise ValueError("Chunk memiliki retrieval_text kosong.")
        if not chunk.get("embedding"):
            raise ValueError("Chunk belum memiliki embedding.")

    collection = _resolve(target_collection, "chunk")
    rows = [(c["content_chunk_id"], c["retrieval_text"], c["embedding"],
             _chunk_metadata(c)) for c in chunks]

    for batch in _batches(rows):
        ids, docs, embeddings, metas = zip(*batch)
        collection.upsert(ids=list(ids), documents=list(docs),
                          embeddings=list(embeddings), metadatas=list(metas))
    return len(rows)


def update_chunk_metadata(chunks, batch_size=BATCH, target_collection=None):
    """Memperbarui metadata tanpa mengubah dokumen dan embedding."""
    collection = _resolve(target_collection, "chunk")
    chunks = list(chunks or [])

    for batch in _batches(chunks, batch_size):
        collection.update(ids=[c["content_chunk_id"] for c in batch],
                          metadatas=[_chunk_metadata(c) for c in batch])
    return len(chunks)


def adopt_legacy_chunks(source_id, document_name, batch_size=BATCH,
                        target_collection=None):
    """Memberikan source_id pada chunk yang belum memilikinya."""
    if not source_id or not document_name:
        raise ValueError("source_id dan document_name wajib diisi.")

    collection = _resolve(target_collection, "chunk")
    data = collection.get(where={"document": document_name},
                          include=["metadatas"])

    unassigned = [(i, {**m, "source_id": source_id})
                  for i, m in zip(data["ids"], data["metadatas"])
                  if not (m or {}).get("source_id")]

    for batch in _batches(unassigned, batch_size):
        collection.update(ids=[i for i, _ in batch],
                          metadatas=[m for _, m in batch])
    return len(unassigned)


def count_unassigned_chunk_records(target_collection=None):
    data = _resolve(target_collection, "chunk").get(include=["metadatas"])
    return sum(1 for m in data["metadatas"] if not (m or {}).get("source_id"))


def get_chunk_ids_by_source(source_id, target_collection=None):
    if not source_id:
        raise ValueError("source_id wajib diisi.")
    data = _resolve(target_collection, "chunk").get(
        where={"source_id": source_id}, include=[]
    )
    return list(data["ids"])


def delete_stale_chunks(source_id, current_chunk_ids, batch_size=BATCH,
                        target_collection=None):
    """Menghapus chunk milik sumber yang tidak ada di hasil terbaru."""
    collection = _resolve(target_collection, "chunk")
    stale = sorted(set(get_chunk_ids_by_source(source_id, collection))
                   - set(current_chunk_ids or []))
    _delete_ids(collection, stale, batch_size)
    return stale


def delete_chunks_by_source(source_id, batch_size=BATCH, target_collection=None):
    """Menghapus seluruh chunk milik satu sumber."""
    collection = _resolve(target_collection, "chunk")
    ids = get_chunk_ids_by_source(source_id, collection)
    _delete_ids(collection, ids, batch_size)
    return len(ids)


def get_chunks_by_heading(heading_id, target_collection=None):
    return _resolve(target_collection, "chunk").get(
        where={"heading_id": heading_id}, include=["documents", "metadatas"]
    )


def get_chunks_by_subheading(heading_id, subheading_id, target_collection=None):
    return _resolve(target_collection, "chunk").get(
        where=_where({"heading_id": heading_id},
                     {"subheading_id": subheading_id}),
        include=["documents", "metadatas"],
    )


# ============================================================
# HEADING / ALIAS / SUBHEADING
# ============================================================

NAME_FIELD = {"heading": "heading", "alias": "alias",
              "subheading": "subheading"}


def get_existing_heading_hashes(target_collection=None):
    """Hash seluruh record heading, alias, dan subheading."""
    data = _resolve(target_collection, "heading").get(include=["metadatas"])
    return {i: (m or {}).get("content_hash")
            for i, m in zip(data["ids"], data["metadatas"])}


def add_heading_item(item_type, item_id, heading_id, text, embedding,
                     content_hash, target_collection=None, **extra):
    """
    Menyimpan heading, alias, atau subheading (embedding dibuat pemanggil).
    `text` disimpan sebagai dokumen; nama pada metadata memakai `name`
    bila diberikan.
    """
    if item_type not in NAME_FIELD:
        raise ValueError(f"Tipe heading tidak valid: {item_type}")
    if not item_id or not heading_id:
        raise ValueError("item_id dan heading_id wajib diisi.")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Teks heading tidak boleh kosong.")
    if not embedding:
        raise ValueError("Embedding heading tidak boleh kosong.")

    name = extra.pop("name", None) or text
    metadata = {"type": item_type, "heading_id": heading_id,
                "content_hash": content_hash, **extra,
                NAME_FIELD[item_type]: name}

    if item_type == "alias":
        metadata["alias_id"] = item_id
    elif item_type == "subheading":
        metadata["subheading_id"] = extra.get("subheading_id") or item_id

    _resolve(target_collection, "heading").upsert(
        ids=[item_id], documents=[text], embeddings=[embedding],
        metadatas=[_clean(metadata)],
    )
    return 1


def delete_heading_items(heading_id, target_collection=None):
    """Menghapus heading beserta alias dan subheading miliknya."""
    if not heading_id:
        raise ValueError("heading_id wajib diisi.")

    collection = _resolve(target_collection, "heading")
    ids = collection.get(where={"heading_id": heading_id}, include=[])["ids"]
    _delete_ids(collection, ids)
    return len(ids)


def get_heading(heading_id, target_collection=None):
    return _resolve(target_collection, "heading").get(
        ids=[heading_id], include=["documents", "metadatas", "embeddings"]
    )


def get_heading_children(heading_id, target_collection=None):
    return _resolve(target_collection, "heading").get(
        where={"heading_id": heading_id},
        include=["documents", "metadatas", "embeddings"],
    )


def _heading_items(item_type, heading_id=None, target_collection=None):
    where = {"type": item_type}
    if heading_id:
        where = _where(where, {"heading_id": heading_id})
    return _resolve(target_collection, "heading").get(
        where=where, include=["documents", "metadatas"]
    )


def get_headings(target_collection=None):
    return _heading_items("heading", target_collection=target_collection)


def get_subheadings(heading_id, target_collection=None):
    return _heading_items("subheading", heading_id, target_collection)


def get_aliases(heading_id, target_collection=None):
    return _heading_items("alias", heading_id, target_collection)


# ============================================================
# DATABASE INFORMATION
# ============================================================

def database_stats():
    return {CHUNK_COLLECTION_NAME: get_chunk_collection().count(),
            HEADING_COLLECTION_NAME: get_heading_collection().count()}


def database_info():
    names = get_active_collection_names()
    return {
        "generation": get_active_generation(),
        "chunk_collection": names["chunk"],
        "heading_collection": names["heading"],
        "chunk_count": get_chunk_collection().count(),
        "heading_count": get_heading_collection().count(),
    }


def print_database_stats():
    print("\n" + "=" * 50 + "\nCHROMADB\n" + "=" * 50)
    for name, count in database_stats().items():
        print(f"{name}: {count}")
    print("=" * 50)


def reset_database(collection_name=None):
    """Reset collection aktif secara manual."""
    active = get_active_collection_names()
    by_base = {BASES[k]: active[k] for k in BASES}

    if collection_name is None:
        targets = list(by_base.values())
    elif collection_name in by_base:
        targets = [by_base[collection_name]]
    elif collection_name in by_base.values():
        targets = [collection_name]
    else:
        raise ValueError("Nama collection reset tidak dikenal.")

    for name in targets:
        if name in _existing_names():
            client.delete_collection(name)
        _collections.pop(name, None)
        _collection(name)

    return {"success": True, "reset_collections": targets}