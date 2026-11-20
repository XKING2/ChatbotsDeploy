import json
import os
import tempfile

from config import KNOWLEDGE_INDEX_PATH

from database.heading_builder import collect_headings
from database.chroma_db import get_aliases

INDEX_VERSION = 2


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(value):
    if not isinstance(value, str):
        return ""

    return " ".join(value.split())


def clean_aliases(aliases, original_name=None):
    """Membersihkan alias dan menghilangkan duplikat."""

    if not isinstance(aliases, list):
        return []

    cleaned = []
    seen = set()

    if original_name:
        seen.add(original_name.strip().casefold())

    for alias in aliases:
        if not isinstance(alias, str):
            continue

        alias = normalize_text(alias)

        if not alias:
            continue

        key = alias.casefold()

        if key in seen:
            continue

        cleaned.append(alias)
        seen.add(key)

    return sorted(cleaned, key=str.casefold)


# =========================================================
# CHROMADB ALIAS
# =========================================================

def get_heading_aliases(heading_id, heading_collection=None):
    """
    Mengambil alias dari ChromaDB.

    heading_collection: collection sumber alias (mis. staging).
    Jika None, memakai collection aktif.

    Return: (heading_aliases, {subheading_id: [alias]})
    """

    try:
        data = get_aliases(
            heading_id, target_collection=heading_collection
        ) or {}

    except Exception as exc:
        raise RuntimeError(
            f"Gagal mengambil alias heading {heading_id}: {exc}"
        ) from exc

    heading_aliases = []
    subheading_aliases = {}

    for metadata in data.get("metadatas") or []:
        if not isinstance(metadata, dict):
            continue

        alias = metadata.get("alias")

        if not isinstance(alias, str):
            continue

        alias = normalize_text(alias)

        if not alias:
            continue

        scope = metadata.get("alias_scope")

        if scope == "heading":
            heading_aliases.append(alias)

        elif scope == "subheading":
            subheading_id = str(
                metadata.get("subheading_id") or ""
            ).strip()

            if subheading_id:
                subheading_aliases.setdefault(
                    subheading_id, []
                ).append(alias)

    return heading_aliases, subheading_aliases


# =========================================================
# CHUNK GROUPING
# =========================================================

def index_chunks_by_subheading(chunks):
    """
    Mengelompokkan referensi chunk berdasarkan subheading_id.
    Tidak menyimpan isi chunk maupun embedding.
    """

    grouped = {}
    seen_ids = set()

    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue

        subheading_id = str(chunk.get("subheading_id") or "").strip()
        chunk_id = str(chunk.get("content_chunk_id") or "").strip()

        if not subheading_id or not chunk_id or chunk_id in seen_ids:
            continue

        seen_ids.add(chunk_id)

        grouped.setdefault(subheading_id, []).append({
            "content_chunk_id": chunk_id,
            "chunk_index": chunk.get("chunk_index", 0),
            "total_chunks": chunk.get("total_chunks", 0),
            "source_id": chunk.get("source_id"),
            "source_file": chunk.get("source_file"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
        })

    # Urut per PDF, lalu per chunk_index (subheading bisa dipakai >1 PDF).
    for items in grouped.values():
        items.sort(
            key=lambda item: (
                item.get("source_id") or "",
                item.get("chunk_index") or 0,
            )
        )

    return grouped


# =========================================================
# HEADING & SUBHEADING INDEX
# =========================================================

def build_subheadings(
    subheadings,
    chunks_by_subheading,
    subheading_aliases,
):
    results = []

    for subheading_id, data in subheadings.items():
        subheading = normalize_text(data.get("subheading"))

        if not subheading:
            continue

        results.append({
            "subheading_id": subheading_id,
            "subheading": subheading,
            "aliases": clean_aliases(
                subheading_aliases.get(subheading_id, []),
                original_name=subheading,
            ),
            "content_chunks": chunks_by_subheading.get(
                subheading_id, []
            ),
        })

    results.sort(key=lambda item: item["subheading"].casefold())

    return results


def build_heading_index(
    heading_data,
    chunks_by_subheading,
    heading_collection=None,
):
    heading_id = heading_data["heading_id"]
    heading = normalize_text(heading_data["heading"])

    heading_aliases, subheading_aliases = get_heading_aliases(
        heading_id, heading_collection
    )

    return {
        "heading_id": heading_id,
        "heading": heading,
        "aliases": clean_aliases(heading_aliases, original_name=heading),
        "subheadings": build_subheadings(
            subheadings=heading_data["subheadings"],
            chunks_by_subheading=chunks_by_subheading,
            subheading_aliases=subheading_aliases,
        ),
    }


# =========================================================
# JSON STORAGE
# =========================================================

def save_knowledge_index(knowledge):
    """Simpan JSON secara atomik (temp file + os.replace)."""

    path = os.path.abspath(os.fspath(KNOWLEDGE_INDEX_PATH))
    directory = os.path.dirname(path)

    os.makedirs(directory, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=".knowledge_index_",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = file.name

            json.dump(knowledge, file, ensure_ascii=False, indent=4)

            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)

    except Exception:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

        raise


# =========================================================
# MAIN BUILDER
# =========================================================

def build_knowledge_index(sections, chunks, heading_collection=None):
    """
    Membangun knowledge_index.json.

    sections           : hasil build_sections()
    chunks             : hasil chunk_sections()
    heading_collection : collection heading sumber alias
                         (mis. staging). None = collection aktif.
    """

    if not sections:
        raise ValueError("Sections kosong. Knowledge index tidak dibuat.")

    print("\n" + "=" * 65)
    print("MEMBANGUN KNOWLEDGE INDEX")
    print("=" * 65)

    chunks_by_subheading = index_chunks_by_subheading(chunks)
    heading_groups = collect_headings(sections)

    if not heading_groups:
        raise ValueError("Tidak ditemukan heading yang valid.")

    heading_index = []

    for heading_data in heading_groups:
        item = build_heading_index(
            heading_data=heading_data,
            chunks_by_subheading=chunks_by_subheading,
            heading_collection=heading_collection,
        )

        heading_index.append(item)

        print(
            f"HEADING: {item['heading']} "
            f"| subheadings={len(item['subheadings'])} "
            f"| aliases={len(item['aliases'])} "
            f"| subheading_aliases="
            f"{sum(len(s['aliases']) for s in item['subheadings'])}"
        )

    heading_index.sort(key=lambda item: item["heading"].casefold())

    subheadings = [
        sub for item in heading_index for sub in item["subheadings"]
    ]

    statistics = {
        "total_headings": len(heading_index),
        "total_subheadings": len(subheadings),
        "total_heading_aliases": sum(
            len(item["aliases"]) for item in heading_index
        ),
        "total_subheading_aliases": sum(
            len(sub["aliases"]) for sub in subheadings
        ),
        "total_chunks": sum(
            len(sub["content_chunks"]) for sub in subheadings
        ),
    }

    knowledge = {
        "version": INDEX_VERSION,
        "headings": heading_index,
        "statistics": statistics,
    }

    save_knowledge_index(knowledge)

    print("-" * 65)
    print(f"Jumlah heading          : {statistics['total_headings']}")
    print(f"Jumlah subheading       : {statistics['total_subheadings']}")
    print(f"Alias heading           : {statistics['total_heading_aliases']}")
    print(f"Alias subheading        : {statistics['total_subheading_aliases']}")
    print(f"Referensi chunks        : {statistics['total_chunks']}")
    print(f"Knowledge index         : {KNOWLEDGE_INDEX_PATH}")
    print("=" * 65)

    return knowledge