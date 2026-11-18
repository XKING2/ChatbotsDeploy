
import json
import os
import tempfile

from config import KNOWLEDGE_INDEX_PATH

from database.heading_builder import collect_headings
from database.chroma_db import get_aliases


# =========================================================
# CONFIGURATION
# =========================================================

INDEX_VERSION = 2


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(value):
    if not isinstance(value, str):
        return ""

    return " ".join(value.split())


def clean_aliases(aliases, original_name=None):
    """
    Membersihkan alias dan menghilangkan duplikat.
    """

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

def get_heading_aliases(heading_id):
    """
    Mengambil alias dari ChromaDB.

    Alias dipisahkan menjadi:
    - heading aliases
    - subheading aliases
    """

    try:
        data = get_aliases(heading_id) or {}

    except Exception as exc:
        raise RuntimeError(
            f"Gagal mengambil alias heading {heading_id}: {exc}"
        ) from exc

    metadatas = data.get("metadatas") or []

    heading_aliases = []
    subheading_aliases = {}

    for metadata in metadatas:

        if not isinstance(metadata, dict):
            continue

        alias = metadata.get("alias")

        if not isinstance(alias, str):
            continue

        alias = normalize_text(alias)

        if not alias:
            continue

        alias_scope = metadata.get("alias_scope")

        if alias_scope == "heading":
            heading_aliases.append(alias)

        elif alias_scope == "subheading":

            subheading_id = str(
                metadata.get("subheading_id") or ""
            ).strip()

            if not subheading_id:
                continue

            subheading_aliases.setdefault(
                subheading_id, []
            ).append(alias)

    return (
        heading_aliases,
        subheading_aliases,
    )


# =========================================================
# CHUNK GROUPING
# =========================================================

def index_chunks_by_subheading(chunks):
    """
    Mengelompokkan referensi chunk berdasarkan subheading_id.

    Tidak mengambil atau menyimpan isi chunk maupun embedding.
    """

    grouped = {}
    seen_ids = set()

    for chunk in chunks or []:

        if not isinstance(chunk, dict):
            continue

        subheading_id = str(
            chunk.get("subheading_id") or ""
        ).strip()

        chunk_id = str(
            chunk.get("content_chunk_id") or ""
        ).strip()

        if not subheading_id or not chunk_id:
            continue

        # Hindari referensi chunk ganda.
        if chunk_id in seen_ids:
            continue

        seen_ids.add(chunk_id)

        grouped.setdefault(
            subheading_id, []
        ).append({
            "content_chunk_id": chunk_id,
            "chunk_index": chunk.get("chunk_index", 0),
            "total_chunks": chunk.get("total_chunks", 0),
            "source_id": chunk.get("source_id"),
            "source_file": chunk.get("source_file"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
        })

    for items in grouped.values():
        items.sort(
            key=lambda item: item.get("chunk_index", 0)
        )

    return grouped


# =========================================================
# SECTION GROUPING
# =========================================================

def collect_section_details(sections):
    """
    Menghubungkan section dengan struktur heading
    dan subheading.
    """

    heading_groups = collect_headings(sections)

    section_lookup = {}

    for section in sections:

        heading_id = section.get("heading_id")
        subheading_id = section.get("subheading_id")

        if not heading_id or not subheading_id:
            continue

        key = (heading_id, subheading_id)

        section_lookup.setdefault(key, []).append(section)

    for heading_data in heading_groups:

        heading_id = heading_data["heading_id"]

        for subheading_id, subheading_data in (
            heading_data["subheadings"].items()
        ):

            subheading_data["_sections"] = section_lookup.get(
                (heading_id, subheading_id), []
            )

    return heading_groups


# =========================================================
# SUBHEADING INDEX
# =========================================================

def build_subheadings(
    heading_data,
    chunks_by_subheading,
    subheading_aliases,
):
    """
    Membangun metadata subheading dan referensi chunk.
    """

    results = []

    for subheading_id, data in (
        heading_data.items()
    ):

        subheading = normalize_text(
            data.get("subheading")
        )

        if not subheading:
            continue

        aliases = clean_aliases(
            subheading_aliases.get(subheading_id, []),
            original_name=subheading,
        )

        content_chunks = chunks_by_subheading.get(
            subheading_id, []
        )

        results.append({
            "subheading_id": subheading_id,
            "subheading": subheading,
            "aliases": aliases,
            "content_chunks": content_chunks,
        })

    results.sort(
        key=lambda item: item["subheading"].casefold()
    )

    return results


# =========================================================
# HEADING INDEX
# =========================================================

def build_heading_index(
    heading_data,
    chunks_by_subheading,
):
    """
    Membangun satu heading beserta subheading-nya.
    """

    heading_id = heading_data["heading_id"]

    heading = normalize_text(
        heading_data["heading"]
    )

    heading_aliases, subheading_aliases = (
        get_heading_aliases(heading_id)
    )

    heading_aliases = clean_aliases(
        heading_aliases,
        original_name=heading,
    )

    subheadings = build_subheadings(
        heading_data=heading_data["subheadings"],
        chunks_by_subheading=chunks_by_subheading,
        subheading_aliases=subheading_aliases,
    )

    return {
        "heading_id": heading_id,
        "heading": heading,
        "aliases": heading_aliases,
        "subheadings": subheadings,
    }


# =========================================================
# JSON STORAGE
# =========================================================

def save_knowledge_index(knowledge):
    """
    Menyimpan JSON secara atomik menggunakan
    temporary file dan os.replace().
    """

    path = os.path.abspath(
        os.fspath(KNOWLEDGE_INDEX_PATH)
    )

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

            json.dump(
                knowledge,
                file,
                ensure_ascii=False,
                indent=4,
            )

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

def build_knowledge_index(sections, chunks):
    """
    Membangun knowledge_index.json.

    Input:
        sections: hasil build_sections()
        chunks: hasil chunk_sections()

    Output:
        knowledge_index.json

    Tanggung jawab:
        - Menyusun struktur heading.
        - Menyusun struktur subheading.
        - Mengambil alias dari ChromaDB.
        - Menghubungkan referensi content chunks.
        - Menyimpan indeks struktural dalam JSON.

    Tidak melakukan:
        - Membuat embedding.
        - Menyimpan embedding.
        - Membuat embedding text.
        - Menghitung hash semantic.
        - Melakukan semantic retrieval.
        - Mempertahankan data lama.
    """

    if not sections:
        raise ValueError(
            "Sections kosong. Knowledge index tidak dibuat."
        )

    print("\n" + "=" * 65)
    print("MEMBANGUN KNOWLEDGE INDEX")
    print("=" * 65)

    # 1. Kelompokkan chunk.
    chunks_by_subheading = index_chunks_by_subheading(
        chunks
    )

    # 2. Kelompokkan heading dan subheading.
    heading_groups = collect_section_details(
        sections
    )

    if not heading_groups:
        raise ValueError(
            "Tidak ditemukan heading yang valid."
        )

    # 3. Bangun indeks heading.
    heading_index = []

    for heading_data in heading_groups:

        heading_item = build_heading_index(
            heading_data=heading_data,
            chunks_by_subheading=chunks_by_subheading,
        )

        heading_index.append(heading_item)

        subheading_count = len(
            heading_item["subheadings"]
        )

        heading_alias_count = len(
            heading_item["aliases"]
        )

        subheading_alias_count = sum(
            len(item["aliases"])
            for item in heading_item["subheadings"]
        )

        print(
            f"HEADING: {heading_item['heading']} "
            f"| subheadings={subheading_count} "
            f"| aliases={heading_alias_count} "
            f"| subheading_aliases={subheading_alias_count}"
        )

    # 4. Urutkan heading.
    heading_index.sort(
        key=lambda item: item["heading"].casefold()
    )

    # 5. Hitung statistik.
    total_subheadings = sum(
        len(item["subheadings"])
        for item in heading_index
    )

    total_heading_aliases = sum(
        len(item["aliases"])
        for item in heading_index
    )

    total_subheading_aliases = sum(
        len(subheading["aliases"])
        for item in heading_index
        for subheading in item["subheadings"]
    )

    total_chunks = sum(
        len(subheading["content_chunks"])
        for item in heading_index
        for subheading in item["subheadings"]
    )

    # 6. Bentuk struktur JSON.
    knowledge = {
        "version": INDEX_VERSION,
        "headings": heading_index,
        "statistics": {
            "total_headings": len(heading_index),
            "total_subheadings": total_subheadings,
            "total_heading_aliases": total_heading_aliases,
            "total_subheading_aliases": total_subheading_aliases,
            "total_chunks": total_chunks,
        },
    }

    # 7. Simpan JSON.
    save_knowledge_index(knowledge)

    # 8. Tampilkan ringkasan.
    print("-" * 65)
    print(f"Jumlah heading          : {len(heading_index)}")
    print(f"Jumlah subheading       : {total_subheadings}")
    print(f"Alias heading           : {total_heading_aliases}")
    print(f"Alias subheading        : {total_subheading_aliases}")
    print(f"Referensi chunks        : {total_chunks}")
    print(f"Knowledge index         : {KNOWLEDGE_INDEX_PATH}")
    print("=" * 65)

    return knowledge
