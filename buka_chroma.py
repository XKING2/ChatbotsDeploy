
from collections import Counter, defaultdict

from database.chroma_db import (
    entity_collection,
    chunk_collection,
)


def safe_get(metadata, key):
    return metadata.get(key) if metadata else None


def get_items(collection, item_type=None):
    """Ambil seluruh item collection beserta dokumen dan metadata."""
    kwargs = {
        "include": ["documents", "metadatas"],
    }

    if item_type:
        kwargs["where"] = {"type": item_type}

    result = collection.get(**kwargs)

    return [
        {
            "id": item_id,
            "document": document,
            "metadata": metadata or {},
        }
        for item_id, document, metadata in zip(
            result["ids"],
            result["documents"],
            result["metadatas"],
        )
    ]


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# =========================================================
# 1. STATISTIK COLLECTION
# =========================================================

print_section("STATISTIK CHROMADB")

print("Entity collection:", entity_collection.count())
print("Chunk collection :", chunk_collection.count())


# =========================================================
# 2. AMBIL DATA ENTITY, SUBTOPIC, DAN ALIAS
# =========================================================

entities = get_items(entity_collection, "entity")
subtopics = get_items(entity_collection, "subtopic")
aliases = get_items(entity_collection, "alias")

print_section("JUMLAH ITEM BERDASARKAN TYPE")

print("Entity          :", len(entities))
print("Subtopic        :", len(subtopics))
print("Semua alias     :", len(aliases))


# =========================================================
# 3. PISAHKAN ALIAS ENTITY DAN ALIAS SUBTOPIC
# =========================================================
# Alias subtopic yang dibuat oleh script sinkronisasi
# mempunyai metadata alias_scope="subtopic".
#
# Fallback subtopic_id digunakan untuk mendeteksi data dari
# versi script sebelumnya yang belum memiliki alias_scope.

entity_aliases = []
subtopic_aliases = []
unclassified_aliases = []

for item in aliases:
    metadata = item["metadata"]
    scope = metadata.get("alias_scope")
    subtopic_id = metadata.get("subtopic_id")

    if scope == "subtopic":
        subtopic_aliases.append(item)
    elif scope == "entity":
        entity_aliases.append(item)
    elif subtopic_id:
        subtopic_aliases.append(item)
    else:
        entity_aliases.append(item)

print_section("PEMISAHAN ALIAS")

print("Alias entity       :", len(entity_aliases))
print("Alias subtopic     :", len(subtopic_aliases))
print("Alias tanpa klasifikasi khusus:", len(unclassified_aliases))


# =========================================================
# 4. CONTOH ALIAS SUBTOPIC
# =========================================================

print_section("CONTOH ALIAS SUBTOPIC - MAKSIMAL 20")

for item in subtopic_aliases[:20]:
    metadata = item["metadata"]

    print({
        "id": item["id"],
        "document": item["document"],
        "alias": metadata.get("alias"),
        "entity": metadata.get("entity"),
        "entity_id": metadata.get("entity_id"),
        "subtopic": metadata.get("subtopic"),
        "subtopic_id": metadata.get("subtopic_id"),
        "alias_scope": metadata.get("alias_scope"),
        "content_hash": metadata.get("content_hash"),
        "embedding_text": metadata.get("embedding_text"),
    })


# =========================================================
# 5. HITUNG ALIAS PER HEADING
# =========================================================

print_section("JUMLAH ALIAS PER HEADING")

aliases_by_heading = defaultdict(list)

for item in subtopic_aliases:
    metadata = item["metadata"]

    key = (
        metadata.get("entity_id"),
        metadata.get("subtopic_id"),
        metadata.get("subtopic"),
    )

    aliases_by_heading[key].append(
        metadata.get("alias") or item["document"]
    )

if aliases_by_heading:
    for (entity_id, subtopic_id, heading), values in sorted(
        aliases_by_heading.items(),
        key=lambda x: (
            str(x[0][0]),
            str(x[0][2]),
        ),
    ):
        print({
            "entity_id": entity_id,
            "subtopic_id": subtopic_id,
            "heading": heading,
            "jumlah_alias": len(values),
            "aliases": values,
        })
else:
    print("Belum ditemukan alias subtopic.")


# =========================================================
# 6. VALIDASI HUBUNGAN ALIAS DENGAN SUBTOPIC
# =========================================================

print_section("VALIDASI HUBUNGAN ALIAS DENGAN SUBTOPIC")

# Gunakan pasangan entity_id + subtopic_id agar heading
# dari entity berbeda tidak tercampur.

subtopic_keys = set()
subtopic_by_key = {}

for item in subtopics:
    metadata = item["metadata"]
    entity_id = metadata.get("entity_id")
    subtopic_id = metadata.get("subtopic_id")

    if entity_id is None or subtopic_id is None:
        continue

    key = (str(entity_id), str(subtopic_id))
    subtopic_keys.add(key)
    subtopic_by_key[key] = item

alias_keys = set()
orphan_aliases = []

for item in subtopic_aliases:
    metadata = item["metadata"]
    entity_id = metadata.get("entity_id")
    subtopic_id = metadata.get("subtopic_id")

    if entity_id is None or subtopic_id is None:
        orphan_aliases.append(item)
        continue

    key = (str(entity_id), str(subtopic_id))

    if key in subtopic_keys:
        alias_keys.add(key)
    else:
        orphan_aliases.append(item)

subtopics_with_alias = len(alias_keys)
subtopics_without_alias = []

for key, item in subtopic_by_key.items():
    if key not in alias_keys:
        metadata = item["metadata"]

        subtopics_without_alias.append({
            "id": item["id"],
            "entity_id": metadata.get("entity_id"),
            "subtopic_id": metadata.get("subtopic_id"),
            "subtopic": (
                metadata.get("subtopic")
                or item["document"]
            ),
        })

print("Total subtopic dengan ID valid :", len(subtopic_keys))
print("Subtopic memiliki alias        :", subtopics_with_alias)
print("Subtopic tanpa alias           :", len(subtopics_without_alias))
print("Alias tanpa pasangan subtopic  :", len(orphan_aliases))

if subtopic_keys:
    percentage = (
        subtopics_with_alias / len(subtopic_keys)
    ) * 100
    print(f"Persentase subtopic beralias   : {percentage:.2f}%")


if subtopics_without_alias:
    print("\nContoh subtopic tanpa alias (maksimal 20):")
    for item in subtopics_without_alias[:20]:
        print(item)

if orphan_aliases:
    print("\nContoh alias tanpa pasangan subtopic (maksimal 20):")
    for item in orphan_aliases[:20]:
        metadata = item["metadata"]

        print({
            "id": item["id"],
            "alias": metadata.get("alias") or item["document"],
            "entity_id": metadata.get("entity_id"),
            "subtopic_id": metadata.get("subtopic_id"),
            "subtopic": metadata.get("subtopic"),
        })


# =========================================================
# 7. VALIDASI METADATA ALIAS
# =========================================================

print_section("VALIDASI METADATA ALIAS SUBTOPIC")

missing_subtopic_id = 0
missing_entity_id = 0
missing_content_hash = 0
missing_embedding_text = 0

for item in subtopic_aliases:
    metadata = item["metadata"]

    if not metadata.get("subtopic_id"):
        missing_subtopic_id += 1

    if not metadata.get("entity_id"):
        missing_entity_id += 1

    if not metadata.get("content_hash"):
        missing_content_hash += 1

    if not metadata.get("embedding_text"):
        missing_embedding_text += 1

print("Alias tanpa subtopic_id  :", missing_subtopic_id)
print("Alias tanpa entity_id    :", missing_entity_id)
print("Alias tanpa content_hash :", missing_content_hash)
print("Alias tanpa embedding_text:", missing_embedding_text)

if subtopic_aliases:
    print("\nCatatan:")
    print(
        "Metadata embedding_text hanya tersedia jika fungsi "
        "add_entity_item menyimpannya. Jika kosong, hal itu "
        "tidak otomatis berarti vektor embedding tidak ada."
    )


# =========================================================
# 8. RINGKASAN AKHIR
# =========================================================

print_section("RINGKASAN AKHIR")

print("Total item entity collection :", entity_collection.count())
print("Total entity                 :", len(entities))
print("Total subtopic               :", len(subtopics))
print("Total alias                  :", len(aliases))
print("Total alias entity           :", len(entity_aliases))
print("Total alias subtopic         :", len(subtopic_aliases))
print("Subtopic memiliki alias      :", subtopics_with_alias)
print("Subtopic tanpa alias         :", len(subtopics_without_alias))
print("Alias tanpa pasangan         :", len(orphan_aliases))

print("\nPemeriksaan selesai.")
