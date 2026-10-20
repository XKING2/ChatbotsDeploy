
from database.chroma_db import (
    entity_collection,
    chunk_collection,
)

print("=== STATISTIK ===")
print("Entity collection:", entity_collection.count())
print("Chunk collection:", chunk_collection.count())

print("\n=== CONTOH ITEM ENTITY ===")
data = entity_collection.get(
    limit=10,
    include=["documents", "metadatas"]
)

for doc_id, document, metadata in zip(
    data["ids"],
    data["documents"],
    data["metadatas"],
):
    print({
        "id": doc_id,
        "document": document,
        "type": metadata.get("type"),
        "entity": metadata.get("entity"),
        "alias": metadata.get("alias"),
        "subtopic": metadata.get("subtopic"),
        "description": metadata.get("description"),
        "examples": metadata.get("examples"),
    })

print("\n=== JUMLAH TIPE ITEM ===")
for item_type in ("entity", "alias", "subtopic"):
    result = entity_collection.get(
        where={"type": item_type},
        include=["metadatas"],
    )
    print(f"{item_type}: {len(result['ids'])}")

print("\n=== CONTOH ALIAS ===")
aliases = entity_collection.get(
    where={"type": "alias"},
    include=["documents", "metadatas"],
    limit=10,
)

for doc_id, document, metadata in zip(
    aliases["ids"],
    aliases["documents"],
    aliases["metadatas"],
):
    print({
        "id": doc_id,
        "document": document,
        "alias": metadata.get("alias"),
        "entity_id": metadata.get("entity_id"),
    })
