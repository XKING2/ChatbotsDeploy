import chromadb

from config import (
    CHROMA_DB_PATH,
    CHUNK_COLLECTION_NAME,
    ENTITY_COLLECTION_NAME
)


client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)


chunk_collection = client.get_or_create_collection(
    name=CHUNK_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


entity_collection = client.get_or_create_collection(
    name=ENTITY_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# ============================================================
# ID
# ============================================================

def alias_id(entity_id, alias):
    from database.hashing import compute_hash

    alias_key = alias.strip().lower()

    return (
        f"{entity_id}"
        f"::alias::{compute_hash(alias_key)[:12]}"
    )


def subtopic_id(entity_id, index):
    return f"{entity_id}::subtopic::{index:03d}"


def chunk_id(entity_id, subtopic_index, index):
    return (
        f"{entity_id}"
        f"::subtopic::{subtopic_index:03d}"
        f"::chunk::{index:03d}"
    )


# ============================================================
# CHUNK
# ============================================================

def get_existing_chunk_hashes():

    data = chunk_collection.get(
        include=["metadatas"]
    )

    return {
        id_: meta.get("content_hash")
        for id_, meta in zip(
            data["ids"],
            data["metadatas"]
        )
    }


def add_chunks(chunks):

    if not chunks:
        return

    chunk_collection.upsert(
        ids=[
            chunk["content_chunk_id"]
            for chunk in chunks
        ],

        documents=[
            chunk["retrieval_text"]
            for chunk in chunks
        ],

        embeddings=[
            chunk["embedding"]
            for chunk in chunks
        ],

        metadatas=[
            {
                "type": "chunk",
                "document": chunk["document"],
                "category": chunk["category"],
                "entity_id": chunk["entity_id"],
                "entity": chunk["entity"],
                "subtopic_id": chunk["subtopic_id"],
                "subtopic": chunk["subtopic"],
                "content_chunk_id": chunk["content_chunk_id"],
                "content_hash": chunk["content_hash"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"]
            }
            for chunk in chunks
        ]
    )

    print(
        f"✓ {len(chunks)} chunk disimpan."
    )


def get_chunks_by_entity(entity_id):

    return chunk_collection.get(
        where={
            "entity_id": entity_id
        },
        include=[
            "documents",
            "metadatas"
        ]
    )


def get_chunks_by_subtopic(
    entity_id,
    subtopic_id
):

    return chunk_collection.get(
        where={
            "$and": [
                {
                    "entity_id": entity_id
                },
                {
                    "subtopic_id": subtopic_id
                }
            ]
        },
        include=[
            "documents",
            "metadatas"
        ]
    )


# ============================================================
# ENTITY / ALIAS / SUBTOPIC
# ============================================================

def get_existing_entity_hashes():

    data = entity_collection.get(
        include=["metadatas"]
    )

    return {
        id_: meta.get("content_hash")
        for id_, meta in zip(
            data["ids"],
            data["metadatas"]
        )
    }


def add_entity_item(
    item_type,
    item_id,
    entity_id,
    text,
    embedding,
    content_hash,
    **extra
):

    if item_type not in {
        "entity",
        "alias",
        "subtopic"
    }:
        raise ValueError(
            f"Tipe entity tidak valid: {item_type}"
        )

    metadata = {
        "type": item_type,
        "entity_id": entity_id,
        "content_hash": content_hash,
        **extra
    }

    if item_type == "entity":

        metadata["entity"] = text

    elif item_type == "alias":

        metadata["alias_id"] = item_id
        metadata["alias"] = text

    elif item_type == "subtopic":

        metadata["subtopic_id"] = item_id
        metadata["subtopic"] = text

    entity_collection.upsert(
        ids=[item_id],

        documents=[text],

        embeddings=[embedding],

        metadatas=[metadata]
    )


# ============================================================
# RELATION
# ============================================================

def get_entity(entity_id):

    return entity_collection.get(
        ids=[entity_id],
        include=[
            "documents",
            "metadatas",
            "embeddings"
        ]
    )


def get_entity_children(entity_id):

    return entity_collection.get(
        where={
            "entity_id": entity_id
        },
        include=[
            "documents",
            "metadatas",
            "embeddings"
        ]
    )


def get_entities():

    return entity_collection.get(
        where={
            "type": "entity"
        },
        include=[
            "documents",
            "metadatas"
        ]
    )


def get_subtopics(entity_id):

    return entity_collection.get(
        where={
            "$and": [
                {
                    "type": "subtopic"
                },
                {
                    "entity_id": entity_id
                }
            ]
        },
        include=[
            "documents",
            "metadatas"
        ]
    )


def get_aliases(entity_id):

    return entity_collection.get(
        where={
            "$and": [
                {
                    "type": "alias"
                },
                {
                    "entity_id": entity_id
                }
            ]
        },
        include=[
            "documents",
            "metadatas"
        ]
    )


# ============================================================
# INFO
# ============================================================

def database_stats():

    return {
        CHUNK_COLLECTION_NAME: chunk_collection.count(),
        ENTITY_COLLECTION_NAME: entity_collection.count()
    }


def print_database_stats():

    stats = database_stats()

    print("\n" + "=" * 50)
    print("CHROMADB")
    print("=" * 50)

    for name, count in stats.items():
        print(
            f"{name}: {count}"
        )

    print("=" * 50)


# ============================================================
# RESET
# ============================================================

def reset_database(collection_name=None):

    names = (
        [collection_name]
        if collection_name
        else [
            CHUNK_COLLECTION_NAME,
            ENTITY_COLLECTION_NAME
        ]
    )

    for name in names:

        try:
            client.delete_collection(name)

        except Exception:
            pass

        client.get_or_create_collection(
            name=name,
            metadata={
                "hnsw:space": "cosine"
            }
        )

    print(
        "✓ Database berhasil direset."
    )