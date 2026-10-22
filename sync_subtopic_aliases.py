import hashlib
import json
import logging
import re
import sys
from pathlib import Path

from config import DATA_DIR
from database.embedding import create_embedding
from database.chroma_db import (
    add_entity_item,
    get_existing_entity_hashes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

KNOWLEDGE_INDEX_PATH = Path(DATA_DIR) / "knowledge_index.json"


def normalize_text(text: str) -> str:
    """Normalisasi whitespace tanpa mengubah ejaan alias."""
    return re.sub(r"\s+", " ", str(text)).strip()


def make_hash(text: str) -> str:
    """Hash stabil untuk mendeteksi perubahan konten."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_alias_id(
    entity_id: str,
    subtopic_id: str,
    alias: str,
) -> str:
    alias_key = normalize_text(alias).casefold()
    alias_hash = make_hash(alias_key)[:16]

    return (
        f"{entity_id}::subtopic::{subtopic_id}"
        f"::alias::{alias_hash}"
    )


def load_knowledge_index() -> list:
    if not KNOWLEDGE_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge index tidak ditemukan: "
            f"{KNOWLEDGE_INDEX_PATH}"
        )

    with KNOWLEDGE_INDEX_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Format knowledge_index.json harus berupa object JSON."
        )

    entities = data.get("entities")

    if not isinstance(entities, list):
        raise ValueError(
            "Field 'entities' tidak ditemukan atau bukan list."
        )

    return entities


def build_alias_embedding_text(
    entity_name: str,
    heading_name: str,
    alias: str,
) -> str:

    return (
        f"Layanan: {entity_name}\n"
        f"Heading: {heading_name}\n"
        f"Alias: {alias}"
    )


def collect_subtopic_aliases(entities: list) -> list:

    records = []
    seen = set()

    for entity in entities:
        entity_id = normalize_text(
            entity.get("entity_id", "")
        )
        entity_name = normalize_text(
            entity.get("entity", "")
        )

        if not entity_id or not entity_name:
            logger.warning(
                "Entity dilewati karena entity_id/entity kosong."
            )
            continue

        subtopics = entity.get("subtopics", [])

        if not isinstance(subtopics, list):
            continue

        for subtopic in subtopics:
            if not isinstance(subtopic, dict):
                continue

            subtopic_id = normalize_text(
                subtopic.get("subtopic_id", "")
            )
            heading_name = normalize_text(
                subtopic.get("subtopic", "")
            )
            aliases = subtopic.get("aliases", [])

            if not subtopic_id or not heading_name:
                logger.warning(
                    "Subtopic dilewati pada entity %s "
                    "karena ID/nama heading kosong.",
                    entity_id,
                )
                continue

            if not isinstance(aliases, list):
                continue

            for raw_alias in aliases:
                if not isinstance(raw_alias, str):
                    continue

                alias = normalize_text(raw_alias)
                if not alias:
                    continue

                # Deduplikasi hanya dalam heading yang sama.
                dedupe_key = (
                    entity_id,
                    subtopic_id,
                    alias.casefold(),
                )
                if dedupe_key in seen:
                    continue

                seen.add(dedupe_key)

                embedding_text = build_alias_embedding_text(
                    entity_name,
                    heading_name,
                    alias,
                )

                records.append({
                    "entity_id": entity_id,
                    "entity": entity_name,
                    "subtopic_id": subtopic_id,
                    "subtopic": heading_name,
                    "alias": alias,
                    "embedding_text": embedding_text,
                    "item_id": make_alias_id(
                        entity_id,
                        subtopic_id,
                        alias,
                    ),
                })

    return records


def sync_subtopic_aliases(dry_run: bool = False) -> None:
    entities = load_knowledge_index()
    records = collect_subtopic_aliases(entities)

    logger.info("Jumlah entity JSON : %d", len(entities))
    logger.info("Jumlah alias heading: %d", len(records))

    if not records:
        logger.warning("Tidak ada subtopic alias untuk diproses.")
        return

    if dry_run:
        for record in records[:10]:
            logger.info(
                "[DRY RUN] heading=%s | alias=%s | id=%s",
                record["subtopic"],
                record["alias"],
                record["item_id"],
            )

        logger.info(
            "Dry run selesai. Tidak ada embedding atau data "
            "yang ditulis ke ChromaDB."
        )
        return

    # Menggunakan helper proyek untuk mengecek hash item yang ada.
    existing_hashes = get_existing_entity_hashes()

    inserted = 0
    skipped = 0
    failed = 0

    for index, record in enumerate(records, start=1):
        try:
            item_id = record["item_id"]
            embedding_text = record["embedding_text"]
            content_hash = make_hash(embedding_text)

            # Lewati item jika teks embedding tidak berubah.
            if existing_hashes.get(item_id) == content_hash:
                skipped += 1
                continue

            embedding = create_embedding(embedding_text)

            if embedding is None:
                raise ValueError(
                    "create_embedding mengembalikan None."
                )

            add_entity_item(
                item_type="alias",
                item_id=item_id,
                entity_id=record["entity_id"],
                text=record["alias"],
                embedding=embedding,
                content_hash=content_hash,
                alias_scope="subtopic",
                subtopic_id=record["subtopic_id"],
                subtopic=record["subtopic"],
                entity=record["entity"],
                embedding_text=embedding_text,
            )

            # Mencegah pemrosesan ulang ID yang sama dalam run ini.
            existing_hashes[item_id] = content_hash
            inserted += 1

            logger.info(
                "[%d/%d] Tersimpan | %s | %s",
                index,
                len(records),
                record["subtopic"],
                record["alias"],
            )

        except Exception:
            failed += 1
            logger.exception(
                "Gagal memproses alias '%s' pada heading '%s'.",
                record["alias"],
                record["subtopic"],
            )

    logger.info("=" * 55)
    logger.info("Sinkronisasi selesai")
    logger.info("Alias baru/berubah : %d", inserted)
    logger.info("Alias tidak berubah: %d", skipped)
    logger.info("Alias gagal        : %d", failed)
    logger.info("=" * 55)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    try:
        sync_subtopic_aliases(dry_run=dry_run)
    except Exception:
        logger.exception("Sinkronisasi gagal.")
        sys.exit(1)
