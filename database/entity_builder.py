from database.hashing import compute_hash


def collect_entities(sections):
    entities = {}

    for section in sections:
        entity = (section.get("entity") or "").strip()
        entity_id = (section.get("entity_id") or "").strip()

        if not entity or not entity_id:
            continue

        if entity_id not in entities:
            entities[entity_id] = {
                "entity_id": entity_id,
                "entity": entity,
                "subtopics": {}
            }

        subtopic = (section.get("subtopic") or "").strip()
        subtopic_id = (section.get("subtopic_id") or "").strip()

        if not subtopic or not subtopic_id:
            continue

        if subtopic_id not in entities[entity_id]["subtopics"]:
            entities[entity_id]["subtopics"][subtopic_id] = {
                "subtopic_id": subtopic_id,
                "subtopic": subtopic
            }

    return list(entities.values())


def build_entity_embedding_text(
    entity,
    aliases=None,
    subtopics=None
):
    entity = (entity or "").strip()

    parts = [
        "Layanan",
        entity
    ]

    if aliases:
        aliases = [
            str(alias).strip()
            for alias in aliases
            if str(alias).strip()
        ]

        if aliases:
            parts.extend([
                "Nama lain",
                ", ".join(aliases)
            ])

    if subtopics:
        subtopics = [
            str(subtopic).strip()
            for subtopic in subtopics
            if str(subtopic).strip()
        ]

        if subtopics:
            parts.extend([
                "Topik layanan",
                ", ".join(subtopics)
            ])

    return "\n\n".join(parts).strip()


def build_subtopic_embedding_text(
    entity,
    subtopic,
    description=None,
    examples=None
):
    entity = (entity or "").strip()
    subtopic = (subtopic or "").strip()
    description = (description or "").strip()

    if not isinstance(examples, list):
        examples = [examples] if examples else []

    examples = [
        str(example).strip()
        for example in examples
        if str(example).strip()
    ]

    parts = [
        "Layanan",
        entity,
        "Subtopic",
        subtopic
    ]

    if description:
        parts.extend([
            "Deskripsi",
            description
        ])

    if examples:
        parts.extend([
            "Contoh pertanyaan",
            " | ".join(examples)
        ])

    return "\n\n".join(parts).strip()


def generate_entity_hash(entity_text):
    return compute_hash(entity_text)


def generate_subtopic_hash(subtopic_text):
    return compute_hash(subtopic_text)