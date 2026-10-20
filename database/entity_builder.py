from database.hashing import compute_hash


def collect_entities(sections):
    entities = {}

    for section in sections:
        entity = (section.get("entity") or "").strip()
        entity_id = (section.get("entity_id") or "").strip()
        if not entity or not entity_id:
            continue

        item = entities.setdefault(entity_id, {
            "entity_id": entity_id,
            "entity": entity,
            "subtopics": {}
        })

        subtopic = (section.get("subtopic") or "").strip()
        subtopic_id = (section.get("subtopic_id") or "").strip()
        if subtopic and subtopic_id:
            item["subtopics"].setdefault(subtopic_id, {
                "subtopic_id": subtopic_id,
                "subtopic": subtopic
            })

    return list(entities.values())


def build_entity_embedding_text(entity, aliases=None, subtopics=None):
    entity = (entity or "").strip()
    parts = ["Layanan", entity]

    aliases = [
        str(value).strip() for value in (aliases or [])
        if str(value).strip()
    ]
    if aliases:
        parts.extend(["Nama lain", ", ".join(aliases)])

    subtopics = [
        str(value).strip() for value in (subtopics or [])
        if str(value).strip()
    ]
    if subtopics:
        parts.extend(["Topik layanan", ", ".join(subtopics)])

    return "\n\n".join(parts).strip()


def build_subtopic_embedding_text(entity, subtopic, description=None, examples=None):
    entity = (entity or "").strip()
    subtopic = (subtopic or "").strip()
    description = (description or "").strip()

    if not isinstance(examples, list):
        examples = [examples] if examples else []
    examples = [str(value).strip() for value in examples if str(value).strip()]

    parts = ["Layanan", entity, "Subtopic", subtopic]
    if description:
        parts.extend(["Deskripsi", description])
    if examples:
        parts.extend(["Contoh pertanyaan", " | ".join(examples)])

    return "\n\n".join(parts).strip()


def generate_entity_hash(entity_text):
    return compute_hash(entity_text)


def generate_subtopic_hash(subtopic_text):
    return compute_hash(subtopic_text)
