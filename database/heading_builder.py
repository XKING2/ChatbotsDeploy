
from database.hashing import compute_hash


def collect_headings(sections):
    headings = {}

    for section in sections:
        heading_id = section.get("heading_id")
        heading = section.get("heading")

        subheading_id = section.get("subheading_id")
        subheading = section.get("subheading")

        if not all([
            heading_id,
            heading,
            subheading_id,
            subheading
        ]):
            continue

        if heading_id not in headings:
            headings[heading_id] = {
                "heading_id": heading_id,
                "heading": heading,
                "subheadings": {}
            }

        headings[heading_id]["subheadings"][
            subheading_id
        ] = {
            "subheading_id": subheading_id,
            "subheading": subheading
        }

    return list(headings.values())


def build_heading_embedding_text(
    heading,
    aliases=None,
    subheadings=None
):
    parts = [
        "Layanan:",
        heading
    ]

    aliases = aliases or []
    subheadings = subheadings or []

    if aliases:
        parts.extend([
            "Nama lain:",
            ", ".join(aliases)
        ])

    if subheadings:
        parts.extend([
            "Bagian layanan:",
            ", ".join(subheadings)
        ])

    return "\n".join(parts)


def build_subheading_embedding_text(
    heading,
    subheading,
    description=None,
    examples=None
):
    parts = [
        f"Heading: {heading}",
        f"Subheading: {subheading}"
    ]

    if description:
        parts.extend([
            "Deskripsi:",
            description
        ])

    if examples:
        parts.extend([
            "Contoh pertanyaan:",
            "\n".join(
                f"- {example}"
                for example in examples
            )
        ])

    return "\n".join(parts)


def generate_heading_hash(heading, aliases=None):
    embedding_text = build_heading_embedding_text(
        heading,
        aliases=aliases
    )

    return compute_hash(embedding_text)


def generate_subheading_hash(
    heading,
    subheading,
    description=None,
    examples=None
):
    embedding_text = build_subheading_embedding_text(
        heading,
        subheading,
        description,
        examples
    )

    return compute_hash(embedding_text)
