import os
import re
import sys
import fitz

from database.hashing import compute_hash


DOCUMENT_NAME = "Knowledge Base Kantor Imigrasi"
CATEGORY = "Layanan Imigrasi"

ENTITY_PATTERN = re.compile(
    r"(?i)^Heading\s*:\s*(.+?)\s*$"
)

SUBTOPIC_PATTERN = re.compile(
    r"(?i)^Sub\s*Heading\s*:\s*(.+?)\s*$"
)


def parse_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF tidak ditemukan:\n{pdf_path}"
        )

    parsed_data = []

    with fitz.open(pdf_path) as document:

        for page_number, page in enumerate(
            document,
            start=1
        ):
            text = page.get_text("text")

            if not text:
                continue

            for line in text.splitlines():

                line = line.strip()

                if line:
                    parsed_data.append({
                        "text": line,
                        "page": page_number
                    })

    return parsed_data


def clean_marker_text(pattern, text):
    match = pattern.match(text.strip())

    return (
        match.group(1).strip()
        if match
        else text.strip()
    )


def is_separator(text):
    text = text.strip()

    return (
        len(text) > 5
        and set(text) == {"="}
    )


def generate_entity_id(entity):
    return (
        "ent_"
        + compute_hash(
            entity.strip().lower()
        )[:12]
    )


def generate_subtopic_id(entity, subtopic):
    key = (
        f"{entity.strip().lower()}::"
        f"{subtopic.strip().lower()}"
    )

    return (
        "sub_"
        + compute_hash(key)[:12]
    )


def save_section(
    sections,
    section_id,
    entity,
    subtopic,
    page_start,
    page_end,
    content,
    occurrence_counter
):

    if not entity or not subtopic:
        return section_id

    content = " ".join(
        part.strip()
        for part in content
        if part and part.strip()
    ).strip()

    if not content:
        return section_id

    entity = entity.strip()
    subtopic = subtopic.strip()

    entity_id = generate_entity_id(entity)

    subtopic_id = generate_subtopic_id(
        entity,
        subtopic
    )

    occurrence = (
        occurrence_counter.get(
            subtopic_id,
            0
        ) + 1
    )

    occurrence_counter[subtopic_id] = occurrence

    sections.append({
        "section_id": section_id,
        "source": "knowledge_base",
        "document": DOCUMENT_NAME,
        "category": CATEGORY,
        "entity_id": entity_id,
        "entity": entity,
        "subtopic_id": subtopic_id,
        "subtopic": subtopic,
        "subtopic_occurrence": occurrence,
        "page_start": page_start,
        "page_end": page_end,
        "content": content
    })

    return section_id + 1


def build_sections(parsed_data):

    sections = []
    section_id = 1

    occurrence_counter = {}

    entity = None
    subtopic = None

    content = []

    page_start = None
    page_end = None

    for item in parsed_data:

        text = (
            item.get("text") or ""
        ).strip()

        page = item.get("page")

        if not text:
            continue

        if is_separator(text):
            continue

        # ==========================================
        # HEADING / ENTITY
        # ==========================================

        if ENTITY_PATTERN.match(text):

            section_id = save_section(
                sections,
                section_id,
                entity,
                subtopic,
                page_start,
                page_end,
                content,
                occurrence_counter
            )

            entity = clean_marker_text(
                ENTITY_PATTERN,
                text
            )

            subtopic = None
            content = []

            page_start = None
            page_end = None

            continue

        # ==========================================
        # SUB HEADING / SUBTOPIC
        # ==========================================

        if SUBTOPIC_PATTERN.match(text):

            section_id = save_section(
                sections,
                section_id,
                entity,
                subtopic,
                page_start,
                page_end,
                content,
                occurrence_counter
            )

            subtopic = clean_marker_text(
                SUBTOPIC_PATTERN,
                text
            )

            content = []

            page_start = page
            page_end = page

            continue

        # ==========================================
        # CONTENT
        # Semua teks setelah Sub Heading
        # dianggap sebagai content
        # ==========================================

        if entity and subtopic:

            content.append(text)

            if page_start is None:
                page_start = page

            page_end = page

    # ==========================================
    # SIMPAN SECTION TERAKHIR
    # ==========================================

    save_section(
        sections,
        section_id,
        entity,
        subtopic,
        page_start,
        page_end,
        content,
        occurrence_counter
    )

    return sections


def print_sections(sections):

    print("\n" + "=" * 80)
    print("HASIL SECTION BUILDER")
    print("=" * 80)

    print(
        f"Total Sections : "
        f"{len(sections)}"
    )

    for section in sections:

        print("\n" + "-" * 80)

        print(
            f"SECTION ID          : "
            f"{section['section_id']}"
        )

        print(
            f"DOCUMENT            : "
            f"{section['document']}"
        )

        print(
            f"CATEGORY            : "
            f"{section['category']}"
        )

        print(
            f"ENTITY              : "
            f"{section['entity']}"
        )

        print(
            f"ENTITY ID           : "
            f"{section['entity_id']}"
        )

        print(
            f"SUBTOPIC            : "
            f"{section['subtopic']}"
        )

        print(
            f"SUBTOPIC ID         : "
            f"{section['subtopic_id']}"
        )

        print(
            f"OCCURRENCE          : "
            f"{section['subtopic_occurrence']}"
        )

        print(
            f"PAGE                : "
            f"{section['page_start']} - "
            f"{section['page_end']}"
        )

        print("\nCONTENT:")
        print(section["content"])


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "python database/section_builder.py "
            "<path_pdf>"
        )

        sys.exit(1)

    pdf_path = sys.argv[1]

    print("=" * 80)
    print("SECTION BUILDER TEST")
    print("=" * 80)

    print(f"PDF : {pdf_path}")

    try:

        parsed_data = parse_pdf(
            pdf_path
        )

        print(
            f"Baris PDF terbaca : "
            f"{len(parsed_data)}"
        )

        sections = build_sections(
            parsed_data
        )

        print_sections(
            sections
        )

    except Exception as e:

        print("\nERROR")
        print("=" * 80)

        print(
            type(e).__name__
        )

        print(e)

        sys.exit(1)