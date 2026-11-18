
import re

from database.hashing import compute_hash


DOCUMENT_NAME = "Knowledge Base Kantor Imigrasi"
CATEGORY = "Layanan Imigrasi"


# =========================================================
# DOCUMENT MARKERS
# =========================================================

HEADING_PATTERN = re.compile(
    r"^Heading\s*:\s*(.+?)\s*$",
    re.IGNORECASE
)

SUBHEADING_PATTERN = re.compile(
    r"^Sub\s*Heading\s*:\s*(.+?)\s*$",
    re.IGNORECASE
)

CONTENT_PATTERN = re.compile(
    r"^Content\s*:\s*(.*)$",
    re.IGNORECASE
)


# =========================================================
# TEXT UTILITIES
# =========================================================

def clean_marker_text(pattern, text):
    match = pattern.match(text.strip())

    return (
        match.group(1).strip()
        if match
        else text.strip()
    )


def is_separator(text):
    text = text.strip()

    return len(text) > 5 and set(text) == {"="}


# =========================================================
# ID GENERATION
# =========================================================

def generate_heading_id(heading):
    return "head_" + compute_hash(
        heading.strip().casefold()
    )[:12]


def generate_subheading_id(heading, subheading):
    key = (
        f"{heading.strip().casefold()}::"
        f"{subheading.strip().casefold()}"
    )

    return "sub_" + compute_hash(key)[:12]


# =========================================================
# SECTION STORAGE
# =========================================================

def save_section(
    sections,
    heading,
    subheading,
    page_start,
    page_end,
    content,
    occurrences
):
    if not heading or not subheading or not content:
        return

    clean_content = " ".join(
        part.strip()
        for part in content
        if part and part.strip()
    )

    if not clean_content:
        return

    heading = heading.strip()
    subheading = subheading.strip()

    heading_id = generate_heading_id(heading)

    subheading_id = generate_subheading_id(
        heading,
        subheading
    )

    occurrences[subheading_id] = (
        occurrences.get(subheading_id, 0) + 1
    )

    sections.append({
        "section_id": len(sections) + 1,
        "source": "knowledge_base",
        "document": DOCUMENT_NAME,
        "category": CATEGORY,

        "heading_id": heading_id,
        "heading": heading,

        "subheading_id": subheading_id,
        "subheading": subheading,

        "subheading_occurrence": occurrences[subheading_id],

        "page_start": page_start,
        "page_end": page_end,

        "content": clean_content
    })


# =========================================================
# SECTION BUILDER
# =========================================================

def build_sections(parsed_data):
    sections = []
    occurrences = {}

    current_heading = None
    current_subheading = None
    current_content = []

    page_start = None
    page_end = None

    for item in parsed_data:

        text = (item.get("text") or "").strip()
        page = item.get("page")

        if not text or is_separator(text):
            continue

        # -------------------------------------------------
        # HEADING
        # -------------------------------------------------

        if HEADING_PATTERN.match(text):

            save_section(
                sections,
                current_heading,
                current_subheading,
                page_start,
                page_end,
                current_content,
                occurrences
            )

            current_heading = clean_marker_text(
                HEADING_PATTERN,
                text
            )

            current_subheading = None
            current_content = []

            page_start = None
            page_end = None

            continue

        # -------------------------------------------------
        # SUBHEADING
        # -------------------------------------------------

        if SUBHEADING_PATTERN.match(text):

            save_section(
                sections,
                current_heading,
                current_subheading,
                page_start,
                page_end,
                current_content,
                occurrences
            )

            current_subheading = clean_marker_text(
                SUBHEADING_PATTERN,
                text
            )

            current_content = []

            page_start = page
            page_end = page

            continue

        # -------------------------------------------------
        # CONTENT
        # -------------------------------------------------

        if current_heading and current_subheading:

            if CONTENT_PATTERN.match(text):
                text = clean_marker_text(
                    CONTENT_PATTERN,
                    text
                )

            if text:

                current_content.append(text)

                if page_start is None:
                    page_start = page

                page_end = page

    # -----------------------------------------------------
    # SAVE LAST SECTION
    # -----------------------------------------------------

    save_section(
        sections,
        current_heading,
        current_subheading,
        page_start,
        page_end,
        current_content,
        occurrences
    )

    return sections
