
from config import MAX_CHUNK_SIZE, CHUNK_OVERLAP
from database.hashing import compute_hash


def split_text(text):
    """Memecah teks berdasarkan batas kalimat, kata, dan overlap."""

    text = " ".join(str(text or "").split())

    if not text:
        return []

    if MAX_CHUNK_SIZE <= 0:
        raise ValueError(
            "MAX_CHUNK_SIZE harus lebih besar dari 0."
        )

    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= MAX_CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP harus lebih kecil dari MAX_CHUNK_SIZE."
        )

    if len(text) <= MAX_CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + MAX_CHUNK_SIZE, len(text))

        if end < len(text):
            # Prioritaskan batas akhir kalimat
            boundary = max(
                text.rfind(mark, start, end)
                for mark in (". ", "! ", "? ", "; ")
            )

            if boundary >= start + int(MAX_CHUNK_SIZE * 0.6):
                end = boundary + 1

            else:
                # Jika tidak ada batas kalimat, gunakan batas kata
                boundary = text.rfind(" ", start, end)

                if boundary > start:
                    end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Mulai chunk berikutnya dengan overlap
        next_start = max(start + 1, end - CHUNK_OVERLAP)

        # Hindari memulai chunk di tengah kata
        space = text.find(" ", next_start, end)

        if space != -1:
            next_start = space + 1

        start = next_start

    return chunks


def build_retrieval_text(section, chunk):
    """Menambahkan konteks heading dan subheading ke teks retrieval."""

    return (
        f"Dokumen: {section['document']}\n"
        f"Kategori: {section['category']}\n"
        f"Heading: {section['heading']}\n"
        f"Subheading: {section['subheading']}\n\n"
        f"Informasi:\n{chunk}"
    )


def generate_content_chunk_id(
    subheading_id,
    subheading_occurrence,
    chunk_index,
    source_id=None
):
    """Membuat ID chunk berdasarkan sumber dan subheading."""

    local_id = (
        f"{subheading_id}"
        f"_o{subheading_occurrence}"
        f"_c{chunk_index}"
    )

    return (
        f"{source_id}::{local_id}"
        if source_id
        else local_id
    )


def chunk_sections(sections, source_id=None, source_file=None):
    """Memecah setiap section menjadi chunk beserta metadata dan hash."""

    all_chunks = []

    required_fields = (
        "document",
        "category",
        "section_id",
        "heading_id",
        "heading",
        "subheading_id",
        "subheading",
        "page_start",
        "page_end"
    )

    for section in sections:
        missing = [
            field for field in required_fields
            if field not in section
        ]

        if missing:
            raise ValueError(
                f"Section {section.get('section_id', '?')} "
                f"tidak memiliki field: {', '.join(missing)}"
            )

        current_source_id = (
            section.get("source_id") or source_id
        )

        current_source_file = (
            section.get("source_file") or source_file or ""
        )

        text_chunks = split_text(
            section.get("content", "")
        )

        total_chunks = len(text_chunks)

        for index, chunk in enumerate(text_chunks, start=1):
            retrieval_text = build_retrieval_text(
                section,
                chunk
            )

            chunk_id = generate_content_chunk_id(
                section["subheading_id"],
                section.get("subheading_occurrence", 1),
                index,
                current_source_id
            )

            all_chunks.append({
                "content_chunk_id": chunk_id,
                "content_hash": compute_hash(retrieval_text),

                "source_id": current_source_id,
                "source_file": current_source_file,

                "document": section["document"],
                "category": section["category"],

                "section_id": section["section_id"],

                "heading_id": section["heading_id"],
                "heading": section["heading"],

                "subheading_id": section["subheading_id"],
                "subheading": section["subheading"],
                "subheading_occurrence": section.get(
                    "subheading_occurrence",
                    1
                ),

                "page_start": section["page_start"],
                "page_end": section["page_end"],

                "chunk_index": index,
                "total_chunks": total_chunks,

                "display_text": chunk,
                "retrieval_text": retrieval_text
            })

    return all_chunks
