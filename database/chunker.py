
import re

from config import MAX_CHUNK_SIZE, CHUNK_OVERLAP
from database.hashing import compute_hash


def split_text(text: str):
    text = text.strip()

    if not text:
        return []

    if len(text) <= MAX_CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + MAX_CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP

    return chunks


def count_sentences(text: str):
    """Menghitung kalimat secara perkiraan berdasarkan tanda baca."""
    if not text or not text.strip():
        return 0

    sentences = re.findall(
        r'[^.!?]+(?:[.!?]+|$)',
        text.strip()
    )

    return sum(1 for sentence in sentences if sentence.strip())


def build_retrieval_text(section, chunk):
    return f"""
Dokumen Layanan Keimigrasian

Nama Dokumen:
{section["document"]}

Kategori:
{section["category"]}

Entity:
{section["entity"]}

Subtopic:
{section["subtopic"]}

Dokumen ini berisi informasi mengenai
{section["subtopic"].lower()}
dari
{section["entity"]}.

Isi Informasi:

{chunk}
""".strip()


def generate_content_chunk_id(
    subtopic_id,
    subtopic_occurrence,
    chunk_index
):
    return f"{subtopic_id}_o{subtopic_occurrence}_c{chunk_index}"


def print_chunk_summary(sections, all_chunks):
    total_sentences = sum(
        count_sentences(section.get("content", ""))
        for section in sections
    )

    print("\n" + "=" * 68)
    print("RINGKASAN HASIL CHUNKING")
    print("=" * 68)

    headers = [
        "Max Chunk Size",
        "Total Kalimat",
        "Total Chunk"
    ]

    values = [
        str(MAX_CHUNK_SIZE),
        str(total_sentences),
        str(len(all_chunks))
    ]

    widths = [20, 20, 20]

    def print_row(items):
        print(
            "| " +
            " | ".join(
                item.center(width)
                for item, width in zip(items, widths)
            ) +
            " |"
        )

    border = "+" + "+".join(
        "-" * (width + 2) for width in widths
    ) + "+"

    print(border)
    print_row(headers)
    print(border)
    print_row(values)
    print(border)

    print(f"Chunk overlap : {CHUNK_OVERLAP} karakter")
    print("=" * 68)


def chunk_sections(sections):
    all_chunks = []

    for section in sections:
        text_chunks = split_text(section.get("content", ""))
        total_chunks = len(text_chunks)

        for index, chunk in enumerate(text_chunks, start=1):
            retrieval_text = build_retrieval_text(
                section,
                chunk
            )

            content_chunk_id = generate_content_chunk_id(
                section["subtopic_id"],
                section.get("subtopic_occurrence", 1),
                index
            )

            all_chunks.append({
                "content_chunk_id": content_chunk_id,
                "content_hash": compute_hash(retrieval_text),

                "document": section["document"],
                "category": section["category"],
                "section_id": section["section_id"],

                "entity_id": section["entity_id"],
                "entity": section["entity"],

                "subtopic_id": section["subtopic_id"],
                "subtopic": section["subtopic"],

                "page_start": section["page_start"],
                "page_end": section["page_end"],

                "chunk_index": index,
                "total_chunks": total_chunks,

                "display_text": chunk,
                "retrieval_text": retrieval_text
            })

    print_chunk_summary(sections, all_chunks)

    return all_chunks


def print_chunk_summary(sections, all_chunks):
    total_sentences = sum(
        count_sentences(section.get("content", ""))
        for section in sections
    )

    print("\n" + "=" * 90)
    print("RINGKASAN HASIL CHUNKING")
    print("=" * 90)

    print(
        f"| {'Max Chunk Size':<20} "
        f"| {'Total Kalimat':<20} "
        f"| {'Total Chunk':<20} |"
    )
    print("-" * 70)

    print(
        f"| {MAX_CHUNK_SIZE:<20} "
        f"| {total_sentences:<20} "
        f"| {len(all_chunks):<20} |"
    )

    print(f"\nChunk overlap: {CHUNK_OVERLAP} karakter")

    print("\n" + "=" * 120)
    print("DETAIL ID CHUNK")
    print("=" * 120)

    headers = [
        "No.",
        "Content Chunk ID",
        "Subtopic ID",
        "Chunk",
        "Total",
        "Karakter"
    ]

    widths = [5, 38, 25, 8, 8, 10]

    def print_row(values):
        print(
            "| " + " | ".join(
                str(value)[:width].ljust(width)
                for value, width in zip(values, widths)
            ) + " |"
        )

    border = "+" + "+".join(
        "-" * (width + 2) for width in widths
    ) + "+"

    print(border)
    print_row(headers)
    print(border)

    for number, chunk in enumerate(all_chunks[:10], start=1):
        print_row([
            number,
            chunk["content_chunk_id"],
            chunk["subtopic_id"],
            chunk["chunk_index"],
            chunk["total_chunks"],
            len(chunk["display_text"])
        ])

    print(border)
    print(f"Total ID chunk ditampilkan: {min(len(all_chunks), 10)}")
    print("=" * 120)
