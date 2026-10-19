from config import MAX_CHUNK_SIZE, CHUNK_OVERLAP
from database.hashing import compute_hash


def split_text(text: str):
    text = text.strip()

    if len(text) <= MAX_CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + MAX_CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP

    return chunks


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


def chunk_sections(sections):
    all_chunks = []

    for section in sections:
        text_chunks = split_text(section["content"])
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

    return all_chunks