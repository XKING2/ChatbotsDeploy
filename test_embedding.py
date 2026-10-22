"""Bandingkan skor alias yang tersimpan dengan skor bila alias di-embed POLOS.

Tujuan: memastikan apakah bungkus teks ("Layanan / <entity> / Nama lain / <alias>")
pada embedding alias mengencerkan makna alias dan menurunkan skor.

  python uji_embedding_alias.py "Apa saja dokumen yang harus dibawa untuk mengurus Affidavit"

Biaya: 1 embedding query + 1 embedding per alias yang diuji (default 8).
"""
import math
import sys

from database.embedding import create_embedding
from retrival.retrieval_pipeline import RetrievalPipeline
from userquery.query_embedding import create_query_embedding
from userquery.query_normalizer import normalize_query

HOW_MANY = 8


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    if len(sys.argv) < 2:
        print('Pemakaian: python uji_embedding_alias.py "<pertanyaan>"')
        return
    question = " ".join(sys.argv[1:])
    result = RetrievalPipeline().retrieve(question)
    hits = result.get("alias_hits", [])[:HOW_MANY]
    if not hits:
        print("Tidak ada alias hit.")
        return

    q_emb = create_query_embedding(normalize_query(question))
    print(f"Question: {question}\n")
    print(f"{'tersimpan':>10} {'polos':>8} {'selisih':>8}  alias  ->  entity | subtopik")
    gains = []
    for hit in hits:
        raw = create_embedding(hit["alias"])
        if raw is None:
            print(f"{hit['score']:>10.4f} {'gagal':>8}  {hit['alias']!r}")
            continue
        plain = cosine(q_emb, raw)
        gains.append(plain - hit["score"])
        print(
            f"{hit['score']:>10.4f} {plain:>8.4f} {plain - hit['score']:>+8.4f}  "
            f"{hit['alias'][:48]!r} -> {(hit.get('entity') or '')[:28]} | {(hit.get('subtopic') or '')[:28]}"
        )
    if gains:
        print(f"\nRata-rata selisih (polos - tersimpan): {sum(gains) / len(gains):+.4f}")
        print("Selisih positif konsisten = bungkus teks mengencerkan alias.")


if __name__ == "__main__":
    main()