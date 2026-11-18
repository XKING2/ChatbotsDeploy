"""
Pengelolaan database pengetahuan (ChromaDB generation).

Letakkan di folder proyek (sejajar dengan config.py dan folder database/), lalu:

  python manage_knowledge.py list
  python manage_knowledge.py find   <kata>                 cari chunk aktif yang memuat kata
  python manage_knowledge.py show   <chunk_id> [--generation G]
  python manage_knowledge.py delete-chunk <chunk_id> [<chunk_id> ...] [--yes]
  python manage_knowledge.py subheadings [kata]             daftar heading > subheading beserta id
  python manage_knowledge.py delete-subheading <subheading_id> [--heading <heading_id>] [--yes]
  python manage_knowledge.py activate <generation> [--yes]   (aktifkan / rollback)
  python manage_knowledge.py delete-generation <generation> [--yes]

Setelah activate atau delete-chunk, restart aplikasi Flask agar semua proses memakai data terbaru.
"""
import sys

from database.chroma_db import (
    activate_generation,
    get_heading_collection,
    client,
    delete_generation,
    get_active_generation,
    get_chunk_collection,
    validate_generation,
)
from database.chroma_db import _generation_names  # nama collection per generation


def generations():
    names = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    return sorted({n.rpartition("__g_")[2] for n in names if "__g_" in n})


def chunk_collection_of(generation):
    return client.get_collection(_generation_names(generation)[1]["chunk"])


def confirm(question, yes):
    if yes:
        return True
    return input(f"{question} [ketik 'ya' untuk lanjut]: ").strip().lower() == "ya"


def cmd_list():
    active = get_active_generation()
    print(f"Generation aktif: {active}\n")
    print(f"{'generation':<16}{'status':<10}{'chunk':>7}{'record heading':>16}")
    for generation in generations():
        try:
            info = validate_generation(generation)
            counts = (info["chunk_count"], info["heading_count"])
        except Exception as error:
            counts = ("?", f"rusak: {error}")
        print(f"{generation:<16}{'AKTIF' if generation == active else '-':<10}"
              f"{counts[0]!s:>7}{counts[1]!s:>16}")


def cmd_find(word):
    data = get_chunk_collection().get(include=["documents", "metadatas"])
    hits = [(i, d, m) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
            if word.casefold() in (d or "").casefold()]
    print(f"{len(hits)} chunk aktif memuat '{word}':")
    for rid, doc, meta in hits:
        print(f"- {rid}\n    {meta.get('heading')} > {meta.get('subheading')} | "
              f"hash {str(meta.get('content_hash'))[:12]}\n    "
              f"{' '.join(doc.split())[:160]}")


def cmd_show(chunk_id, generation):
    collection = chunk_collection_of(generation or get_active_generation())
    data = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not data["ids"]:
        print("Chunk tidak ditemukan di generation tersebut.")
        return
    meta = data["metadatas"][0]
    print(f"id         : {chunk_id}")
    for key in ("heading_id", "heading", "subheading_id", "subheading", "content_hash",
                "chunk_index", "total_chunks", "page_start", "page_end", "source_file"):
        print(f"{key:<11}: {meta.get(key)}")
    print("\nIsi:\n" + data["documents"][0])


def cmd_delete_chunks(ids, yes):
    active = get_active_generation()
    collection = get_chunk_collection()
    data = collection.get(ids=ids, include=["metadatas", "documents"])
    found = set(data["ids"])
    for rid in ids:
        if rid not in found:
            print(f"! tidak ada di generation aktif: {rid}")
    if not found:
        return
    print(f"Akan dihapus dari generation AKTIF {active}:")
    for rid, meta, doc in zip(data["ids"], data["metadatas"], data["documents"]):
        print(f"- {rid} | {meta.get('heading')} > {meta.get('subheading')} | "
              f"\"{' '.join(doc.split())[:90]}\"")
    print("\nPerhatian: penghapusan langsung ini tidak bisa dibatalkan di generation aktif, "
          "dan chunk akan muncul lagi pada ingestion berikutnya selama teksnya masih ada di PDF. "
          "Cara permanen: hapus teksnya dari PDF lalu jalankan ingestion.")
    if not confirm("Hapus chunk ini?", yes):
        print("Dibatalkan.")
        return
    collection.delete(ids=sorted(found))
    print(f"Terhapus: {len(found)} chunk. Sisa chunk aktif: {collection.count()}")


def subheading_records(sid, heading_id=None):
    """Record subheading aktif dengan id tersebut (bisa lebih dari satu heading)."""
    data = get_heading_collection().get(where={"type": "subheading"},
                                        include=["metadatas"])
    found = []
    for rid, meta in zip(data["ids"], data["metadatas"]):
        meta = meta or {}
        if sid in (rid, str(meta.get("subheading_id"))):
            if heading_id is None or meta.get("heading_id") == heading_id:
                found.append((rid, meta))
    return found


def cmd_subheadings(word=None):
    heads = get_heading_collection()
    data = heads.get(include=["metadatas"])
    chunks = get_chunk_collection().get(include=["metadatas"])
    chunk_count, alias_count, names = {}, {}, {}
    for meta in chunks["metadatas"]:
        key = (meta.get("heading_id"), str(meta.get("subheading_id")))
        chunk_count[key] = chunk_count.get(key, 0) + 1
    for rid, meta in zip(data["ids"], data["metadatas"]):
        meta = meta or {}
        if meta.get("type") == "alias" and meta.get("alias_scope") == "subheading":
            key = (meta.get("heading_id"), str(meta.get("subheading_id")))
            alias_count[key] = alias_count.get(key, 0) + 1
        elif meta.get("type") == "heading":
            names[rid] = meta.get("heading") or meta.get("name") or rid
    subs = [(m.get("heading_id"), r, m) for r, m in zip(data["ids"], data["metadatas"])
            if (m or {}).get("type") == "subheading"]
    shown = 0
    for hid, rid, meta in sorted(subs, key=lambda x: (str(names.get(x[0])), str(x[2].get("name")))):
        label = meta.get("name") or meta.get("subheading") or rid
        if word and word.casefold() not in f"{names.get(hid, '')} {label}".casefold():
            continue
        key = (hid, str(meta.get("subheading_id") or rid))
        print(f"{names.get(hid, hid)}  [{hid}]\n   > {label}  [{rid}] | "
              f"{chunk_count.get(key, 0)} chunk | {alias_count.get(key, 0)} alias")
        shown += 1
    print(f"\n{shown} subheading" + (f" cocok dengan '{word}'" if word else ""))


def cmd_delete_subheading(sid, heading_id, yes):
    active = get_active_generation()
    records = subheading_records(sid, heading_id)
    if not records:
        raise SystemExit(f"Subheading '{sid}' tidak ditemukan di generation aktif. "
                         "Lihat daftar: python manage_knowledge.py subheadings")
    if len(records) > 1:
        print("Id itu dipakai oleh lebih dari satu heading; tambahkan --heading <heading_id>:")
        for rid, meta in records:
            print(f"- heading {meta.get('heading_id')} | {meta.get('name') or meta.get('subheading')}")
        raise SystemExit(1)

    rid, meta = records[0]
    hid = meta.get("heading_id")
    label = meta.get("name") or meta.get("subheading") or rid
    heads, chunk_col = get_heading_collection(), get_chunk_collection()

    alias_data = heads.get(where={"type": "alias"}, include=["metadatas"])
    alias_ids = [a for a, m in zip(alias_data["ids"], alias_data["metadatas"])
                 if (m or {}).get("alias_scope") == "subheading"
                 and (m or {}).get("heading_id") == hid
                 and str((m or {}).get("subheading_id")) == str(meta.get("subheading_id") or rid)]
    chunk_data = chunk_col.get(include=["metadatas"])
    chunk_ids = [c for c, m in zip(chunk_data["ids"], chunk_data["metadatas"])
                 if (m or {}).get("heading_id") == hid
                 and str((m or {}).get("subheading_id")) == str(meta.get("subheading_id") or rid)]
    heading_row = heads.get(ids=[hid], include=["metadatas"])
    siblings = [m for m in heads.get(where={"type": "subheading"}, include=["metadatas"])["metadatas"]
                if (m or {}).get("heading_id") == hid]

    print(f"Generation aktif : {active}")
    print(f"Heading          : {(heading_row['metadatas'] or [{}])[0].get('heading') if heading_row['ids'] else hid} [{hid}]")
    print(f"Subheading       : {label} [{rid}]")
    print(f"Akan dihapus     : 1 record subheading, {len(alias_ids)} alias subheading, "
          f"{len(chunk_ids)} chunk")
    for cid in chunk_ids:
        print(f"   - chunk {cid}")
    if len(siblings) <= 1:
        print("\n! Ini subheading TERAKHIR pada heading tersebut; heading akan tersisa tanpa isi.")
    print("\nPerhatian: penghapusan ini langsung di generation aktif dan tidak bisa dibatalkan di "
          "generation itu (salinan lama hanya ada di generation sebelumnya). Teks embedding heading "
          "masih menyebut subheading ini sampai ingestion berikutnya. Cara permanen: hapus "
          "subheading itu dari PDF lalu jalankan ingestion.")
    if not confirm("Hapus subheading ini?", yes):
        print("Dibatalkan.")
        return

    if chunk_ids:
        chunk_col.delete(ids=chunk_ids)
    if alias_ids:
        heads.delete(ids=alias_ids)
    heads.delete(ids=[rid])
    if heading_row["ids"]:   # jaga metadata jumlah subheading pada heading
        remaining = len(siblings) - 1
        try:
            heads.update(ids=[hid], metadatas=[{"subheading_count": remaining}])
        except Exception as error:
            print(f"(metadata subheading_count tidak diperbarui: {error})")
    print(f"Terhapus: subheading '{label}', {len(alias_ids)} alias, "
          f"{len(chunk_ids)} chunk. Restart aplikasi agar semua proses memakai data terbaru.")


def cmd_activate(generation, yes):
    if generation not in generations():
        raise SystemExit(f"Generation '{generation}' tidak ada. Lihat: python manage_knowledge.py list")
    active = get_active_generation()
    if generation == active:
        raise SystemExit("Generation itu sudah aktif.")
    info = validate_generation(generation)
    print(f"Aktif sekarang : {active}")
    print(f"Akan diaktifkan: {generation} ({info['chunk_count']} chunk, "
          f"{info['heading_count']} record heading)")
    if not confirm("Aktifkan generation ini?", yes):
        print("Dibatalkan.")
        return
    activate_generation(generation)
    print(f"Aktif sekarang: {get_active_generation()}. Restart aplikasi agar memakai data ini.")


def cmd_delete_generation(generation, yes):
    if generation == get_active_generation():
        raise SystemExit("Generation aktif tidak boleh dihapus. Aktifkan generation lain dulu.")
    if not confirm(f"Hapus generation {generation} beserta seluruh isinya?", yes):
        print("Dibatalkan.")
        return
    print("Terhapus:", delete_generation(generation))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    yes = "--yes" in sys.argv
    generation = heading_id = None
    if "--heading" in sys.argv:
        heading_id = sys.argv[sys.argv.index("--heading") + 1]
        args = [a for a in args if a != heading_id]
    if "--generation" in sys.argv:
        generation = sys.argv[sys.argv.index("--generation") + 1]
        args = [a for a in args if a != generation]
    if not args:
        raise SystemExit(__doc__)
    command, rest = args[0], args[1:]
    if command == "list":
        cmd_list()
    elif command == "find" and rest:
        cmd_find(" ".join(rest))
    elif command == "show" and rest:
        cmd_show(rest[0], generation)
    elif command == "delete-chunk" and rest:
        cmd_delete_chunks(rest, yes)
    elif command == "subheadings":
        cmd_subheadings(" ".join(rest) or None)
    elif command == "delete-subheading" and rest:
        cmd_delete_subheading(rest[0], heading_id, yes)
    elif command == "activate" and rest:
        cmd_activate(rest[0], yes)
    elif command == "delete-generation" and rest:
        cmd_delete_generation(rest[0], yes)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()