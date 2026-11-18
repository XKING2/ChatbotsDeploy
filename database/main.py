import difflib
import json
import os
import sys
import time
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO
import math
import config
from config import CHROMA_DB_PATH, PDF_FOLDER

from database.parser import parse_pdf
from database.preprocessing import preprocess_blocks
from database.section_builder import build_sections
from database.chunker import chunk_sections
from database.heading_builder import (
    collect_headings,
    build_heading_embedding_text,
    build_subheading_embedding_text,
)
from database.alias_generator import (
    generate_heading_aliases_batch,
    generate_subheading_aliases_batch,
)
from database.heading_prompt import (
    build_heading_description,
    build_heading_examples,
)
from database.embedding import create_embeddings_batch
from database.hashing import compute_hash
from database.index_builder import build_knowledge_index
from database.chroma_db import (
    client,
    create_staging_generation,
    get_chunk_collection,
    get_heading_collection,
    get_active_generation,
    heading_alias_id,
    add_chunks,
    add_heading_item,
    validate_generation,
    activate_generation,
    delete_generation,
    discard_staging_generation,
    database_info,
)

# ============================================================
# KONFIGURASI
# ============================================================

CHUNK_BATCH_SIZE = 200
EMBEDDING_BATCH_SIZE = 20

DEFAULT_PDF_NAME = "TesT Update Data.pdf"
ALIAS_CACHE_PATH = os.path.join(CHROMA_DB_PATH, "alias_cache.json")
REPORT_DIR = os.path.join(CHROMA_DB_PATH, "ingestion_reports")

# Default: SEMUA item dicetak (juga yang tidak berubah).
# Jalankan dengan --ringkas agar yang tidak berubah hanya dihitung.
VERBOSE = "--ringkas" not in sys.argv

HASH_LEN = 12          # panjang hash yang dicetak (hash penuh ada di file laporan)
PREVIEW_LEN = 110      # panjang potongan teks pada tampilan perubahan

REPORT = []            # semua baris log; ditulis ke file laporan di akhir


# ============================================================
# LOG: cetak ke layar + simpan ke file laporan
# ============================================================

def log(text=""):
    print(text)
    REPORT.append(str(text))


def short_hash(value):
    return str(value)[:HASH_LEN] if value else "-"


def short_text(text, limit=PREVIEW_LEN):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def save_report(generation):
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORT_DIR, f"ingestion_{stamp}_{generation or 'gagal'}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(REPORT) + "\n")
        print(f"Laporan lengkap  : {path}")
    except OSError as error:
        print(f"Peringatan: laporan tidak tersimpan: {error}")


# ============================================================
# HELPER
# ============================================================

def normalize(text):
    return " ".join(str(text or "").casefold().split())


def make_source_identity(pdf_path):
    """source_id dari KNOWLEDGE_SOURCE_ID (config) atau lokasi PDF."""
    identity = getattr(config, "KNOWLEDGE_SOURCE_ID", None) or os.path.relpath(
        os.path.abspath(pdf_path), os.path.abspath(PDF_FOLDER)
    )
    source_id = "src_" + compute_hash(
        str(identity).replace("\\", "/").casefold()
    )[:16]
    return source_id, os.path.basename(pdf_path)


def clean_aliases(aliases):
    """Membersihkan alias dan membuang duplikat (tanpa beda kapitalisasi)."""
    if not isinstance(aliases, list):
        return []

    result, seen = [], set()
    for alias in aliases:
        alias = " ".join(alias.split()) if isinstance(alias, str) else ""
        if alias and alias.casefold() not in seen:
            seen.add(alias.casefold())
            result.append(alias)
    return result


def as_list(vector):
    return [float(x) for x in vector]


def validate_vectors(vectors, expected, label):
    if vectors is None:
        raise RuntimeError(
            f"Embedding {label} gagal dibuat. "
            "Periksa log error dari embedding.py."
        )

    if len(vectors) != expected:
        raise RuntimeError(
            f"Jumlah embedding {label} tidak sesuai: "
            f"seharusnya {expected}, didapat {len(vectors)}."
        )

    dimension = None

    for index, vector in enumerate(vectors):

        if vector is None or len(vector) == 0:
            raise RuntimeError(
                f"Embedding {label} kosong pada index {index}."
            )

        if not all(math.isfinite(float(value)) for value in vector):
            raise RuntimeError(
                f"Embedding {label} mengandung nilai tidak valid "
                f"pada index {index}."
            )

        if dimension is None:
            dimension = len(vector)

        elif len(vector) != dimension:
            raise RuntimeError(
                f"Dimensi embedding {label} tidak konsisten "
                f"pada index {index}."
            )


# ============================================================
# SNAPSHOT DATABASE AKTIF + PERBANDINGAN
# ============================================================

def snapshot(collection, label):
    """
    Isi database aktif: {id: {"hash", "meta", "doc"}}.
    None bila database aktif tidak bisa dibaca (dianggap belum ada).
    """
    rows = {}
    try:
        total = collection.count()
        for offset in range(0, total, 500):
            data = collection.get(limit=500, offset=offset,
                                  include=["documents", "metadatas"])
            for rid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                meta = meta or {}
                rows[rid] = {"hash": meta.get("content_hash"), "meta": meta,
                             "doc": doc or ""}
    except Exception as error:
        log(f"  Peringatan: database aktif ({label}) tidak dapat dibaca: {error}")
        return None
    return rows


def compare(items, old):
    """
    items: [{"id", "hash", "text"}]. Mengembalikan:
      status  {id: "BARU" | "BERUBAH" | "SAMA"}
      removed [id lama yang tidak ada lagi di data baru]
    """
    if old is None:
        return {item["id"]: "BARU" for item in items}, []

    status = {}
    for item in items:
        previous = old.get(item["id"])
        if previous is None:
            status[item["id"]] = "BARU"
        elif previous["hash"] != item["hash"]:
            status[item["id"]] = "BERUBAH"
        else:
            status[item["id"]] = "SAMA"

    wanted = {item["id"] for item in items}
    return status, sorted(rid for rid in old if rid not in wanted)


def describe_change(old_text, new_text, max_parts=3):
    """Potongan teks yang dihapus (-) dan ditambahkan (+) antara versi lama dan baru."""
    old_words, new_words = str(old_text).split(), str(new_text).split()
    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    lines = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before, after = " ".join(old_words[i1:i2]), " ".join(new_words[j1:j2])
        if before:
            lines.append(f"- dihapus  : \"{short_text(before)}\"")
        if after:
            lines.append(f"+ ditambah : \"{short_text(after)}\"")
        if len(lines) >= max_parts * 2:
            lines.append("… (perubahan lain tidak ditampilkan)")
            break
    if not lines:
        lines.append("(isi teks sama; hanya hash/metadata yang berbeda)")
    return lines


def count_status(status):
    result = {"BARU": 0, "BERUBAH": 0, "SAMA": 0}
    for value in status.values():
        result[value] += 1
    return result


# ============================================================
# EMBEDDING (memakai ulang vektor yang hash-nya sama)
# ============================================================

def old_vectors(collection, wanted):
    """
    Vektor dari database aktif untuk id yang content_hash-nya sama.
    wanted: {id: content_hash}.
    """
    found, ids = {}, list(wanted)
    try:
        for i in range(0, len(ids), 500):
            data = collection.get(ids=ids[i:i + 500],
                                  include=["metadatas", "embeddings"])
            embeddings = data["embeddings"]
            for rid, meta, vec in zip(data["ids"], data["metadatas"],
                                      [] if embeddings is None else embeddings):
                if (meta or {}).get("content_hash") == wanted[rid] and len(vec):
                    found[rid] = as_list(vec)
    except Exception as error:
        log(f"Peringatan: vektor lama tidak dapat dipakai ulang: {error}")
        return {}
    return found


def embed_items(items, active, label, force, status):
    """
    items: [{"id", "text", "hash"}]. Mengembalikan vektor sejajar dengan items.
    Hanya item baru/berubah yang dikirim ke API embedding.
    status: {id: BARU|BERUBAH|SAMA} dari perbandingan sebelumnya (untuk alasan).
    """
    reuse = {} if force else old_vectors(
        active, {item["id"]: item["hash"] for item in items}
    )
    vectors = [reuse.get(item["id"]) for item in items]
    todo = [n for n, v in enumerate(vectors) if v is None]

    log(f"{label}: {len(items)} data | vektor lama dipakai ulang "
        f"{len(items) - len(todo)} | embedding baru {len(todo)}")

    def reason(item):
        if force:
            return "--force (semua dibuat ulang)"
        state = status.get(item["id"])
        if state == "BARU":
            return "id belum ada di database aktif"
        if state == "BERUBAH":
            return "hash berubah → teks diubah"
        return "vektor lama tidak terbaca"

    if todo and VERBOSE:
        log(f"  Item yang di-embedding ulang ({label}):")
        for n in todo:
            item = items[n]
            log(f"    - {item['id']} | hash {short_hash(item['hash'])} | {reason(item)}")

    for start in range(0, len(todo), CHUNK_BATCH_SIZE):
        part = todo[start:start + CHUNK_BATCH_SIZE]
        new = create_embeddings_batch(
            [items[n]["text"] for n in part],
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        validate_vectors(new, len(part), label)
        for n, vector in zip(part, new):
            vectors[n] = as_list(vector)
        log(f"  embedding {label}: {start + len(part)}/{len(todo)}")

    return vectors


# ============================================================
# ALIAS (generate hanya untuk heading/subheading baru)
# ============================================================

def load_alias_cache():
    try:
        with open(ALIAS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_alias_cache(cache):
    tmp = ALIAS_CACHE_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, ALIAS_CACHE_PATH)
    except OSError as error:
        log(f"Peringatan: cache alias tidak tersimpan: {error}")


def read_existing_aliases():
    """Alias di database aktif, dikelompokkan per heading (satu query)."""
    data = get_heading_collection().get(
        where={"type": "alias"}, include=["metadatas"]
    )

    found = {}
    for meta in data["metadatas"]:
        meta = meta or {}
        alias, scope = meta.get("alias"), meta.get("alias_scope")
        if not alias or not meta.get("heading_id"):
            continue

        slot = found.setdefault(
            meta["heading_id"], {"heading": [], "by_id": {}, "by_name": {}}
        )
        if scope == "heading":
            slot["heading"].append(alias)
        elif scope == "subheading":
            slot["by_id"].setdefault(
                str(meta.get("subheading_id")), []).append(alias)
            slot["by_name"].setdefault(
                normalize(meta.get("subheading")), []).append(alias)
    return found


def prepare_aliases(headings):
    """
    Urutan sumber alias: database aktif -> cache -> generate baru.
    Alias yang sudah ada tidak pernah di-generate ulang, dan update isi data
    saja tidak memicu generate. Gagal generate menghentikan proses sebelum
    ada embedding yang dibuat.
    Mengembalikan ({heading_id: [alias]}, {"heading_id::subheading_id": [alias]}).
    """
    existing = read_existing_aliases()
    cache = load_alias_cache()

    heading_aliases, subheading_aliases = {}, {}
    pending_headings, pending_subheadings = [], []
    source = {}  # kunci -> asal alias, untuk laporan

    for item in headings:
        hid, heading = item["heading_id"], item["heading"]
        slot = existing.get(hid, {})

        hkey = f"{hid}::{normalize(heading)}"
        from_db = clean_aliases(slot.get("heading"))
        from_cache = clean_aliases(cache.get(hkey)) if not from_db else []
        aliases = from_db or from_cache
        heading_aliases[hid] = aliases
        source[hid] = ("database aktif" if from_db else
                       "cache alias" if from_cache else "BELUM ADA → di-generate")
        if not aliases:
            pending_headings.append(item)

        for sub in item["subheadings"].values():
            sid, name = str(sub["subheading_id"]), sub["subheading"]
            key = f"{hid}::{sid}"
            db_list = clean_aliases(
                slot.get("by_id", {}).get(sid)
                or slot.get("by_name", {}).get(normalize(name))
            )
            cache_list = (clean_aliases(cache.get(f"{key}::{normalize(name)}"))
                          if not db_list else [])
            aliases = db_list or cache_list
            subheading_aliases[key] = aliases
            source[key] = ("database aktif" if db_list else
                           "cache alias" if cache_list else "BELUM ADA → di-generate")
            if not aliases:
                pending_subheadings.append({
                    "heading_id": hid, "heading": heading,
                    "subheading_id": sub["subheading_id"], "subheading": name,
                })

    log(f"Alias heading perlu generate    : {len(pending_headings)}")
    log(f"Alias subheading perlu generate : {len(pending_subheadings)}")

    if pending_headings:
        log("  Heading yang akan di-generate aliasnya:")
        for item in pending_headings:
            log(f"    - [{item['heading_id']}] {item['heading']}")
    if pending_subheadings:
        log("  Subheading yang akan di-generate aliasnya:")
        for item in pending_subheadings:
            log(f"    - [{item['heading_id']}::{item['subheading_id']}] "
                f"{item['heading']} > {item['subheading']}")

    if pending_headings:
        generated = generate_heading_aliases_batch(
            [item["heading"] for item in pending_headings]
        ) or {}
        for item in pending_headings:
            if item["heading"] not in generated:
                raise RuntimeError(
                    f"Generate alias heading gagal: {item['heading']}")
            aliases = clean_aliases(generated[item["heading"]])
            heading_aliases[item["heading_id"]] = aliases
            source[item["heading_id"]] = "BARU di-generate"
            if aliases:
                cache[f"{item['heading_id']}::{normalize(item['heading'])}"] = aliases

    if pending_subheadings:
        generated = generate_subheading_aliases_batch(pending_subheadings) or {}
        for item in pending_subheadings:
            key = f"{item['heading_id']}::{item['subheading_id']}"
            if key not in generated:
                raise RuntimeError(
                    f"Generate alias subheading gagal: {item['subheading']}")
            aliases = clean_aliases(generated[key])
            subheading_aliases[key] = aliases
            source[key] = "BARU di-generate"
            if aliases:
                cache[f"{key}::{normalize(item['subheading'])}"] = aliases

    if pending_headings or pending_subheadings:
        save_alias_cache(cache)  # tidak hilang bila proses gagal di tengah

    # Ringkasan alias: satu baris per heading; teks alias hanya untuk yang baru dibuat
    log("\n  Alias per heading (isi alias hanya ditulis bila baru di-generate):")
    for item in headings:
        hid = item["heading_id"]
        subs = list(item["subheadings"].values())
        sub_total = sum(len(subheading_aliases.get(f"{hid}::{x['subheading_id']}", []))
                        for x in subs)
        log(f"  - {item['heading']} [{hid}]")
        log(f"      alias heading {len(heading_aliases.get(hid, []))} ({source[hid]}) | "
            f"{len(subs)} subheading dengan total {sub_total} alias")
        if source[hid].startswith("BARU"):
            log("      alias baru: " + " | ".join(heading_aliases[hid]))
        for sub in subs:
            key = f"{hid}::{sub['subheading_id']}"
            if source[key].startswith("BARU"):
                log(f"      subheading baru [{sub['subheading_id']}] {sub['subheading']}: "
                    + " | ".join(subheading_aliases.get(key, [])))

    return heading_aliases, subheading_aliases


# ============================================================
# TAMPILAN SECTION & CHUNK (struktur pohon + hash)
# ============================================================

def print_sections(sections):
    log("\n  SECTION hasil parsing PDF (urut sesuai dokumen):")
    log("  (hash isi = compute_hash dari teks section, hanya untuk pengenal)")
    for section in sections:
        content = section.get("content") or section.get("document") or ""
        heading = section.get("heading") or section.get("entity") or "?"
        sub = section.get("subheading") or section.get("subtopic") or "?"
        hid = section.get("heading_id") or section.get("entity_id") or "?"
        sid = section.get("subheading_id") or section.get("subtopic_id") or "?"
        log(f"  - Section {section.get('section_id')} | hal "
            f"{section.get('page_start')}-{section.get('page_end')} | "
            f"{len(content)} karakter | hash isi {short_hash(compute_hash(content))}")
        log(f"      Heading    : [{hid}] {heading}")
        log(f"      Subheading : [{sid}] {sub} "
            f"(kemunculan ke-{section.get('subheading_occurrence', 1)})")


def chunk_line(chunk, state):
    return (f"[{state:<7}] chunk {chunk.get('chunk_index')}/{chunk.get('total_chunks')} | "
            f"id={chunk['content_chunk_id']} | hash={short_hash(chunk['content_hash'])} | "
            f"hal {chunk.get('page_start')}-{chunk.get('page_end')} | "
            f"{len(chunk.get('retrieval_text') or '')} karakter")


def tally(statuses):
    result = {"BARU": 0, "BERUBAH": 0, "SAMA": 0}
    for value in statuses:
        result[value] += 1
    return result


def tally_text(statuses):
    c = tally(statuses)
    parts = [f"{c['SAMA']} sama"]
    if c["BERUBAH"]:
        parts.append(f"{c['BERUBAH']} BERUBAH")
    if c["BARU"]:
        parts.append(f"{c['BARU']} BARU")
    return f"{len(statuses)} ({', '.join(parts)})"


def print_plan(headings, jobs, chunks, chunk_status, head_status,
               old_chunks, old_heads, force, chunk_removed=()):
    """
    Satu pohon: HEADING -> SUBHEADING -> chunk. Alias yang tidak berubah hanya
    diringkas jumlahnya; alias/chunk yang BARU atau BERUBAH ditulis satu per satu.
    Mengembalikan jumlah status per jenis record heading untuk tabel ringkasan.
    """
    bar = "  " + "=" * 74
    thin = "  " + "-" * 74
    jobs_by = {"heading": {}, "subheading": {}, "h_alias": {}, "s_alias": {}}
    for job in jobs:
        kind = job["type"]
        if kind == "heading":
            jobs_by["heading"][job["id"]] = job
        elif kind == "subheading":
            jobs_by["subheading"][(job["heading_id"], job["id"])] = job
        elif job["extra"].get("alias_scope") == "subheading":
            jobs_by["s_alias"].setdefault(
                (job["heading_id"], job["extra"].get("subheading_id")), []).append(job)
        else:
            jobs_by["h_alias"].setdefault(job["heading_id"], []).append(job)

    chunks_by = {}
    for chunk in chunks:
        chunks_by.setdefault(
            (chunk.get("heading_id"), chunk.get("subheading_id")), []).append(chunk)

    removed_by = {}
    for rid in chunk_removed:
        meta = (old_chunks or {}).get(rid, {}).get("meta", {})
        removed_by.setdefault((meta.get("heading_id"), meta.get("subheading_id")), []).append(rid)

    def changed_job(job):
        return head_status[job["id"]] != "SAMA"

    def sub_changed(hid, sid, sub_job):
        return (changed_job(sub_job) or bool(removed_by.get((hid, sid)))
                or any(changed_job(j) for j in jobs_by["s_alias"].get((hid, sid), []))
                or any(chunk_status[c["content_chunk_id"]] != "SAMA"
                       for c in chunks_by.get((hid, sid), [])))

    def print_alias_changes(items, indent):
        for job in items:
            state = head_status[job["id"]]
            if state == "SAMA":
                continue
            log(f"{indent}{'+' if state == 'BARU' else '~'} alias {state:<7} "
                f"\"{short_text(job['name'], 70)}\"")
            log(f"{indent}    id={job['id']} | hash={short_hash(job['hash'])}")
            if state == "BERUBAH" and old_heads:
                before = old_heads[job["id"]]
                log(f"{indent}    hash lama {short_hash(before['hash'])} → "
                    f"baru {short_hash(job['hash'])}")

    log("\n  RENCANA PER HEADING → SUBHEADING → CHUNK")
    log("  (dibandingkan dengan database aktif; alias yang sama hanya dihitung)")
    unchanged_headings = []

    for item in headings:
        hid = item["heading_id"]
        h_job = jobs_by["heading"][hid]
        subs = list(item["subheadings"].values())
        h_alias = jobs_by["h_alias"].get(hid, [])

        any_change = (changed_job(h_job) or any(changed_job(j) for j in h_alias)
                      or any(sub_changed(hid, str(sb["subheading_id"]),
                                         jobs_by["subheading"][(hid, sb["subheading_id"])])
                             for sb in subs))
        if not any_change and not VERBOSE:
            unchanged_headings.append(item["heading"])
            continue

        log("")
        log(bar)
        log(f"  HEADING : {item['heading']}")
        log(f"  id      : {hid} | hash: {short_hash(h_job['hash'])} | "
            f"status: {head_status[hid]}")
        log(f"  alias   : {tally_text([head_status[j['id']] for j in h_alias])}")
        if head_status[hid] == "BERUBAH" and old_heads:
            before = old_heads[hid]
            log(f"  hash lama {short_hash(before['hash'])} → baru {short_hash(h_job['hash'])}")
            for line in describe_change(before["doc"], h_job["text"]):
                log("    " + line)
        print_alias_changes(h_alias, "  ")
        log(f"  subheading: {len(subs)} | "
            + ("ADA PERUBAHAN" if any_change else "tidak ada perubahan"))

        for sb in subs:
            sid = sb["subheading_id"]
            s_job = jobs_by["subheading"][(hid, sid)]
            s_alias = jobs_by["s_alias"].get((hid, sid), [])
            members = chunks_by.get((hid, sid), [])
            changed = sub_changed(hid, sid, s_job)
            if not VERBOSE and not changed:
                continue

            log(thin)
            log(f"    SUBHEADING : {sb['subheading']}")
            log(f"    id         : {sid} | hash: {short_hash(s_job['hash'])} | "
                f"status: {head_status[s_job['id']]}")
            log(f"    alias      : {tally_text([head_status[j['id']] for j in s_alias])}")
            if head_status[s_job["id"]] == "BERUBAH" and old_heads:
                before = old_heads[s_job["id"]]
                log(f"    hash lama {short_hash(before['hash'])} → baru {short_hash(s_job['hash'])}")
                for line in describe_change(before["doc"], s_job["text"]):
                    log("      " + line)
            print_alias_changes(s_alias, "    ")

            gone = removed_by.get((hid, sid), [])
            log(f"    chunk      : {tally_text([chunk_status[c['content_chunk_id']] for c in members])}"
                + (f" | {len(gone)} DIHAPUS" if gone else ""))
            for rid in gone:
                log(f"      [HAPUS  ] id={rid}")
                log(f"                 hash={short_hash(old_chunks[rid]['hash'])} | "
                    f"\"{short_text(old_chunks[rid]['doc'], 80)}\"")
            for chunk in members:
                state = chunk_status[chunk["content_chunk_id"]]
                if state == "SAMA" and not VERBOSE:
                    continue
                log(f"      [{state:<7}] chunk {chunk.get('chunk_index')}/{chunk.get('total_chunks')}"
                    f" | hal {chunk.get('page_start')}-{chunk.get('page_end')}"
                    f" | {len(chunk.get('retrieval_text') or '')} karakter")
                log(f"                 id={chunk['content_chunk_id']}")
                log(f"                 hash={short_hash(chunk['content_hash'])}"
                    + (f" (lama {short_hash(old_chunks[chunk['content_chunk_id']]['hash'])})"
                       if state == "BERUBAH" and old_chunks else ""))
                if state == "BERUBAH" and old_chunks:
                    for line in describe_change(
                            old_chunks[chunk["content_chunk_id"]]["doc"],
                            chunk["retrieval_text"]):
                        log("                 " + line)
                elif state == "BARU":
                    log(f"                 isi: \"{short_text(chunk['retrieval_text'])}\"")

    known = {(c.get("heading_id"), c.get("subheading_id")) for c in chunks}
    orphan = [c for c in chunks if c.get("heading_id") not in
              {h["heading_id"] for h in headings}]
    if orphan:
        log("\n  CHUNK tanpa heading yang cocok:")
        for chunk in orphan:
            log(f"    [{chunk_status[chunk['content_chunk_id']]}] {chunk['content_chunk_id']}")
    log("")
    log(bar)
    if unchanged_headings:
        log(f"  Heading tanpa perubahan ({len(unchanged_headings)}): "
            + "; ".join(unchanged_headings))

    counts = {}
    for job in jobs:
        kind = job["type"]
        counts.setdefault(kind, {"BARU": 0, "BERUBAH": 0, "SAMA": 0})[head_status[job["id"]]] += 1
    return counts


def print_removed(removed, old, label):
    if not removed:
        log(f"  {label} dihapus/tidak dibawa: tidak ada")
        return
    log(f"  {label} yang ADA di database aktif tapi TIDAK ADA lagi di data baru "
        f"({len(removed)}):")
    log("  (tidak dibawa ke generation baru; generation lama tetap utuh sampai dibersihkan)")
    for rid in removed:
        meta = old[rid]["meta"]
        log(f"    - HAPUS id={rid} | hash={short_hash(old[rid]['hash'])} | "
            f"{meta.get('heading') or meta.get('alias') or ''}"
            f"{' > ' + str(meta.get('subheading')) if meta.get('subheading') else ''}"
            f" | \"{short_text(old[rid]['doc'], 70)}\"")


# ============================================================
# CHUNK
# ============================================================

def process_chunks(chunks, source_id, source_file, collection, active, force, status):
    for chunk in chunks:
        chunk["source_id"] = source_id
        chunk["source_file"] = source_file
        if not chunk.get("retrieval_text"):
            raise ValueError("Retrieval text chunk kosong.")

    vectors = embed_items(
        [{"id": c["content_chunk_id"], "text": c["retrieval_text"],
          "hash": c["content_hash"]} for c in chunks],
        active, "chunk", force, status,
    )
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    saved = add_chunks(chunks, target_collection=collection)
    if saved != len(chunks):
        raise RuntimeError("Jumlah chunk tersimpan tidak sesuai.")
    log(f"  chunk tersimpan ke staging: {saved}")
    return saved


# ============================================================
# HEADING
# ============================================================

def subheading_alias_id(heading_id, subheading_id, alias):
    alias_hash = compute_hash(alias.strip().casefold())[:16]
    return f"{heading_id}::subheading::{subheading_id}::alias::{alias_hash}"


def build_heading_jobs(headings, heading_aliases, subheading_aliases):
    """Semua record heading, subheading, dan alias yang akan disimpan."""
    jobs = []

    def add(kind, record_id, heading_id, text, name, **extra):
        jobs.append({"type": kind, "id": record_id, "heading_id": heading_id,
                     "text": text, "hash": compute_hash(text),
                     "name": name, "extra": extra})

    for item in headings:
        hid, heading = item["heading_id"], item["heading"]
        aliases = heading_aliases.get(hid, [])
        subs = list(item["subheadings"].values())

        add("heading", hid, hid,
            build_heading_embedding_text(
                heading=heading, aliases=aliases,
                subheadings=[s["subheading"] for s in subs]),
            heading, subheading_count=len(subs), alias_count=len(aliases))

        for sub in subs:
            sid, name = sub["subheading_id"], sub["subheading"]
            description = build_heading_description(name, heading)
            examples = build_heading_examples(name, heading)

            add("subheading", sid, hid,
                build_subheading_embedding_text(
                    heading=heading, subheading=name,
                    description=description, examples=examples),
                name, subheading_id=sid, heading=heading,
                description=description, examples=" | ".join(examples))

            for alias in subheading_aliases.get(f"{hid}::{sid}", []):
                add("alias", subheading_alias_id(hid, sid, alias), hid,
                    f"Layanan:\n{heading}\n\nSubheading:\n{name}"
                    f"\n\nAlias subheading:\n{alias}",
                    alias, alias_scope="subheading", heading=heading,
                    subheading_id=sid, subheading=name)

        for alias in aliases:
            add("alias", heading_alias_id(hid, alias), hid,
                f"Layanan:\n{heading}\n\nNama lain:\n{alias}",
                alias, alias_scope="heading", heading=heading)

    return jobs


def process_headings(jobs, collection, active, force, status):
    vectors = embed_items(jobs, active, "heading", force, status)

    for job, vector in zip(jobs, vectors):
        add_heading_item(
            item_type=job["type"], item_id=job["id"],
            heading_id=job["heading_id"], text=job["text"],
            embedding=vector, content_hash=job["hash"],
            target_collection=collection, name=job["name"], **job["extra"],
        )
    log(f"  record heading tersimpan ke staging: {len(jobs)}")
    return len(jobs)


# ============================================================
# VALIDASI & PEMBERSIHAN
# ============================================================

def validate_collection(collection, expected, label):
    """Memeriksa jumlah, kelengkapan metadata, dan dimensi embedding."""
    actual = collection.count()
    if actual != expected:
        raise RuntimeError(
            f"Jumlah record {label} tidak sesuai: "
            f"seharusnya {expected}, didapat {actual}.")

    need = ["content_hash", "heading_id"]
    if label == "chunk":
        need += ["subheading_id", "subheading", "content_chunk_id"]

    dimension = None
    for offset in range(0, expected, 500):
        data = collection.get(limit=500, offset=offset,
                              include=["documents", "metadatas", "embeddings"])
        ids, embeddings = data["ids"], data["embeddings"]

        if embeddings is None or not (
            len(ids) == len(data["documents"])
            == len(data["metadatas"]) == len(embeddings)
        ):
            raise RuntimeError(f"Data {label} tidak lengkap.")

        for rid, doc, meta, vec in zip(
            ids, data["documents"], data["metadatas"], embeddings
        ):
            if not doc or not meta or vec is None or any(
                not meta.get(f) for f in need
            ):
                raise RuntimeError(f"Record {label} tidak lengkap: {rid}")

            dimension = dimension or len(vec)
            if len(vec) != dimension:
                raise RuntimeError(f"Dimensi embedding berbeda: {rid}")

    return dimension


def cleanup_old_generations(keep):
    """Hapus generation lama; yang aktif dan satu sebelumnya (rollback) disimpan."""
    names = {c if isinstance(c, str) else c.name
             for c in client.list_collections()}
    generations = {n.rpartition("__g_")[2] for n in names if "__g_" in n}

    log(f"  Generation tersedia: {sorted(generations)} | dipertahankan: "
        f"{sorted(g for g in keep if g)}")
    for generation in sorted(generations - set(keep)):
        try:
            delete_generation(generation)
            log(f"  Generation lama dihapus: {generation}")
        except Exception as error:
            log(f"  Peringatan: gagal menghapus generation {generation}: {error}")


def update_knowledge_index(sections, chunks):
    try:
        with redirect_stdout(StringIO()):
            build_knowledge_index(sections, chunks)
        log("  Knowledge index berhasil diperbarui.")
    except Exception as error:
        log(f"  Peringatan: knowledge index gagal diperbarui: {error}")


# ============================================================
# ORCHESTRATOR
# ============================================================

TOTAL_STEPS = 8


def step(number, text):
    log(f"\n{'-' * 60}\n[{number}/{TOTAL_STEPS}] {text}\n{'-' * 60}")


def print_summary(chunk_counts, chunk_removed, head_counts, head_removed):
    log("\n  RINGKASAN PERUBAHAN TERHADAP DATABASE AKTIF")
    log(f"  {'jenis':<22}{'baru':>7}{'berubah':>9}{'sama':>7}{'dihapus':>9}")
    log(f"  {'chunk':<22}{chunk_counts['BARU']:>7}{chunk_counts['BERUBAH']:>9}"
        f"{chunk_counts['SAMA']:>7}{chunk_removed:>9}")
    for kind in ("heading", "subheading", "alias"):
        c = head_counts.get(kind, {"BARU": 0, "BERUBAH": 0, "SAMA": 0})
        removed = head_removed.get(kind, 0)
        log(f"  {kind:<22}{c['BARU']:>7}{c['BERUBAH']:>9}{c['SAMA']:>7}{removed:>9}")
    changed = (chunk_counts["BARU"] + chunk_counts["BERUBAH"] + chunk_removed
               + sum(c["BARU"] + c["BERUBAH"] for c in head_counts.values())
               + sum(head_removed.values()))
    log("  → " + ("TIDAK ADA PERUBAHAN: seluruh data sama dengan database aktif "
                  "(semua vektor dipakai ulang)." if not changed else
                  "Ada perubahan; hanya item BARU/BERUBAH yang di-embedding."))


def main():
    started = time.perf_counter()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv  # abaikan vektor lama (mis. model embedding baru)

    generation, activated = None, False

    try:
        pdf_path = args[0] if args else os.path.join(PDF_FOLDER, DEFAULT_PDF_NAME)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF tidak ditemukan: {pdf_path}")

        source_id, source_file = make_source_identity(pdf_path)
        previous = get_active_generation()

        log("=" * 60 + "\nKNOWLEDGE BASE INGESTION\n" + "=" * 60)
        log(f"File       : {source_file}\nSource ID  : {source_id}")
        log(f"Force      : {force}\nMode cetak : "
            f"{'rinci (semua item)' if VERBOSE else 'ringkas (hanya yang berubah)'}")
        log(f"Generation aktif sekarang : {previous}")
        log(f"Database   : {database_info()}")
        log("\nCara membaca laporan ini:")
        log("  id    = kunci unik record di ChromaDB. Id sama + hash sama = tidak berubah.")
        log("  hash  = sidik isi (content_hash). Hash beda untuk id yang sama = isi berubah.")
        log("  BARU    : id belum ada di database aktif  → embedding dibuat")
        log("  BERUBAH : id ada, hash berbeda            → embedding dibuat ulang")
        log("  SAMA    : id dan hash sama                → vektor lama dipakai ulang")
        log("  HAPUS   : ada di database aktif, tidak ada di data baru → tidak dibawa")

        step(1, "Parsing dan preprocessing PDF...")
        parsed = parse_pdf(pdf_path)
        if not parsed:
            raise RuntimeError("Tidak ada teks yang berhasil diekstrak dari PDF.")
        log(f"  Blok teks hasil parsing    : {len(parsed)}")
        parsed = preprocess_blocks(parsed)
        if not parsed:
            raise RuntimeError("Tidak ada teks setelah preprocessing.")
        log(f"  Blok teks setelah preprocessing : {len(parsed)}")

        step(2, "Membentuk section, chunk, heading, dan subheading...")
        sections = build_sections(parsed)
        if not sections:
            raise RuntimeError(
                "Section tidak berhasil dibentuk. "
                "Periksa marker Heading, Sub Heading, dan Content.")
        for section in sections:
            section["source_id"] = source_id
            section["source_file"] = source_file

        chunks = chunk_sections(sections)
        headings = collect_headings(sections)
        if not chunks:
            raise RuntimeError("Chunk tidak berhasil dibentuk.")
        if not headings:
            raise RuntimeError("Heading tidak berhasil dikumpulkan.")

        log(f"  Section {len(sections)} | Chunk {len(chunks)} | "
            f"Heading {len(headings)} | Subheading "
            f"{sum(len(h['subheadings']) for h in headings)}")
        for chunk in chunks:           # source_id dibutuhkan untuk metadata chunk
            chunk["source_id"] = source_id
            chunk["source_file"] = source_file
        if VERBOSE:
            print_sections(sections)

        # Alias dulu: kalau gagal, belum ada staging dan embedding terbuang.
        step(3, "Menyiapkan alias...")
        heading_aliases, subheading_aliases = prepare_aliases(headings)
        jobs = build_heading_jobs(headings, heading_aliases, subheading_aliases)

        step(4, "Membandingkan data baru dengan database aktif...")
        chunk_items = [{"id": c["content_chunk_id"], "hash": c["content_hash"],
                        "text": c["retrieval_text"]} for c in chunks]
        old_chunks = snapshot(get_chunk_collection(), "chunk")
        old_heads = snapshot(get_heading_collection(), "heading")
        if old_chunks is None or old_heads is None:
            log("  Database aktif kosong/tidak terbaca → semua item dianggap BARU.")
        else:
            log(f"  Database aktif: {len(old_chunks)} chunk, {len(old_heads)} record heading")

        chunk_status, chunk_removed = compare(chunk_items, old_chunks)
        head_status, head_removed = compare(jobs, old_heads)

        head_counts = print_plan(headings, jobs, chunks, chunk_status, head_status,
                                 old_chunks, old_heads, force, chunk_removed)
        print_removed(chunk_removed, old_chunks or {}, "Chunk")
        print_removed(head_removed, old_heads or {}, "Record heading/subheading/alias")

        removed_by_type = {}
        for rid in head_removed:
            kind = (old_heads[rid]["meta"] or {}).get("type", "heading")
            removed_by_type[kind] = removed_by_type.get(kind, 0) + 1
        print_summary(count_status(chunk_status), len(chunk_removed),
                      head_counts, removed_by_type)

        step(5, "Membuat database staging...")
        staging = create_staging_generation()
        generation = staging["generation"]
        log(f"  Generation baru (staging): {generation}")
        log(f"  Generation aktif tetap   : {previous} (tidak disentuh sampai validasi lulus)")

        step(6, "Embedding dan penyimpanan chunk...")
        chunk_count = process_chunks(
            chunks, source_id, source_file, staging["chunk_collection"],
            get_chunk_collection(), force, chunk_status)

        step(7, "Embedding dan penyimpanan heading, subheading, alias...")
        heading_count = process_headings(
            jobs, staging["heading_collection"], get_heading_collection(),
            force, head_status)

        step(8, "Validasi dan aktivasi...")
        validate_generation(generation)
        chunk_dim = validate_collection(
            staging["chunk_collection"], chunk_count, "chunk")
        heading_dim = validate_collection(
            staging["heading_collection"], heading_count, "heading")
        if chunk_dim != heading_dim:
            raise RuntimeError("Dimensi embedding chunk dan heading berbeda.")
        log(f"  Validasi lulus: {chunk_count} chunk, {heading_count} record heading, "
            f"dimensi embedding {chunk_dim}")

        activate_generation(generation)
        activated = True
        log(f"  Database berhasil diaktifkan: {previous} → {generation}")

        cleanup_old_generations(keep={generation, previous})
        update_knowledge_index(sections, chunks)

        log("\n" + "=" * 60 + "\nINGESTION BERHASIL\n" + "=" * 60)
        log(f"Generation      : {generation}")
        log(f"Chunk tersimpan : {chunk_count}")
        log(f"Heading records : {heading_count}")
        log(f"Waktu proses    : {time.perf_counter() - started:.2f} detik")
        log(f"Database        : {database_info()}")
        save_report(generation)

    except (Exception, KeyboardInterrupt) as error:
        log("\nINGESTION DIBATALKAN." if isinstance(error, KeyboardInterrupt)
            else f"\nINGESTION GAGAL: {error}")

        if generation and not activated:
            try:
                discard_staging_generation(generation)
                log("Database staging dibersihkan.")
            except Exception as cleanup_error:
                log(f"Gagal membersihkan staging: {cleanup_error}")

        log("Database aktif sebelumnya tidak diganti.")
        save_report(generation)
        raise


if __name__ == "__main__":
    main()