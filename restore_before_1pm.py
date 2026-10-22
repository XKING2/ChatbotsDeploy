from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse, unquote
import json
import os
import shutil

PROJECT = Path(r"C:\Users\IT\Documents\Chatbot")
HISTORY = Path(os.environ["APPDATA"]) / "Code" / "User" / "History"
BACKUP = Path(r"C:\Users\IT\Documents\Chatbot-AfterFilter-Backup")
WITA = timezone(timedelta(hours=8))
CUTOFF = datetime(2026, 10, 21, 13, 0, 0, tzinfo=WITA)

# Jangan memulihkan file ini dari snapshot lama karena mungkin berisi API key bocor.
SKIP_NAMES = {"modelist.py"}


def path_from_uri(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path.replace("/", "\\"))


def backup_candidate(relative: Path) -> Path | None:
    candidates = [
        BACKUP / relative,
        BACKUP / "Chatbot" / relative,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    if not PROJECT.is_dir():
        print(f"Folder proyek tidak ditemukan: {PROJECT}")
        return
    if not HISTORY.is_dir():
        print(f"Folder Local History tidak ditemukan: {HISTORY}")
        return

    prefix = str(PROJECT).rstrip("\\/") + os.sep
    restored_today = 0
    restored_backup = 0
    skipped = 0
    errors = 0

    print("Target: versi 21 Oktober 2026 sebelum 13.00 WITA")
    print("Memproses file proyek saja; tidak mengambil snapshot dari hari sebelumnya.\n")

    # Kelompokkan riwayat berdasarkan file sumber.
    records = {}
    for meta_file in HISTORY.rglob("entries.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            uri = meta.get("resource")
            if not uri:
                continue

            source = path_from_uri(uri)
            source_text = str(source)
            if not source_text.lower().startswith(prefix.lower()):
                continue
            if source.name.lower() in SKIP_NAMES:
                continue

            relative = Path(source_text[len(prefix):])
            candidates = []
            for entry in meta.get("entries", []):
                try:
                    saved = datetime.fromtimestamp(
                        int(entry["timestamp"]) / 1000,
                        tz=timezone.utc
                    ).astimezone(WITA)
                    # WAJIB tanggal hari ini dan sebelum jam 13.00.
                    if saved.date().isoformat() == "2026-10-21" and saved < CUTOFF:
                        candidates.append((saved, entry))
                except (KeyError, TypeError, ValueError, OSError):
                    continue

            if candidates:
                best = max(candidates, key=lambda item: item[0])
                previous = records.get(str(source).lower())
                if previous is None or best[0] > previous[0]:
                    records[str(source).lower()] = (
                        best[0], best[1], meta_file.parent, source, relative
                    )
        except Exception as exc:
            print(f"Metadata dilewati: {meta_file} ({exc})")
            errors += 1

    # Pulihkan hanya versi Local History dari tanggal dan waktu yang diminta.
    for saved, entry, folder, source, relative in records.values():
        snapshot = folder / entry.get("id", "")
        try:
            if not snapshot.is_file():
                print(f"SNAPSHOT TIDAK ADA: {source}")
                errors += 1
                continue
            shutil.copy2(snapshot, source)
            print(f"RESTORED HARI INI [{saved:%H:%M:%S}] {source}")
            restored_today += 1
        except OSError as exc:
            print(f"GAGAL: {source}: {exc}")
            errors += 1

    # Untuk file lain yang mungkin sempat tertimpa oleh script sebelumnya,
    # gunakan folder backup yang sudah ada (jika ditemukan), bukan riwayat lama.
    # Ini hanya memakai backup yang sudah ada; tidak membuat backup baru.
    for current in PROJECT.rglob("*"):
        if not current.is_file():
            continue
        if ".git" in current.parts or "venv" in current.parts or ".venv" in current.parts:
            continue
        if current.name.lower() in SKIP_NAMES:
            continue
        key = str(current).lower()
        if key in records:
            continue
        relative = current.relative_to(PROJECT)
        old_copy = backup_candidate(relative)
        if old_copy:
            try:
                shutil.copy2(old_copy, current)
                print(f"RESTORED DARI BACKUP YANG SUDAH ADA {current}")
                restored_backup += 1
            except OSError as exc:
                print(f"GAGAL BACKUP: {current}: {exc}")
                errors += 1

    print("\n========== HASIL ==========")
    print(f"Versi hari ini sebelum jam 13.00: {restored_today}")
    print(f"File dikembalikan dari backup lama yang sudah ada: {restored_backup}")
    print(f"Kesalahan: {errors}")
    print("File modelist.py sengaja dilewati demi mencegah pemulihan API key lama.")
    print("File yang tidak punya snapshot hari ini atau salinan backup tidak diubah oleh pencarian riwayat.")


if __name__ == "__main__":
    main()
