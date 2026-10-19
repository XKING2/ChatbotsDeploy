#!/bin/sh
# Start command Railway:  sh start.sh
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

SEED_DIR="$APP_DIR/data"                 # data awal yang ikut ter-push ke GitHub
DATA_DIR="${DATA_DIR:-$SEED_DIR}"        # di Railway: /data (volume)
export DATA_DIR

mkdir -p "$DATA_DIR"

# --- 1) Salin data awal ke volume (hanya SEKALI) -------------------------
# Volume Railway mulai kosong dan baru terpasang saat app jalan, bukan saat
# build. Penanda .seeded mencegah data hasil upload admin ikut tertimpa
# setiap kali redeploy. Paksa salin ulang: set FORCE_RESEED=1 sekali saja,
# lalu hapus variabelnya.
if [ "$DATA_DIR" != "$SEED_DIR" ] && [ -d "$SEED_DIR" ]; then
  if [ ! -e "$DATA_DIR/.seeded" ] || [ "${FORCE_RESEED:-0}" = "1" ]; then
    echo "[start] Menyalin data awal ke $DATA_DIR ..."
    cp -a "$SEED_DIR/." "$DATA_DIR/"
    touch "$DATA_DIR/.seeded"
    echo "[start] Data awal selesai disalin."
  else
    echo "[start] Data di $DATA_DIR sudah ada, tidak disalin ulang."
  fi
fi

# --- 2) Buat tabel + akun admin dari env (aman dijalankan berulang) -------
# Kalau gagal, server tetap dijalankan supaya log-nya bisa dibaca di Railway.
python create_admin.py --from-env || echo "[start] PERINGATAN: create_admin.py gagal, cek log di atas."

# --- 3) Jalankan server ---------------------------------------------------
# 1 worker supaya RAM tidak berlipat (Chroma + index dimuat per worker),
# 4 thread untuk melayani beberapa pengunjung sekaligus.
exec gunicorn app:app \
  --bind "0.0.0.0:${PORT:-8000}" \
  --workers 1 \
  --threads 4 \
  --timeout 180
