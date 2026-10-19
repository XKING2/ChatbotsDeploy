sh
#!/bin/sh
# Railway Start Command: sh start.sh
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

SEED_DIR="$APP_DIR/data"
DATA_DIR="${DATA_DIR:-$SEED_DIR}"
export DATA_DIR

echo "[start] APP_DIR=$APP_DIR"
echo "[start] DATA_DIR=$DATA_DIR"

mkdir -p "$DATA_DIR"

# --- 1. Salin data awal ke volume hanya sekali ----------------------------
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

# --- 2. Cari makeadmin.py secara dinamis ----------------------------------
ADMIN_SCRIPT=""

for candidate in \
    "$APP_DIR/makeadmin.py" \
    "$APP_DIR/scripts/makeadmin.py" \
    "$APP_DIR/database/makeadmin.py"
do
    if [ -f "$candidate" ]; then
        ADMIN_SCRIPT="$candidate"
        break
    fi
done

# Cari di subfolder lain jika belum ditemukan.
if [ -z "$ADMIN_SCRIPT" ]; then
    ADMIN_SCRIPT="$(find "$APP_DIR" \
        -maxdepth 4 \
        -type f \
        -name "makeadmin.py" \
        -not -path "*/.venv/*" \
        -not -path "*/venv/*" \
        -print -quit 2>/dev/null || true)"
fi

if [ -n "$ADMIN_SCRIPT" ] && [ -f "$ADMIN_SCRIPT" ]; then
    echo "[start] File admin ditemukan: $ADMIN_SCRIPT"
    echo "[start] Menjalankan makeadmin.py..."

    if ! python "$ADMIN_SCRIPT" --from-env; then
        echo "[start] ERROR: makeadmin.py gagal dijalankan."
        exit 1
    fi

    echo "[start] Inisialisasi admin berhasil."
else
    echo "[start] ERROR: makeadmin.py tidak ditemukan."
    echo "[start] File Python yang tersedia:"
    find "$APP_DIR" -maxdepth 3 -type f -name "*.py" -print 2>/dev/null || true
    exit 1
fi

# --- 3. Jalankan aplikasi -------------------------------------------------
exec gunicorn app:app \
    --chdir "$APP_DIR" \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers 1 \
    --threads 4 \
    --timeout 180
