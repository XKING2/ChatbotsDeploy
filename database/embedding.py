
import time
from google import genai

from config import (
    GEMINI_API_KEY_1,
    GEMINI_API_KEY_2,
    EMBEDDING_MODEL_1,
)


EMBEDDING_ROUTES = [
    {
        "name": "Primary",
        "api_key": GEMINI_API_KEY_1,
        "model": EMBEDDING_MODEL_1,
    },
    {
        "name": "Backup API Key",
        "api_key": GEMINI_API_KEY_2,
        "model": EMBEDDING_MODEL_1,
    },
]

MAX_RETRY = 2
RETRY_DELAY = 5

# Primary akan diuji kembali setelah jeda ini.
PRIMARY_COOLDOWN = 60

active_route_index = 0
primary_retry_after = 0.0


def get_active_route():
    return EMBEDDING_ROUTES[active_route_index]


def switch_to_backup():
    global active_route_index, primary_retry_after

    active_route_index = 1
    primary_retry_after = time.monotonic() + PRIMARY_COOLDOWN

    print("\n[ROUTE] Primary gagal.")
    print("[ROUTE] Beralih ke Backup API Key.")
    print(
        f"[ROUTE] Primary akan dicoba kembali "
        f"setelah {PRIMARY_COOLDOWN} detik."
    )


def switch_to_primary():
    global active_route_index

    active_route_index = 0

    print("\n[ROUTE] Primary berhasil kembali.")
    print("[ROUTE] Request berikutnya menggunakan Primary.")


def request_embedding(route, text):
    """Mencoba satu route dengan retry terbatas."""
    api_key = route["api_key"]
    model = route["model"]

    if not api_key or not model:
        print(f"[SKIP] {route['name']}: API key/model kosong.")
        return None

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        print(f"[ERROR] Gagal membuat client {route['name']}: {exc}")
        return None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            print(
                f"[TRY] {route['name']} "
                f"{attempt}/{MAX_RETRY}"
            )

            response = client.models.embed_content(
                model=model,
                contents=text,
            )

            if not response.embeddings:
                raise ValueError("Response embedding kosong.")

            values = response.embeddings[0].values

            if values is None:
                raise ValueError("Vektor embedding kosong.")

            print(f"[SUCCESS] {route['name']}")
            return values

        except Exception as exc:
            print(
                f"[ERROR] {route['name']} "
                f"{attempt}/{MAX_RETRY}: {exc}"
            )

            if attempt < MAX_RETRY:
                time.sleep(RETRY_DELAY)

    return None


def create_embedding(text):
    global active_route_index, primary_retry_after

    # Jika backup sedang aktif, uji Primary setelah cooldown.
    if active_route_index == 1:
        if time.monotonic() >= primary_retry_after:
            print("[CHECK] Mencoba kembali API key Primary.")

            result = request_embedding(
                EMBEDDING_ROUTES[0],
                text,
            )

            if result is not None:
                switch_to_primary()
                return result

            # Primary masih gagal; jadwalkan pengujian berikutnya.
            primary_retry_after = (
                time.monotonic() + PRIMARY_COOLDOWN
            )
            print("[ROUTE] Primary belum pulih. Tetap memakai Backup.")

    # Coba route yang sedang aktif.
    route = EMBEDDING_ROUTES[active_route_index]
    result = request_embedding(route, text)

    if result is not None:
        return result

    # Jika Primary gagal, langsung coba Backup.
    if active_route_index == 0:
        switch_to_backup()

        backup_result = request_embedding(
            EMBEDDING_ROUTES[1],
            text,
        )

        if backup_result is not None:
            return backup_result

    print("[FATAL] Semua jalur embedding gagal.")
    return None
