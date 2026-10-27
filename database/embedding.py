
import time
from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY_1,
    GEMINI_API_KEY_2,
    EMBEDDING_MODEL_1,
)


# ============================================================
# CONFIGURATION
# ============================================================

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
PRIMARY_COOLDOWN = 60

EMBEDDING_BATCH_SIZE = 32

active_route_index = 0
primary_retry_after = 0.0


# ============================================================
# ROUTE MANAGEMENT
# ============================================================

def get_active_route():
    return EMBEDDING_ROUTES[active_route_index]


def switch_to_backup():
    global active_route_index, primary_retry_after

    active_route_index = 1
    primary_retry_after = time.monotonic() + PRIMARY_COOLDOWN

    print("\n[ROUTE] Primary gagal.")
    print("[ROUTE] Beralih ke Backup API Key.")
    print(
        f"[ROUTE] Primary akan diuji kembali setelah "
        f"{PRIMARY_COOLDOWN} detik."
    )


def switch_to_primary():
    global active_route_index

    active_route_index = 0

    print("\n[ROUTE] Primary berhasil kembali.")
    print("[ROUTE] Request berikutnya menggunakan Primary.")



def _request_batch(route, texts):

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

    # Setiap teks menjadi satu Content terpisah.
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )
        for text in texts
    ]

    for attempt in range(1, MAX_RETRY + 1):
        try:
            print(
                f"[TRY] {route['name']} "
                f"batch={len(texts)} "
                f"{attempt}/{MAX_RETRY}"
            )

            response = client.models.embed_content(
                model=model,
                contents=contents,
            )

            embeddings = response.embeddings or []

            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Jumlah embedding tidak sesuai: "
                    f"{len(embeddings)} dari {len(texts)} input."
                )

            vectors = []

            for index, item in enumerate(embeddings):
                if item.values is None:
                    raise ValueError(
                        f"Embedding kosong pada item ke-{index + 1}."
                    )

                vectors.append(list(item.values))

            print(
                f"[SUCCESS] {route['name']} "
                f"batch={len(vectors)}"
            )

            return vectors

        except Exception as exc:
            print(
                f"[ERROR] {route['name']} "
                f"{attempt}/{MAX_RETRY}: {exc}"
            )

            if attempt < MAX_RETRY:
                time.sleep(RETRY_DELAY)

    return None


def create_embeddings_batch(
    texts,
    batch_size=EMBEDDING_BATCH_SIZE,
):

    global active_route_index, primary_retry_after

    if not texts:
        return []

    if isinstance(texts, str):
        raise TypeError(
            "texts harus berupa list teks, bukan satu string."
        )

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "Semua teks embedding harus berupa string nonkosong."
        )

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size harus berupa integer minimal 1.")

    output = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        result = None

        # Uji kembali Primary jika cooldown sudah selesai.
        if (
            active_route_index == 1
            and time.monotonic() >= primary_retry_after
        ):
            print("[CHECK] Mencoba kembali API key Primary.")

            result = _request_batch(
                EMBEDDING_ROUTES[0],
                batch,
            )

            if result is not None:
                switch_to_primary()
            else:
                primary_retry_after = (
                    time.monotonic() + PRIMARY_COOLDOWN
                )
                print(
                    "[ROUTE] Primary belum pulih. "
                    "Tetap memakai Backup."
                )

        # Coba route aktif jika Primary belum berhasil.
        if result is None:
            route = get_active_route()
            result = _request_batch(route, batch)

        # Jika Primary gagal, coba Backup.
        if result is None and active_route_index == 0:
            switch_to_backup()

            result = _request_batch(
                EMBEDDING_ROUTES[1],
                batch,
            )

        if result is None:
            print(
                f"[FATAL] Semua jalur embedding gagal pada "
                f"item {start + 1}-{start + len(batch)}."
            )

            # Jangan mengembalikan hasil parsial karena dapat
            # menyebabkan pemetaan vektor dan teks menjadi salah.
            return None

        output.extend(result)

    if len(output) != len(texts):
        print("[FATAL] Jumlah hasil akhir tidak sesuai input.")
        return None

    return output


def request_embedding(route, text):

    results = _request_batch(route, [text])

    if results is None:
        return None

    return results[0]


def create_embedding(text):

    results = create_embeddings_batch(
        [text],
        batch_size=1,
    )

    if results is None:
        return None

    return results[0]
