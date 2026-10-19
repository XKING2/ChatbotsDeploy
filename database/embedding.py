import time

from google import genai

from config import (
    GEMINI_API_KEY_1,
    GEMINI_API_KEY_2,
    EMBEDDING_MODEL_1,
)

from LLM.embedding_service import create_embedding


EMBEDDING_ROUTES = [
    {
        "name": "Primary",
        "api_key": GEMINI_API_KEY_1,
        "model": EMBEDDING_MODEL_1
    },
    {
        "name": "Backup API Key",
        "api_key": GEMINI_API_KEY_2,
        "model": EMBEDDING_MODEL_1
    }
]


MAX_RETRY = 2
RETRY_DELAY = 5


# ============================================================
# ACTIVE ROUTE
# ============================================================

active_route_index = 0


def get_active_route():
    return EMBEDDING_ROUTES[active_route_index]


def switch_to_backup():
    global active_route_index

    if active_route_index == 0:

        active_route_index = 1

        print(
            "\n=================================================="
        )

        print(
            "PRIMARY GAGAL"
        )

        print(
            "Beralih ke BACKUP API KEY"
        )

        print(
            "Backup akan digunakan untuk request berikutnya."
        )

        print(
            "=================================================="
        )

        return True

    return False


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    global active_route_index

    route_index = active_route_index

    while route_index < len(EMBEDDING_ROUTES):

        route = EMBEDDING_ROUTES[route_index]

        api_key = route["api_key"]
        model = route["model"]
        route_name = route["name"]

        if not api_key:

            print(
                f"\n[SKIP] {route_name}"
            )

            print(
                "API key tidak tersedia."
            )

            if route_index == 0:

                switch_to_backup()

                route_index = active_route_index

                continue

            return None

        if not model:

            print(
                f"\n[SKIP] {route_name}"
            )

            print(
                "Model tidak tersedia."
            )

            if route_index == 0:

                switch_to_backup()

                route_index = active_route_index

                continue

            return None

        print(
            f"\nEmbedding menggunakan: "
            f"{route_name} | {model}"
        )

        client = genai.Client(
            api_key=api_key
        )

        # ====================================================
        # RETRY ROUTE AKTIF
        # ====================================================

        for attempt in range(1, MAX_RETRY + 1):

            try:

                print(
                    f"[TRY] {route_name} "
                    f"percobaan {attempt}/{MAX_RETRY}"
                )

                response = client.models.embed_content(
                    model=model,
                    contents=text
                )

                if not response.embeddings:

                    raise Exception(
                        "Response embedding kosong."
                    )

                print(
                    f"[SUCCESS] "
                    f"{route_name} berhasil."
                )

                return response.embeddings[0].values

            except Exception as e:

                error_message = str(e)

                print(
                    f"\n[ERROR] {route_name}"
                )

                print(
                    f"Percobaan "
                    f"{attempt}/{MAX_RETRY}"
                )

                print(
                    error_message
                )

                # ============================================
                # MASIH ADA RETRY
                # ============================================

                if attempt < MAX_RETRY:

                    print(
                        f"Mencoba kembali dalam "
                        f"{RETRY_DELAY} detik..."
                    )

                    time.sleep(
                        RETRY_DELAY
                    )

                    continue

                # ============================================
                # RETRY HABIS
                # ============================================

                print(
                    f"\n[FAILED] "
                    f"{route_name} gagal "
                    f"{MAX_RETRY} kali."
                )

        # ====================================================
        # PRIMARY GAGAL -> BACKUP
        # ====================================================

        if route_index == 0:

            switched = switch_to_backup()

            if switched:

                route_index = active_route_index

                continue

        # ====================================================
        # BACKUP JUGA GAGAL
        # ====================================================

        print(
            "\n[FATAL] "
            "Semua jalur embedding gagal."
        )

        return None

    print(
        "\n[FATAL] "
        "Tidak ada route embedding yang tersedia."
    )

    return None