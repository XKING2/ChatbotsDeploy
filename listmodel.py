
from google import genai
from config import GEMINI_API_KEY, LLM_MODEL


def main():
    print("=== CEK MODEL GEMINI ===")
    print(f"Model dari config: {LLM_MODEL}")

    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY tidak ditemukan di .env/config.py")
        return

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Cek apakah API key dapat digunakan
        print("\n[INFO] Mengambil daftar model dari Gemini API...\n")

        models = list(client.models.list())

        if not models:
            print("[INFO] Tidak ada model yang dikembalikan API.")
            return

        for model in models:
            name = getattr(model, "name", "Unknown")
            supported = getattr(
                model, "supported_actions", []
            )

            if "generateContent" in supported:
                print(f"[SUPPORTED] {name}")
            else:
                print(f"[MODEL] {name}")

        print("\n=== TEST MODEL DARI CONFIG ===")

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents="Balas hanya dengan teks: TEST BERHASIL"
        )

        print(f"Model yang diminta: {LLM_MODEL}")
        print(f"Respons: {response.text}")
        print("\n[OK] Request berhasil.")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
