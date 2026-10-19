import os
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")

EMBEDDING_MODEL_1 = os.getenv("EMBEDDING_MODEL_1")
EMBEDDING_MODEL_2 = os.getenv("EMBEDDING_MODEL_2")

LLM_MODEL = os.getenv("LLM_MODEL")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DATA_DIR bisa diarahkan lewat environment variable (di Railway: /data,
# yaitu mount path volume). Kalau tidak di-set, tetap ./data seperti dulu,
# jadi cara jalan di laptop TIDAK berubah.
DATA_DIR = os.getenv("DATA_DIR") or os.path.join(BASE_DIR, "data")
PDF_FOLDER = os.path.join(DATA_DIR, "pdf")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Database riwayat chat ---
# Prioritas:
#   1) DATABASE_URL (Railway / SQLite):  sqlite:////data/chat.db
#   2) Kalau tidak ada -> MySQL dari DB_* seperti sebelumnya (laptop)
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "kantor_imigrasi_chat")

DATABASE_URL = os.getenv("DATABASE_URL") or (
    f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}/{DB_NAME}"
)

MAX_CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

CHUNK_COLLECTION_NAME = "knowledge_chunks"
ENTITY_COLLECTION_NAME = "knowledge_entities"

TOP_K = 5

KNOWLEDGE_INDEX_PATH = os.path.join(
    DATA_DIR,
    "knowledge_index.json"
)