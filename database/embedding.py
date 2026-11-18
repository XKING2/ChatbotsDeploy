import array
import hashlib
import json
import math
import re
import sqlite3
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google import genai
from google.genai import types

import config as _cfg
from config import EMBEDDING_MODEL_1


# ============================================================
# KONFIGURASI
# ============================================================

def _conf(name, default):
    return getattr(_cfg, name, default)


DEFAULT_LIMITS = {
    "rpm": _conf("EMBEDDING_RPM", 100),
    "tpm": _conf("EMBEDDING_TPM", 30_000),
    "rpd": _conf("EMBEDDING_RPD", 1_000),
}

# Sisakan sebagian kapasitas agar selisih hitungan lokal vs server tidak menyebabkan 429.
SAFETY_FACTOR = 0.9

EMBEDDING_BATCH_SIZE = _conf("EMBEDDING_BATCH_SIZE", 20)
USE_CACHE = _conf("EMBEDDING_CACHE", True)

MAX_WAIT_SECONDS = 75          # batas tunggu untuk proses massal (ingest)
QUERY_MAX_WAIT_SECONDS = 15    # batas tunggu untuk satu pertanyaan chatbot
MAX_CONSECUTIVE_FAILURES = 8   # gagal beruntun sebelum menyerah (hasil yang selesai tetap tersimpan)
RETRY_DELAY = 5

_BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = Path(_conf("EMBEDDING_STATE_FILE", _BASE_DIR / "embedding_state.json"))
CACHE_FILE = Path(_conf("EMBEDDING_CACHE_FILE", _BASE_DIR / "embedding_cache.sqlite3"))

EMBEDDING_ROUTES = [
    {"name": "Primary", "api_key": _conf("GEMINI_API_KEY_1", None), "model": EMBEDDING_MODEL_1},
    {"name": "Backup1", "api_key": _conf("GEMINI_API_KEY_2", None), "model": EMBEDDING_MODEL_1},
    {"name": "Backup2", "api_key": _conf("GEMINI_API_KEY_3", None), "model": EMBEDDING_MODEL_1},
    {"name": "Backup3", "api_key": _conf("GEMINI_API_KEY_4", None), "model": EMBEDDING_MODEL_1},
    {"name": "Backup4", "api_key": _conf("GEMINI_API_KEY_5", None), "model": EMBEDDING_MODEL_1},
]

# Dapat ditimpa saat pengujian.
_now = time.monotonic
_sleep = time.sleep


class EmbeddingCountMismatch(Exception):
    """Jumlah vektor dari API tidak sama dengan jumlah teks yang dikirim."""


# ============================================================
# UTILITAS
# ============================================================

def estimate_tokens(text):
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return int((len(text) - non_ascii) / 3.5 + non_ascii * 1.1) + 1


def _quota_day():
    """Kuota harian Gemini reset tengah malam waktu Pasifik."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        return (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%d")


def parse_quota_error(text):
    """Ambil rincian kuota dari pesan 429 Gemini (JSON atau repr dict Python)."""
    info = {}
    for key in ("quotaId", "quotaValue"):
        m = re.search(rf"""["']{key}["']\s*:\s*["']([^"']+)["']""", text)
        if m:
            info[key] = m.group(1)
    m = re.search(r"""["']retryDelay["']\s*:\s*["']([\d.]+)s["']""", text)
    if m:
        info["retryDelay"] = float(m.group(1))
    else:
        m = re.search(r"retry in\s+([\d.]+)\s*s", text, re.IGNORECASE)
        if m:
            info["retryDelay"] = float(m.group(1))
    return info


# ============================================================
# STATE HARIAN (JSON)
# ============================================================

class DayState:
    """Hitungan teks per hari per route, bertahan antar-run."""

    def __init__(self, path):
        self.path, self.day = Path(path), _quota_day()
        self.used, self.exhausted = {}, {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("day") == self.day:
                self.used = dict(data.get("used", {}))
                self.exhausted = dict(data.get("exhausted", {}))
        except (OSError, ValueError):
            pass

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "day": self.day, "used": self.used, "exhausted": self.exhausted,
            }, indent=2), encoding="utf-8")
        except OSError:
            pass


# ============================================================
# CACHE VEKTOR (SQLite)
# ============================================================

class VectorCache:
    def __init__(self, path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, blob BLOB NOT NULL)"
        )
        self.conn.commit()

    @staticmethod
    def make_key(model, task_type, text):
        return hashlib.sha256(f"{model}|{task_type}|{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys):
        found = {}
        for start in range(0, len(keys), 500):
            chunk = keys[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for key, blob in self.conn.execute(
                f"SELECT key, blob FROM vec WHERE key IN ({marks})", chunk
            ):
                values = array.array("f")
                values.frombytes(blob)
                found[key] = list(values)
        return found

    def put_many(self, items):
        rows = [(k, array.array("f", v).tobytes()) for k, v in items]
        self.conn.executemany("INSERT OR REPLACE INTO vec (key, blob) VALUES (?, ?)", rows)
        self.conn.commit()


_cache = None


def _get_cache():
    global _cache
    if not USE_CACHE:
        return None
    if _cache is None:
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _cache = VectorCache(CACHE_FILE)
        except Exception as error:
            print(f"[EMBEDDING] Cache dinonaktifkan: {type(error).__name__}: {error}")
            return None
    return _cache


# ============================================================
# ROUTE + PEMBATAS
# ============================================================

class Route:
    def __init__(self, cfg, state):
        self.cfg, self.state = cfg, state
        self.name = cfg["name"]
        self.model = cfg["model"]
        self.api_key = (cfg.get("api_key") or "").strip()
        limits = {k: cfg.get(k, DEFAULT_LIMITS[k]) for k in DEFAULT_LIMITS}
        self.rpm = max(1, int(limits["rpm"] * SAFETY_FACTOR))
        self.tpm = max(1, int(limits["tpm"] * SAFETY_FACTOR))
        self.rpd = max(1, int(limits["rpd"] * SAFETY_FACTOR))
        self.window = deque()  # (waktu, jumlah_teks, token)
        self.blocked_until = 0.0
        self.disabled = not (self.api_key and self.model)
        self.failures = 0

    @property
    def key_id(self):
        fingerprint = hashlib.sha256(self.api_key.encode()).hexdigest()[:8] if self.api_key else self.name
        return f"{fingerprint}:{self.model}"

    def used_today(self):
        return self.state.used.get(self.key_id, 0)

    def day_exhausted(self):
        return self.key_id in self.state.exhausted or self.used_today() >= self.rpd

    def usable(self):
        return not (self.disabled or self.day_exhausted())

    def status_text(self):
        if not self.api_key:
            return "tidak aktif (API key kosong)"
        if self.disabled:
            return "dinonaktifkan karena error"
        used = self.used_today()
        if self.key_id in self.state.exhausted:
            return (f"habis harian menurut server (tercatat {used}/{self.rpd}). "
                    f"Pesan: {self.state.exhausted[self.key_id]}")
        if used >= self.rpd:
            return f"batas harian lokal tercapai ({used}/{self.rpd})"
        return f"siap ({used}/{self.rpd} teks hari ini)"

    def _purge(self, now):
        while self.window and self.window[0][0] <= now - 60:
            self.window.popleft()

    def plan(self, token_list):
        """Return (n_teks_yang_boleh_dikirim_sekarang, detik_tunggu). wait=inf bila mustahil."""
        if not self.usable():
            return 0, math.inf
        now = _now()
        self._purge(now)
        day_left = self.rpd - self.used_today()
        rpm_left = self.rpm - sum(w[1] for w in self.window)
        tpm_left = self.tpm - sum(w[2] for w in self.window)

        if self.blocked_until > now:
            return 0, self.blocked_until - now

        n, tokens = 0, 0
        while (n < len(token_list) and n + 1 <= min(rpm_left, day_left)
               and tokens + token_list[n] <= tpm_left):
            tokens += token_list[n]
            n += 1
        if n:
            return n, 0.0

        if not token_list or token_list[0] > self.tpm:
            return 0, math.inf  # satu teks melebihi TPM route ini
        # Tunggu sampai entri tertua keluar dari window 60 detik.
        freed_texts, freed_tokens = 0, 0
        for stamp, texts, tok in self.window:
            freed_texts += texts
            freed_tokens += tok
            if rpm_left + freed_texts >= 1 and tpm_left + freed_tokens >= token_list[0]:
                return 0, max(0.0, stamp + 60 - now) + 0.5
        return 0, 1.0

    def record(self, n_texts, tokens):
        self.window.append((_now(), n_texts, tokens))

    def count_daily(self, n_texts):
        self.state.used[self.key_id] = self.used_today() + n_texts
        self.state.save()

    def handle_error(self, error):
        """Tandai route menurut jenis error. Return keterangan singkat."""
        text = str(error)
        upper = text.upper()
        now = _now()
        self.failures += 1
        if "429" in upper or "RESOURCE_EXHAUSTED" in upper or "QUOTA" in upper:
            info = parse_quota_error(text)
            quota_id = info.get("quotaId", "")
            limit = info.get("quotaValue")
            detail = f"{quota_id or 'kuota tidak diketahui'}, batas={limit if limit else '?'}"
            if limit == "0":
                self.disabled = True
                return f"model ini tidak punya kuota di project/tier key ini ({detail})"
            per_day = ("PERDAY" in quota_id.upper() if quota_id
                       else "PERDAY" in upper.replace(" ", "").replace("_", ""))
            if per_day:
                self.state.exhausted[self.key_id] = f"{detail} | {text.replace(chr(10), ' ')[:150]}"
                self.state.save()
                return f"kuota HARIAN habis menurut server ({detail})"
            delay = info.get("retryDelay")
            self.blocked_until = now + (delay + 2 if delay else 65)
            return f"limit per menit dari server ({detail}); tunggu {delay or 65:.0f} detik"
        if any(m in upper for m in ("404", "NOT_FOUND", "401", "403", "API KEY", "PERMISSION_DENIED")):
            self.disabled = True
            return "model/API key tidak bisa dipakai"
        self.blocked_until = now + RETRY_DELAY * min(self.failures, 6)
        return f"error sementara ({type(error).__name__})"


_state = None
_routes = None
_clients = {}
_expected_dimension = None
_preferred_index = 0


def _get_routes():
    global _state, _routes
    if _routes is None:
        _state = DayState(STATE_FILE)
        _routes, seen = [], set()
        for cfg in EMBEDDING_ROUTES:
            route = Route(cfg, _state)
            if route.key_id in seen:
                continue  # key + model yang sama berbagi kuota yang sama
            seen.add(route.key_id)
            used = _state.used.get(route.key_id, 0)
            if route.key_id in _state.exhausted and used < route.rpd:
                _state.exhausted.pop(route.key_id, None)  # penanda lama: diverifikasi ulang
            _routes.append(route)
    return _routes


def print_status():
    print("[EMBEDDING] Status route:")
    for route in _get_routes():
        print(f"  {route.name} [{route.model}] -> {route.status_text()} "
              f"(batas efektif: {route.rpm} teks/menit, {route.tpm} token/menit, {route.rpd} teks/hari)")


def _ordered_routes():
    routes = _get_routes()
    start = min(_preferred_index, len(routes) - 1)
    return routes[start:] + routes[:start]


# ============================================================
# VALIDASI HASIL
# ============================================================

def _validate_embeddings(embeddings, expected_count):
    if embeddings is None:
        raise ValueError("Response Gemini tidak memiliki embeddings.")
    if len(embeddings) != expected_count:
        raise EmbeddingCountMismatch(
            f"Jumlah embedding tidak sesuai: {len(embeddings)} dari {expected_count}."
        )
    vectors = []
    for index, embedding in enumerate(embeddings):
        values = getattr(embedding, "values", None)
        if values is None or len(values) == 0:
            raise ValueError(f"Embedding index {index} kosong.")
        vector = [float(v) for v in values]
        if not all(math.isfinite(v) for v in vector):
            raise ValueError(f"Embedding index {index} mengandung nilai tidak valid.")
        vectors.append(vector)
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("Dimensi vector dalam batch tidak konsisten.")
    return vectors


def _validate_dimension(vectors):
    global _expected_dimension
    dimension = len(vectors[0])
    if _expected_dimension is None:
        _expected_dimension = dimension
        print(f"[EMBEDDING] Dimensi vector ditetapkan: {dimension}")
    elif dimension != _expected_dimension:
        raise ValueError(f"Dimensi embedding berubah: {dimension}, seharusnya {_expected_dimension}.")


# ============================================================
# REQUEST KE GEMINI
# ============================================================

def _get_client(route):
    if route.api_key not in _clients:
        _clients[route.api_key] = genai.Client(api_key=route.api_key)
    return _clients[route.api_key]


def _call_api(route, texts, task_type):
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=t)]) for t in texts
    ]
    kwargs = {"model": route.model, "contents": contents}
    if task_type:
        kwargs["config"] = types.EmbedContentConfig(task_type=task_type)
    response = _get_client(route).models.embed_content(**kwargs)
    vectors = _validate_embeddings(getattr(response, "embeddings", None), len(texts))
    _validate_dimension(vectors)
    return vectors


# ============================================================
# BATCH EMBEDDING
# ============================================================

def _embed_unique(texts, batch_size, task_type, max_wait):
    """Embed teks unik. Return dict {teks: vektor} untuk yang berhasil (bisa sebagian)."""
    cache = _get_cache()
    results = {}
    todo = list(texts)
    tokens_of = {t: estimate_tokens(t) for t in todo}
    single_mode = False
    fail_streak = 0
    total = len(todo)

    while todo:
        limit = 1 if single_mode else batch_size
        window_texts = todo[:limit]
        token_list = [tokens_of[t] for t in window_texts]

        chosen, chosen_n, min_wait = None, 0, math.inf
        for route in _ordered_routes():
            n, wait = route.plan(token_list)
            if n > 0:
                chosen, chosen_n = route, n
                break
            min_wait = min(min_wait, wait)

        if chosen is None:
            if min_wait == math.inf:
                print("[EMBEDDING] Semua route sudah habis kuota harian/tidak tersedia:")
                print_status()
                break
            if min_wait > max_wait:
                print(f"[EMBEDDING] Semua route sedang penuh (tunggu ~{min_wait:.0f} dtk > batas {max_wait} dtk).")
                break
            print(f"[EMBEDDING] Menunggu {min_wait:.1f} detik (batas per menit)...")
            _sleep(min_wait)
            continue

        batch = window_texts[:chosen_n]
        batch_tokens = sum(tokens_of[t] for t in batch)
        chosen.record(len(batch), batch_tokens)
        try:
            vectors = _call_api(chosen, batch, task_type)
        except EmbeddingCountMismatch as error:
            print(f"[EMBEDDING] {error} -> beralih ke satu teks per request.")
            single_mode = True
            continue
        except Exception as error:
            reason = chosen.handle_error(error)
            fail_streak += 1
            print(f"[EMBEDDING] {chosen.name} gagal: {reason}\n"
                  f"    Pesan: {type(error).__name__}: {str(error)[:300]}")
            if fail_streak >= MAX_CONSECUTIVE_FAILURES:
                print("[EMBEDDING] Terlalu banyak kegagalan beruntun; berhenti (hasil yang selesai tersimpan).")
                break
            continue

        fail_streak = 0
        chosen.failures = 0
        chosen.count_daily(len(batch))
        for text, vector in zip(batch, vectors):
            results[text] = vector
        if cache:
            cache.put_many(
                (VectorCache.make_key(_get_routes()[0].model, task_type, t), v)
                for t, v in zip(batch, vectors)
            )
        del todo[:len(batch)]
        print(f"[EMBEDDING] {chosen.name} OK | {len(batch)} teks | selesai {len(results)}/{total}")
    return results


def create_embeddings_batch(texts, batch_size=EMBEDDING_BATCH_SIZE, task_type=None,
                            allow_partial=False, max_wait=MAX_WAIT_SECONDS):
    """
    Embed banyak teks dengan batas RPM/TPM/RPD, cache, dan failover.

    Return list vektor sejajar dengan `texts`. Jika ada yang gagal (mis. kuota harian habis):
    - allow_partial=False (bawaan): return None. Hasil yang sudah selesai tetap ada di cache,
      jadi menjalankan ulang melanjutkan tanpa memakai kuota lagi untuk teks yang sama.
    - allow_partial=True: return list dengan None pada posisi yang gagal.
    """
    if not texts:
        return []
    if not isinstance(texts, list):
        raise TypeError("texts harus berupa list.")
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Semua teks embedding harus berupa string nonkosong.")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size harus berupa integer minimal 1.")

    model = _get_routes()[0].model
    cache = _get_cache()
    by_text = {}
    unique = list(dict.fromkeys(texts))

    if cache:
        keys = {t: VectorCache.make_key(model, task_type, t) for t in unique}
        cached = cache.get_many(list(keys.values()))
        for text, key in keys.items():
            if key in cached:
                by_text[text] = cached[key]

    missing = [t for t in unique if t not in by_text]
    print(f"\n[EMBEDDING] Total teks: {len(texts)} | unik: {len(unique)} | "
          f"dari cache: {len(unique) - len(missing)} | perlu dikirim: {len(missing)}")
    if missing:
        by_text.update(_embed_unique(missing, batch_size, task_type, max_wait))

    output = [by_text.get(t) for t in texts]
    failed = sum(1 for v in output if v is None)
    if failed:
        print(f"[EMBEDDING] {failed} teks belum ter-embed. Jalankan ulang setelah kuota tersedia; "
              "yang sudah selesai tidak diulang.")
        return output if allow_partial else None
    print(f"[EMBEDDING] SELESAI | teks: {len(texts)}")
    return output


# ============================================================
# SINGLE EMBEDDING (query chatbot)
# ============================================================

def create_embedding(text, task_type=None):
    """Embed satu teks. Menunggu paling lama QUERY_MAX_WAIT_SECONDS agar chatbot tidak menggantung."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Teks embedding tidak boleh kosong.")
    results = create_embeddings_batch(
        [text], batch_size=1, task_type=task_type, max_wait=QUERY_MAX_WAIT_SECONDS
    )
    return results[0] if results else None


def request_embedding(route, text):
    """Kompatibilitas lama: embed satu teks lewat route tertentu (tanpa cache)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Teks embedding tidak boleh kosong.")
    if route not in EMBEDDING_ROUTES:
        raise ValueError("Route embedding tidak dikenal.")
    target = next((r for r in _get_routes() if r.cfg is route), None)
    if target is None or not target.usable():
        return None
    try:
        return _call_api(target, [text], None)[0]
    except Exception as error:
        print(f"[EMBEDDING] {target.name} gagal: {target.handle_error(error)}")
        return None


# ============================================================
# KOMPATIBILITAS FUNGSI LAMA
# ============================================================

def get_active_route():
    return EMBEDDING_ROUTES[min(_preferred_index, len(EMBEDDING_ROUTES) - 1)]


def switch_to_backup():
    global _preferred_index
    _preferred_index = 1 if len(EMBEDDING_ROUTES) > 1 else 0


def switch_to_primary():
    global _preferred_index
    _preferred_index = 0