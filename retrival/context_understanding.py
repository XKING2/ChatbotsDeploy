import re
import threading
from typing import Any, Callable, Optional

from database.chroma_db import get_active_generation, heading_collection
from userquery.query_normalizer import normalize_query

# ============================================================
# KONFIGURASI
# ============================================================

RRF_K = 10               # makin kecil, peringkat atas makin berpengaruh
EXPLICIT_WEIGHT = 2.0    # nama layanan yang disebut langsung = 2 kanal peringkat 1
AGREE_RANK = 3           # "kanal setuju" = heading ada di peringkat <= 3
AGREE_CHANNELS = 2       # jumlah kanal yang harus setuju agar relevan
MAX_REFERENCE_TOKENS = 4 # kata rujukan ("itu", "it") hanya dihitung di kalimat pendek

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

STRONG_FOLLOWUPS = (
    "yang tadi", "yang sebelumnya", "yang dimaksud", "yang tersebut",
    "yang itu", "hal tadi", "hal tersebut", "seperti tadi",
    "seperti sebelumnya", "tadi bagaimana", "kalau begitu", "lalu bagaimana",
    "bagaimana dengan itu", "the previous one", "the one before",
    "as mentioned earlier", "what about that", "how about that",
    "regarding that", "about the same thing", "刚才那个", "之前那个",
    "那个怎么样", "那怎么办", "那么呢", "关于那个", "さっきの", "先ほどの",
    "前のもの", "それについて", "それはどうですか", "ではどうですか",
    "как насчет этого", "что насчет этого", "об этом",
    "как насчет предыдущего", "то же самое",
)

SHORT_FOLLOWUPS = {
    "berapa", "bagaimana", "kapan", "di mana", "dimana", "siapa", "apakah",
    "mengapa", "kenapa", "syaratnya apa", "berapa biayanya", "berapa harganya",
    "bagaimana caranya", "apa saja syaratnya", "apa persyaratannya",
    "dokumen apa saja", "berapa lama", "di mana mengurusnya", "how much",
    "how long", "when", "where", "who", "what are the requirements",
    "what documents are needed", "how do i apply", "how can i apply",
    "what is the cost", "what about the cost", "多少钱", "需要多久", "什么时候",
    "在哪里", "需要什么材料", "有什么要求", "怎么办理", "いくらですか",
    "どのくらいかかりますか", "いつですか", "どこですか", "必要な書類は何ですか",
    "どうすればいいですか", "сколько стоит", "сколько времени", "когда", "где",
    "какие документы нужны", "какие требования", "как подать заявление",
}

REFERENCE_TOKENS = {"ini", "itu", "tersebut", "tadi", "this", "that", "it",
                    "those", "these", "это", "этот", "эта", "эти"}
REFERENCE_CJK = ("这个", "那个", "这些", "那些", "それ", "これ", "あれ", "その")


# ============================================================
# TEKS
# ============================================================

def normalize(text: Any) -> str:
    text = re.sub(r"[^\w\s]", " ", str(text or "").casefold())
    return re.sub(r"\s+", " ", text).strip()


def _contains(query: str, target: str) -> bool:
    """
    Target ada di query sebagai urutan KATA utuh (tidak memotong di tengah
    kata). Teks CJK tidak berspasi, jadi dicocokkan sebagai substring.
    """
    if not query or not target:
        return False

    if CJK.search(target):
        return len(target.replace(" ", "")) >= 2 and (
            target.replace(" ", "") in query.replace(" ", "")
        )

    q, t = query.split(), target.split()
    if not t or (len(t) == 1 and len(t[0]) < 3) or len(t) > len(q):
        return False
    return any(q[i:i + len(t)] == t for i in range(len(q) - len(t) + 1))


# ============================================================
# KATALOG (diambil dari Chroma, dimuat ulang saat generation berganti)
# ============================================================

class Catalog:
    def __init__(self, generation, headings, subheadings, patterns):
        self.generation = generation
        self.headings = headings          # {heading_id: nama}
        self.subheadings = subheadings    # {subheading_id: {name, heading_id}}
        self.patterns = patterns          # [(heading_id, teks_ternormalisasi)]
        self._ids_by_name = {normalize(n): i for i, n in headings.items()}

    def heading_name(self, heading_id):
        return self.headings.get(str(heading_id)) if heading_id else None

    def heading_id(self, name):
        return self._ids_by_name.get(normalize(name)) if name else None

    def subheading(self, subheading_id):
        return self.subheadings.get(str(subheading_id)) if subheading_id else None

    @classmethod
    def load(cls, generation):
        data = heading_collection.get(include=["documents", "metadatas"])
        headings, subheadings, aliases = {}, {}, []

        for rid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            meta = meta or {}
            kind = meta.get("type")
            hid = str(meta.get("heading_id") or (rid if kind == "heading" else ""))

            if kind == "heading" and hid:
                headings[hid] = str(meta.get("heading") or doc or "").strip()
            elif kind == "subheading":
                sid = str(meta.get("subheading_id") or rid)
                subheadings[sid] = {
                    "name": str(meta.get("subheading") or doc or "").strip(),
                    "heading_id": hid,
                }
            elif kind == "alias" and meta.get("alias_scope") == "heading" and hid:
                aliases.append((hid, str(meta.get("alias") or "").strip()))

        # Nama yang boleh disebut pengguna: nama heading (dengan dan tanpa
        # tanda kurung), singkatan di dalam kurung, dan alias tingkat heading.
        names = []
        for hid, name in headings.items():
            names.append((hid, name))
            names.append((hid, re.sub(r"\([^()]*\)", " ", name)))
            names += [(hid, a) for a in re.findall(r"\(([^()]{2,15})\)", name)]
        names += aliases

        patterns = sorted({(hid, normalize(text)) for hid, text in names
                           if normalize(text)})
        return cls(generation, headings, subheadings, patterns)

    def find_explicit(self, texts):
        """Heading dengan kecocokan terpanjang (paling spesifik) dan panjangnya."""
        best = {}
        for hid, target in self.patterns:
            if any(_contains(text, target) for text in texts):
                best[hid] = max(best.get(hid, 0), len(target))
        if not best:
            return [], 0
        longest = max(best.values())
        return sorted(h for h, n in best.items() if n == longest), longest


# ============================================================
# CONTEXT UNDERSTANDING (tanpa state per pertanyaan: aman multi-thread)
# ============================================================

class ContextUnderstanding:

    def __init__(self):
        self._catalog = None
        self._lock = threading.Lock()

    def catalog(self) -> Catalog:
        try:
            generation = get_active_generation()
        except Exception:
            if self._catalog is not None:
                return self._catalog
            raise

        if self._catalog is None or self._catalog.generation != generation:
            with self._lock:
                if self._catalog is None or self._catalog.generation != generation:
                    self._catalog = Catalog.load(generation)
        return self._catalog

    # ---------------------------------------------------- follow-up
    @staticmethod
    def is_contextual_followup(question: str, has_previous: bool = True) -> bool:
        """
        Lanjutan dari topik sebelumnya. Kata rujukan seperti "itu"/"it" hanya
        dihitung pada kalimat pendek, supaya "paspor yang hilang itu bisa
        diganti?" atau "...if it expires" tidak dianggap lanjutan.
        """
        query = normalize(question)
        if not has_previous or not query:
            return False

        if any(_contains(query, normalize(p)) for p in STRONG_FOLLOWUPS):
            return True
        if query in {normalize(p) for p in SHORT_FOLLOWUPS}:
            return True

        tokens = query.split()
        if re.match(r"^(apa|apakah) itu\b", query):
            return False
        if len(tokens) <= MAX_REFERENCE_TOKENS and REFERENCE_TOKENS & set(tokens):
            return True
        return len(query) <= 12 and any(r in query for r in REFERENCE_CJK)

    # ---------------------------------------------------- ranking
    @staticmethod
    def _ranks(items, key="heading_id"):
        """{id: posisi terbaik} dari daftar yang sudah terurut (terbaik dulu)."""
        ranks = {}
        for position, item in enumerate(items or [], start=1):
            value = item.get(key)
            if value:
                ranks.setdefault(str(value), position)
        return ranks

    @staticmethod
    def _best_scores(items, key="heading_id"):
        scores = {}
        for item in items or []:
            value, score = item.get(key), item.get("score")
            if value and score is not None:
                scores[str(value)] = max(scores.get(str(value), score), score)
        return scores

    @staticmethod
    def _channel(items, key="heading_id"):
        """
        {id: (peringkat terbaik, kedekatan 0..1)}. Kedekatan = skor relatif
        terhadap rentang skor di kanal itu (min-max), sehingga kandidat
        "ikut-ikutan" di peringkat bawah tidak menyaingi kandidat yang
        benar-benar menonjol. Tanpa ambang skor tetap.
        """
        scores = [i.get("score") for i in items or [] if i.get("score") is not None]
        lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
        out = {}
        for position, item in enumerate(items or [], start=1):
            value = item.get(key)
            if not value or str(value) in out:
                continue
            score = item.get("score")
            closeness = 1.0 if (score is None or hi - lo < 1e-6) else (score - lo) / (hi - lo)
            out[str(value)] = (position, closeness)
        return out

    def rank_headings(self, heading_results, alias_results, subheading_results,
                      chunk_results, explicit_ids):
        """RRF berbobot: tiap heading mendapat satu suara terbaik per kanal."""
        channels = {
            "heading": self._channel(heading_results),
            "alias": self._channel(alias_results),
            "subheading": self._channel(subheading_results),
            "chunk": self._channel(chunk_results),
            "explicit": {hid: (1, 1.0) for hid in explicit_ids},
        }
        cosine = {}
        for items in (heading_results, alias_results, subheading_results,
                      chunk_results):
            for hid, score in self._best_scores(items).items():
                cosine[hid] = max(cosine.get(hid, score), score)

        ranked = []
        for hid in {h for votes in channels.values() for h in votes}:
            votes = {c: v[hid] for c, v in channels.items() if hid in v}
            weighted = sum(
                (EXPLICIT_WEIGHT if c == "explicit" else 1.0)
                * (0.5 + 0.5 * closeness) / (RRF_K + rank)
                for c, (rank, closeness) in votes.items()
            )
            ranks = {c: rank for c, (rank, _) in votes.items()}
            ranked.append({
                "heading_id": hid,
                "score": weighted,
                "rrf_score": weighted,
                "channel_ranks": ranks,
                "sources": sorted(ranks),
                "source_count": len(ranks),
                "best_cosine": cosine.get(hid),
            })

        ranked.sort(key=lambda r: (-r["rrf_score"], -r["source_count"],
                                   -(r["best_cosine"] or 0.0), r["heading_id"]))
        return ranked

    def rank_subheadings(self, heading_id, subheading_results, alias_results,
                         chunk_results, catalog):
        """
        Subheading di dalam heading terpilih, digabung dengan RRF dari
        pencarian subheading, alias subheading, dan chunk dalam heading itu.
        """
        evidence = {}

        def feed(channel, items):
            for position, item in enumerate(items or [], start=1):
                sid = item.get("subheading_id")
                if not sid or str(item.get("heading_id")) != str(heading_id):
                    continue
                slot = evidence.setdefault(
                    str(sid), {"ranks": {}, "cosine": None, "item": item}
                )
                slot["ranks"].setdefault(channel, position)
                score = item.get("score")
                if score is not None and (slot["cosine"] is None
                                          or score > slot["cosine"]):
                    slot["cosine"] = score

        feed("subheading", subheading_results)
        feed("alias", alias_results)
        feed("chunk", chunk_results)

        candidates = []
        for sid, slot in evidence.items():
            info = catalog.subheading(sid)
            candidates.append({
                "subheading_id": sid,
                "subheading": (slot["item"].get("subheading")
                               or (info["name"] if info else None)),
                "rrf_score": sum(1.0 / (RRF_K + r) for r in slot["ranks"].values()),
                "channel_ranks": slot["ranks"],
                "channel_count": len(slot["ranks"]),
                "score": slot["cosine"],
            })

        candidates.sort(key=lambda c: (
            -c["rrf_score"],
            c["channel_ranks"].get("chunk", 99),
            c["channel_ranks"].get("subheading", 99),
            c["subheading_id"],
        ))
        return candidates

    # ---------------------------------------------------- analisis
    def analyze(
        self,
        question,
        previous_heading=None,
        previous_heading_id=None,
        normalized_question=None,
        query_embedding=None,
        heading_results=None,
        alias_results=None,
        subheading_results=None,
        chunk_results=None,
        scoped_chunks: Optional[Callable[[str], list]] = None,
        language_info=None,  # diabaikan; bahasa dikenali LLM
    ):
        catalog = self.catalog()
        normalized_question = (
            normalized_question if normalized_question is not None
            else normalize_query(question)
        )
        texts = [t for t in dict.fromkeys(
            (normalize(question), normalize(normalized_question))) if t]

        previous_id = (
            str(previous_heading_id)
            if previous_heading_id and str(previous_heading_id) in catalog.headings
            else catalog.heading_id(previous_heading)
        )

        explicit_ids, _ = catalog.find_explicit(texts)
        is_followup = bool(
            previous_id and not explicit_ids and any(
                self.is_contextual_followup(t, True) for t in texts)
        )

        ranked = self.rank_headings(
            heading_results, alias_results, subheading_results,
            chunk_results, explicit_ids,
        )
        top = ranked[0] if ranked else None
        by_id = {r["heading_id"]: r for r in ranked}

        if is_followup:
            selected_id, reason = previous_id, "follow_up_context"
        elif explicit_ids:
            # Nama layanan yang disebut langsung menang atas kebetulan
            # peringkat; bila ada beberapa yang sama panjang, pilih yang
            # peringkatnya terbaik.
            selected_id = next(r["heading_id"] for r in ranked
                               if r["heading_id"] in explicit_ids)
            reason = "explicit_heading"
        elif top:
            selected_id, reason = top["heading_id"], "rrf_retrieval"
        else:
            selected_id, reason = None, "no_heading_candidates"

        # Relevan bila lanjutan, disebut langsung, atau beberapa kanal
        # sama-sama menempatkan heading ini di peringkat atas.
        selected = by_id.get(selected_id)
        # Pada katalog kecil, "peringkat 3" tidak berarti apa-apa; batasi
        # menjadi sepertiga jumlah heading.
        agree_rank = min(AGREE_RANK, max(1, len(catalog.headings) // 3))
        agreeing = sum(
            1 for r in (selected or {}).get("channel_ranks", {}).values()
            if r <= agree_rank
        )
        domain_relevant = bool(selected_id) and (
            is_followup
            or selected_id in explicit_ids
            or agreeing >= AGREE_CHANNELS
        )

        subheading_candidates, scoped = [], []
        if domain_relevant:
            if callable(scoped_chunks):
                scoped = scoped_chunks(selected_id) or []
            subheading_candidates = self.rank_subheadings(
                selected_id, subheading_results, alias_results, scoped, catalog
            )
        best = subheading_candidates[0] if subheading_candidates else None

        heading = catalog.heading_name(selected_id) or (
            previous_heading if selected_id and selected_id == previous_id else None
        )

        candidates = [
            {**r, "heading": catalog.heading_name(r["heading_id"])} for r in ranked
        ]

        return {
            "question": question,
            "normalized_question": normalized_question,

            "heading": heading,
            "heading_id": selected_id,

            "detected_heading": (
                catalog.heading_name(top["heading_id"]) if top else None),
            "detected_score": top["rrf_score"] if top else None,
            "explicit_heading": (
                catalog.heading_name(explicit_ids[0])
                if len(explicit_ids) == 1 else None),
            "ambiguous_candidates": (
                [catalog.heading_name(h) for h in explicit_ids]
                if len(explicit_ids) > 1 else []),

            "previous_heading": previous_heading,
            "previous_heading_id": previous_id,
            "is_followup": is_followup,

            "domain_relevant": domain_relevant,
            "resolution_reason": reason,
            "agreeing_channels": agreeing,

            "headings": [c["heading"] for c in candidates if c["heading"]],
            "heading_result": {
                "found": bool(candidates),
                "status": "found" if candidates else "not_found",
                "results": candidates,
                "top3": candidates[:3],
            },

            "subheading": best["subheading"] if best else None,
            "subheading_id": best["subheading_id"] if best else None,
            "subheading_score": best["score"] if best else None,
            "subheading_match_type": (
                "rrf_" + "+".join(sorted(best["channel_ranks"]))
                if best else None),
            "subheading_candidates": subheading_candidates[:5],

            "query_embedding": query_embedding,
            "anchored_topic": None,
        }