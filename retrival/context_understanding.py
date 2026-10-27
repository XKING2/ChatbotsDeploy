import re
from typing import Any, Dict, List, Optional, Tuple

from retrival.entity_retrieval import EntityRetrieval
from userquery.query_normalizer import normalize_query


class ContextUnderstanding:

    def __init__(self):
        self.entity_retrieval = EntityRetrieval()
        self.entities = self.entity_retrieval.get_entities() or []
        self.aliases = self.entity_retrieval.get_aliases() or []

        self.entity_ids_by_name: Dict[str, str] = {}
        self.entity_names_by_id: Dict[str, str] = {}
        self.aliases_by_entity: Dict[str, List[str]] = {}

        for item in self.entities:
            entity_id = item.get("entity_id")
            name = (item.get("name") or item.get("entity") or "").strip()
            if entity_id and name:
                self.entity_ids_by_name[name] = entity_id
                self.entity_names_by_id[entity_id] = name

        for item in self.aliases:
            entity_id = item.get("entity_id")
            alias = (item.get("alias") or "").strip()
            if entity_id and alias:
                bucket = self.aliases_by_entity.setdefault(entity_id, [])
                if alias not in bucket:
                    bucket.append(alias)

        # Akronim di dalam nama entity, mis. "(KITAS)", dianggap alias.
        for entity_id, name in self.entity_names_by_id.items():
            for acronym in re.findall(r"\(([^()]{2,15})\)", name):
                acronym = acronym.strip()
                bucket = self.aliases_by_entity.setdefault(entity_id, [])
                if acronym and acronym not in bucket:
                    bucket.append(acronym)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def normalize(text: Any) -> str:
        text = str(text or "").casefold()
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _score(item: Optional[Dict[str, Any]]) -> Optional[float]:
        if not item:
            return None
        try:
            value = item.get("score")
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def entity_id(self, entity: Optional[str]) -> Optional[str]:
        if not entity:
            return None
        found = self.entity_ids_by_name.get(entity)
        if found:
            return found
        target = self.normalize(entity)
        for name, entity_id in self.entity_ids_by_name.items():
            if self.normalize(name) == target:
                return entity_id
        return None

    def is_contextual_followup(
        self, question: str, previous_entity: Optional[str] = None
    ) -> bool:
        q = self.normalize(question)
        if not q:
            return False

        strong_phrases = (
            "yang tadi", "yang sebelumnya", "yang dimaksud", "yang tersebut",
            "yang itu", "hal tadi", "hal tersebut", "seperti tadi",
            "seperti sebelumnya", "tadi bagaimana", "kalau begitu",
            "lalu bagaimana",
        )
        if any(phrase in q for phrase in strong_phrases):
            return True

        if previous_entity:
            has_reference = any(
                t in {"ini", "itu", "tersebut", "tadi"} for t in q.split()
            )
            if has_reference and not re.match(r"^(apa|apakah) itu\b", q):
                return True

        short_followups = {
            "berapa", "berapa biayanya", "bagaimana", "bagaimana caranya",
            "kapan", "di mana", "dimana", "siapa", "apakah", "mengapa",
            "kenapa", "syaratnya apa", "kalau begitu",
        }
        return bool(previous_entity and q in short_followups)

    def find_explicit_entity(self, question: str) -> Tuple[Optional[str], List[str]]:
        """Hanya informasi diagnostik: nama/alias entity yang tertulis utuh.

        Tidak ikut menentukan pilihan entity (itu murni skor embedding).
        """
        q_tokens = self.normalize(question).split()
        matches = []

        for entity_id, name in self.entity_names_by_id.items():
            for raw in [name] + self.aliases_by_entity.get(entity_id, []):
                tokens = self.normalize(raw).split()
                if not tokens or len(tokens) > len(q_tokens):
                    continue
                for i in range(len(q_tokens) - len(tokens) + 1):
                    if q_tokens[i:i + len(tokens)] == tokens:
                        matches.append((len(tokens), entity_id, name))
                        break

        if not matches:
            return None, []

        longest = max(m[0] for m in matches)
        best = {eid: name for length, eid, name in matches if length == longest}
        if len(best) == 1:
            return next(iter(best.values())), []
        return None, sorted(best.values())

    # --------------------------------------------------------------- evidence
    def _collect_evidence(self, entity_results, subtopic_results, chunk_results):
        """entity_id -> {sumber: (skor_terbaik, item)}."""
        evidence: Dict[str, Dict[str, Tuple[float, Dict[str, Any]]]] = {}
        sources = (
            ("entity", entity_results),
            ("subtopic", subtopic_results),
            ("chunk", chunk_results),
        )
        for source, items in sources:
            for item in items:
                entity_id = item.get("entity_id")
                score = self._score(item)
                if not entity_id or score is None:
                    continue
                slot = evidence.setdefault(entity_id, {})
                if source not in slot or score > slot[source][0]:
                    slot[source] = (score, item)
        return evidence

    @staticmethod
    def _rank_entities(evidence):
        ranked = []
        for entity_id, slot in evidence.items():
            ranked.append({
                "entity_id": entity_id,
                "score": max(score for score, _ in slot.values()),
                "sources": sorted(slot),
            })
        # Skor terbaik dulu; jumlah sumber hanya memutus seri.
        ranked.sort(key=lambda r: (r["score"], len(r["sources"])), reverse=True)
        return ranked

    def _best_subtopic_item(self, entity_id, items):
        """(skor, item) subtopic terbaik milik entity, dari subtopic dan chunk."""
        best = None
        for item in items:
            if item.get("entity_id") != entity_id or not item.get("subtopic_id"):
                continue
            score = self._score(item)
            if score is None:
                continue
            if best is None or score > best[0]:
                best = (score, item)
        return best

    # ---------------------------------------------------------------- analyze
    def analyze(
        self,
        question,
        previous_entity=None,
        normalized_question=None,
        query_embedding=None,
        entity_results=None,
        subtopic_results=None,
        chunk_results=None,
    ):
        normalized_question = normalized_question or normalize_query(question)
        entity_results = list(entity_results or [])
        subtopic_results = list(subtopic_results or [])
        chunk_results = list(chunk_results or [])

        is_followup = self.is_contextual_followup(normalized_question, previous_entity)
        explicit_entity, ambiguous = self.find_explicit_entity(normalized_question)
        previous_id = self.entity_id(previous_entity)

        evidence = self._collect_evidence(entity_results, subtopic_results, chunk_results)
        ranked = self._rank_entities(evidence)
        top = ranked[0] if ranked else None
        margin = (
            top["score"] - ranked[1]["score"] if top and len(ranked) > 1 else None
        )

        # Pilih entity.
        if is_followup and previous_id:
            entity_id, resolution_reason = previous_id, "follow_up_context"
        elif top:
            entity_id, resolution_reason = top["entity_id"], "semantic_score"
        else:
            entity_id, resolution_reason = None, "no_entity_candidates"
        entity = self.entity_names_by_id.get(entity_id) if entity_id else None

        # Pilih subtopic dari skor embedding milik entity terpilih.
        subtopic = subtopic_id = subtopic_score = subtopic_match_type = None
        if entity_id:
            best = self._best_subtopic_item(entity_id, subtopic_results + chunk_results)
            if best:
                subtopic_score, item = best
                subtopic = item.get("subtopic")
                subtopic_id = item.get("subtopic_id")
                subtopic_match_type = (
                    "semantic_chunk" if item.get("type") == "chunk" else "semantic_subtopic"
                )

        entity_candidates = [
            {
                "entity_id": r["entity_id"],
                "entity": self.entity_names_by_id.get(r["entity_id"]),
                "score": r["score"],
                "sources": r["sources"],
            }
            for r in ranked
        ]

        return {
            "question": question,
            "normalized_question": normalized_question,
            "entity": entity,
            "entity_id": entity_id,
            "detected_entity": (
                self.entity_names_by_id.get(top["entity_id"]) if top else None
            ),
            "detected_score": top["score"] if top else None,
            "entity_score_margin": margin,
            "explicit_entity": explicit_entity,
            "alias_overridden": False,
            "ambiguous_candidates": ambiguous,
            "previous_entity": previous_entity,
            "is_followup": is_followup,
            "domain_relevant": bool(entity_id),
            "resolution_reason": resolution_reason,
            "entities": [c["entity"] for c in entity_candidates],
            "entity_result": {
                "found": bool(entity_candidates),
                "status": "found" if entity_candidates else "not_found",
                "results": entity_candidates,
                "top3": entity_candidates[:3],
            },
            "subtopic": subtopic,
            "subtopic_id": subtopic_id,
            "subtopic_score": subtopic_score,
            "subtopic_match_type": subtopic_match_type,
            "query_embedding": query_embedding,
            "anchored_topic": None,
        }