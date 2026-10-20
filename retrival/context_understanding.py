import re
from typing import Any, Dict, List, Optional, Tuple

from retrival.entity_retrieval import EntityRetrieval
from userquery.query_normalizer import normalize_query
from retrival.language_detector import LanguageDetector


ENTITY_SEMANTIC_MIN_SCORE = 0.65
ENTITY_MIN_SCORE_MARGIN = 0.00
SUBTOPIC_MIN_SCORE = 0.60
CHUNK_MIN_SCORE = 0.55


class ContextUnderstanding:


    def __init__(self, language_detector=None):
        self.entity_retrieval = EntityRetrieval()
        # Reuse the project's Lingua detector when supplied by ImmigrationChatbot.
        self.language_detector = language_detector or LanguageDetector()
        self.entities = self.entity_retrieval.get_entities() or []
        self.aliases = self.entity_retrieval.get_aliases() or []

        self.aliases_by_entity: Dict[str, List[str]] = {}
        self.entity_ids_by_name: Dict[str, str] = {}
        self.entity_names_by_id: Dict[str, str] = {}

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
                aliases = self.aliases_by_entity.setdefault(entity_id, [])
                if alias not in aliases:
                    aliases.append(alias)

        # Automatically recognize acronyms explicitly written in entity names.
        for item in self.entities:
            entity_id = item.get("entity_id")
            name = (item.get("name") or item.get("entity") or "").strip()
            if not entity_id or not name:
                continue
            for acronym in re.findall(r"\(([^()]{2,15})\)", name):
                acronym = acronym.strip()
                aliases = self.aliases_by_entity.setdefault(entity_id, [])
                if acronym and acronym not in aliases:
                    aliases.append(acronym)

    @staticmethod
    def normalize(text: Any) -> str:
        """Normalize text for rule-based matching while preserving Unicode words."""
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

    @staticmethod
    def _candidate_entity_name(item: Optional[Dict[str, Any]]) -> Optional[str]:
        if not item:
            return None
        return item.get("entity") or item.get("name")

    def _entity_name_for_result(self, result: Dict[str, Any]) -> Optional[str]:
        """Resolve an entity result even when it only supplies entity_id."""
        name = self._candidate_entity_name(result)
        if name:
            return name
        return self.entity_names_by_id.get(result.get("entity_id"))

    def is_contextual_followup(
        self, question: str, previous_entity: Optional[str] = None
    ) -> bool:
        """Detect clear references to previous turns, not ordinary 'apa itu X?'."""
        q = self.normalize(question)
        if not q:
            return False

        strong_phrases = (
            "yang tadi", "yang sebelumnya", "yang dimaksud",
            "yang tersebut", "yang itu", "hal tadi", "hal tersebut",
            "seperti tadi", "seperti sebelumnya", "tadi bagaimana",
            "kalau begitu", "lalu bagaimana",
        )
        if any(phrase in q for phrase in strong_phrases):
            return True

        tokens = q.split()
        has_reference = any(
            token in {"ini", "itu", "tersebut", "tadi"} for token in tokens
        )
        if previous_entity and has_reference:
            if not re.match(r"^(apa|apakah) itu\b", q):
                return True

        short_followups = {
            "berapa", "berapa biayanya", "bagaimana", "bagaimana caranya",
            "kapan", "di mana", "dimana", "siapa", "apakah", "mengapa",
            "kenapa", "syaratnya apa", "kalau begitu",
        }
        return bool(previous_entity and q in short_followups)

    def find_explicit_entity(
        self, question: str
    ) -> Tuple[Optional[str], List[str]]:
        """Match entity names/aliases as whole token sequences, including Unicode."""
        q_tokens = self.normalize(question).split()
        matches = []

        for item in self.entities:
            entity_id = item.get("entity_id")
            name = (item.get("name") or item.get("entity") or "").strip()
            if not entity_id or not name:
                continue

            phrases = [name] + self.aliases_by_entity.get(entity_id, [])
            for raw_phrase in phrases:
                phrase = self.normalize(raw_phrase)
                tokens = phrase.split()
                if not tokens or len(tokens) > len(q_tokens):
                    continue
                for i in range(len(q_tokens) - len(tokens) + 1):
                    if q_tokens[i:i + len(tokens)] == tokens:
                        matches.append({
                            "name": name,
                            "entity_id": entity_id,
                            "matched_phrase": phrase,
                            "length": len(tokens),
                        })

        if not matches:
            return None, []

        longest = max(item["length"] for item in matches)
        best_matches = [item for item in matches if item["length"] == longest]
        unique_entities = {item["entity_id"]: item for item in best_matches}

        if len(unique_entities) == 1:
            best = next(iter(unique_entities.values()))
            return best["name"], []

        return None, sorted(item["name"] for item in unique_entities.values())

    def find_explicit_subtopic(
        self, question: str, entity_id: Optional[str]
    ) -> Tuple[Optional[str], Optional[str], float, Optional[str]]:
        if not entity_id:
            return None, None, 0.0, None

        q = self.normalize(question)
        query_tokens = set(q.split())
        subtopics = self.entity_retrieval.get_subtopics(entity_id=entity_id) or []
        generic_tokens = {
            "apa", "apakah", "itu", "yang", "dan", "atau", "untuk", "dengan",
            "dari", "di", "ke", "pada", "adalah", "tentang", "bagaimana",
            "syarat", "persyaratan", "informasi", "layanan", "pengertian",
            "jelaskan", "maksud", "indonesia",
        }
        matches = []

        for item in subtopics:
            name = (item.get("subtopic") or "").strip()
            subtopic_id = item.get("subtopic_id")
            if not name or not subtopic_id:
                continue

            phrase = self.normalize(name)
            if not phrase:
                continue

            # Exact phrase matching is language-agnostic for aliases stored in
            # the subtopic name, including Chinese/Japanese/etc. Unicode text.
            if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", q):
                matches.append({
                    "subtopic": name,
                    "subtopic_id": subtopic_id,
                    "score": 1.0,
                    "match_type": "explicit",
                    "matched_phrase": phrase,
                })
                continue

            meaningful_tokens = set(phrase.split()) - generic_tokens
            if meaningful_tokens and meaningful_tokens.issubset(query_tokens):
                matches.append({
                    "subtopic": name,
                    "subtopic_id": subtopic_id,
                    "score": 0.95,
                    "match_type": "explicit_token_match",
                    "matched_phrase": phrase,
                })

        if not matches:
            return None, None, 0.0, None

        matches.sort(
            key=lambda item: (item["score"], len(item["matched_phrase"])),
            reverse=True,
        )
        best = matches[0]
        return (
            best["subtopic"], best["subtopic_id"],
            best["score"], best["match_type"],
        )

    def _is_alias_only_match(self, question: str, entity_name: Optional[str]) -> bool:
        """True when the entity was matched via a short alias/acronym, not its full name."""
        if not entity_name:
            return False
        q = " " + self.normalize(question) + " "
        full = " " + self.normalize(entity_name) + " "
        return full not in q

    @staticmethod
    def _top_chunk_entity_id(chunk_results: List[Dict[str, Any]]) -> Optional[str]:
        best, best_score = None, None
        for chunk in chunk_results:
            score = ContextUnderstanding._score(chunk)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best, best_score = chunk.get("entity_id"), score
        return best

    def resolve_entity(
        self,
        explicit_entity: Optional[str],
        ambiguous_candidates: List[str],
        semantic_entity: Optional[str],
        previous_entity: Optional[str],
        is_followup: bool,
    ) -> Optional[str]:
        if explicit_entity:
            return explicit_entity

        if ambiguous_candidates:
            if semantic_entity and semantic_entity in ambiguous_candidates:
                return semantic_entity
            if (
                is_followup
                and previous_entity
                and previous_entity in ambiguous_candidates
            ):
                return previous_entity
            return None

        if is_followup and previous_entity:
            return previous_entity

        return semantic_entity

    def entity_id(self, entity: Optional[str]) -> Optional[str]:
        if not entity:
            return None
        entity_id = self.entity_ids_by_name.get(entity)
        if entity_id:
            return entity_id
        # Candidate names can differ in surrounding whitespace/case.
        normalized = self.normalize(entity)
        for name, candidate_id in self.entity_ids_by_name.items():
            if self.normalize(name) == normalized:
                return candidate_id
        return None

    def _get_candidate_terms(self, entity_id: Optional[str]) -> set:
        terms = set()
        if not entity_id:
            return terms

        name = self.entity_names_by_id.get(entity_id, "")
        terms.update(self.normalize(name).split())
        for alias in self.aliases_by_entity.get(entity_id, []):
            terms.update(self.normalize(alias).split())
        return terms

    def _aligned_chunk_evidence(
        self, topic: Dict[str, Any], chunk_results: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[float]]:
        """Require chunk evidence aligned to the same entity and, when present, topic."""
        topic_entity_id = topic.get("entity_id")
        topic_subtopic_id = topic.get("subtopic_id")
        best_score = None

        for chunk in chunk_results:
            if chunk.get("entity_id") != topic_entity_id:
                continue

            chunk_subtopic_id = chunk.get("subtopic_id")
            if topic_subtopic_id and chunk_subtopic_id != topic_subtopic_id:
                continue

            score = self._score(chunk)
            if score is None or score < CHUNK_MIN_SCORE:
                continue
            if best_score is None or score > best_score:
                best_score = score

        return best_score is not None, best_score

    def analyze(
        self,
        question,
        previous_entity=None,
        normalized_question=None,
        query_embedding=None,
        entity_results=None,
        subtopic_results=None,
        chunk_results=None,
        language_info=None,
    ):
        """Resolve context from retrieval candidates and the detected query language."""
        normalized_question = normalized_question or normalize_query(question)
        language_info = language_info or self.language_detector.detect(normalized_question)
        language_code = language_info.get("language_code", "unknown")
        language_status = language_info.get("status", "unknown")
        language_confidence = language_info.get("confidence", 0.0)
        entity_results = list(entity_results or [])
        subtopic_results = list(subtopic_results or [])
        chunk_results = list(chunk_results or [])

        is_followup = self.is_contextual_followup(
            normalized_question, previous_entity
        )
        explicit_entity, ambiguous = self.find_explicit_entity(
            normalized_question
        )

        # Normalize entity candidate names and score values. Do not assume the
        # result always includes a display-name field if entity_id is present.
        candidates = []
        for result in entity_results:
            name = self._entity_name_for_result(result)
            score = self._score(result)
            if name:
                normalized_result = dict(result)
                normalized_result["entity"] = name
                normalized_result["score"] = score
                candidates.append(normalized_result)

        candidates.sort(
            key=lambda item: (
                item.get("score") if item.get("score") is not None else -1.0
            ),
            reverse=True,
        )
        top_entity = candidates[0] if candidates else None
        semantic_entity = self._candidate_entity_name(top_entity)
        semantic_score = self._score(top_entity)
        second_score = self._score(candidates[1]) if len(candidates) > 1 else None
        score_margin = (
            semantic_score - second_score
            if semantic_score is not None and second_score is not None
            else None
        )

        stopwords = {
            "apa", "apakah", "siapa", "kapan", "dimana", "mana", "di", "ke",
            "dari", "yang", "dan", "atau", "itu", "ini", "saya", "aku", "mau",
            "ingin", "tolong", "dong", "gimana", "bagaimana", "caranya", "cara",
            "sekarang", "tentang", "untuk", "dengan", "persyaratan", "syarat",
            "berapa", "adalah", "pengertian", "jelaskan", "please", "what",
            "are", "the", "for", "how", "do", "i", "my", "is", "can", "could",
            "would", "tell", "me", "requirements",
        }
        query_terms = set(self.normalize(normalized_question).split()) - stopwords
        candidate_id = top_entity.get("entity_id") if top_entity else None
        lexical_anchor = bool(
            query_terms & (self._get_candidate_terms(candidate_id) - stopwords)
        )

        semantic_confident = (
            semantic_score is not None
            and semantic_score >= ENTITY_SEMANTIC_MIN_SCORE
            and (
                score_margin is None
                or score_margin >= ENTITY_MIN_SCORE_MARGIN
            )
        )
        # For Chinese/Japanese/Russian, entity labels in the KB may be Indonesian
        # or English, so token overlap is not reliable cross-language evidence.
        multilingual_query = language_code in {"zh", "ja", "ru"}
        semantic_accepted = bool(
            semantic_entity and (
                lexical_anchor
                or semantic_confident
            )
        )

        # Key multilingual fix:
        # Do not require token overlap between the user language and the KB
        # language. A semantically strong subtopic AND a matching chunk from
        # that exact topic are sufficient corroborating evidence. This allows
        # a Mandarin query to retrieve an Indonesian/English Golden Visa topic,
        # while still requiring two aligned retrieval signals.
        topic_evidence = []
        for topic in subtopic_results:
            topic_name = (topic.get("subtopic") or "").strip()
            topic_entity_id = topic.get("entity_id")
            topic_score = self._score(topic)
            topic_threshold = SUBTOPIC_MIN_SCORE
            if multilingual_query:
                topic_threshold = max(SUBTOPIC_MIN_SCORE, 0.60)
            if (
                not topic_name
                or not topic_entity_id
                or topic_score is None
                or topic_score < topic_threshold
            ):
                continue

            has_chunk, chunk_score = self._aligned_chunk_evidence(
                topic, chunk_results
            )
            if has_chunk:
                topic_evidence.append({
                    **topic,
                    "subtopic": topic_name,
                    "score": topic_score,
                    "_aligned_chunk_score": chunk_score,
                })

        topic_evidence.sort(
            key=lambda item: (
                item.get("score", 0.0),
                item.get("_aligned_chunk_score", 0.0),
            ),
            reverse=True,
        )
        anchored_topic = topic_evidence[0] if topic_evidence else None

        # An acronym alias (e.g. "kitas" from "(KITAS)") is a weak, single-word
        # signal: it cannot tell "buat kitas" from "perpanjang kitas". When the
        # top entity AND the top chunk both agree on a different entity, trust
        # that agreement instead of the bare alias. Full-name matches and
        # multi-entity ambiguity are untouched, and no score threshold is used.
        alias_overridden = False
        if (
            explicit_entity
            and not ambiguous
            and semantic_entity
            and semantic_entity != explicit_entity
            and self._is_alias_only_match(normalized_question, explicit_entity)
            and self._top_chunk_entity_id(chunk_results) == (
                top_entity.get("entity_id") if top_entity else None
            )
            and (top_entity or {}).get("entity_id") is not None
        ):
            alias_overridden = True
            overridden_explicit = explicit_entity
            explicit_entity = None

        if alias_overridden:
            resolved_semantic_entity = semantic_entity
            semantic_accepted = True
        elif explicit_entity:
            resolved_semantic_entity = explicit_entity
        elif is_followup and previous_entity:
            # Preserve prior context for follow-ups; semantic retrieval may
            # still supply the current topic below.
            resolved_semantic_entity = semantic_entity or previous_entity
        elif anchored_topic:
            # Parent entity is the KB entity; the topic is not promoted into a
            # new entity (e.g. Golden Visa remains a subtopic of immigration services).
            resolved_semantic_entity = (
                anchored_topic.get("entity")
                or self.entity_names_by_id.get(anchored_topic.get("entity_id"))
                or semantic_entity
            )
            semantic_accepted = True
        elif semantic_accepted:
            resolved_semantic_entity = semantic_entity
        else:
            resolved_semantic_entity = None

        entity = self.resolve_entity(
            explicit_entity=explicit_entity,
            ambiguous_candidates=ambiguous,
            semantic_entity=resolved_semantic_entity,
            previous_entity=previous_entity,
            is_followup=is_followup,
        )
        entity_id = self.entity_id(entity)

        explicit_subtopic, explicit_subtopic_id, explicit_subtopic_score, explicit_match_type = (
            self.find_explicit_subtopic(normalized_question, entity_id)
            if entity_id
            else (None, None, 0.0, None)
        )

        selected_subtopic = None
        selected_subtopic_id = None
        selected_subtopic_score = None
        subtopic_match_type = None

        if explicit_subtopic:
            selected_subtopic = explicit_subtopic
            selected_subtopic_id = explicit_subtopic_id
            selected_subtopic_score = explicit_subtopic_score
            subtopic_match_type = explicit_match_type
        elif anchored_topic and anchored_topic.get("entity_id") == entity_id:
            selected_subtopic = anchored_topic.get("subtopic")
            selected_subtopic_id = anchored_topic.get("subtopic_id")
            selected_subtopic_score = anchored_topic.get("score")
            subtopic_match_type = "semantic_subtopic_chunk"
        elif entity_id:
            matching_topics = [
                item for item in subtopic_results
                if item.get("entity_id") == entity_id
            ]
            if matching_topics:
                best_subtopic = max(
                    matching_topics,
                    key=lambda item: (
                        self._score(item)
                        if self._score(item) is not None else -1.0
                    ),
                )
                selected_subtopic = best_subtopic.get("subtopic")
                selected_subtopic_id = best_subtopic.get("subtopic_id")
                selected_subtopic_score = self._score(best_subtopic)
                subtopic_match_type = "semantic"

        entity_result = {
            "found": bool(candidates),
            "status": "found" if candidates else "not_found",
            "results": candidates,
            "top3": candidates[:3],
        }

        domain_relevant = bool(entity_id)
        if alias_overridden:
            resolution_reason = "semantic_overrides_alias"
        elif explicit_entity:
            resolution_reason = "explicit_entity"
        elif is_followup and previous_entity and entity:
            resolution_reason = "follow_up_context"
        elif anchored_topic and entity:
            resolution_reason = "subtopic_chunk_anchor"
        elif semantic_accepted and lexical_anchor:
            resolution_reason = "semantic_with_lexical_anchor"
        elif semantic_accepted:
            resolution_reason = "semantic_confident"
        elif ambiguous:
            resolution_reason = "ambiguous_entity"
        elif candidates:
            resolution_reason = "insufficient_relevance_evidence"
        else:
            resolution_reason = "no_entity_candidates"

        return {
            "question": question,
            "normalized_question": normalized_question,
            "entity": entity,
            "entity_id": entity_id,
            "detected_entity": semantic_entity,
            "detected_score": semantic_score,
            "entity_score_margin": score_margin,
            "explicit_entity": explicit_entity or (
                overridden_explicit if alias_overridden else None
            ),
            "alias_overridden": alias_overridden,
            "ambiguous_candidates": ambiguous,
            "previous_entity": previous_entity,
            "is_followup": is_followup,
            "domain_relevant": domain_relevant,
            "resolution_reason": resolution_reason,
            "entities": [self._candidate_entity_name(item) for item in candidates],
            "entity_result": entity_result,
            "subtopic": selected_subtopic,
            "subtopic_id": selected_subtopic_id,
            "subtopic_score": selected_subtopic_score,
            "subtopic_match_type": subtopic_match_type,
            "query_embedding": query_embedding,
            "language": language_info.get("language", "Unknown"),
            "language_code": language_code,
            "language_confidence": language_confidence,
            "language_status": language_status,
            "multilingual_query": multilingual_query,
            # Useful for diagnostics and regression tests.
            "anchored_topic": (
                {
                    "subtopic": anchored_topic.get("subtopic"),
                    "subtopic_id": anchored_topic.get("subtopic_id"),
                    "score": anchored_topic.get("score"),
                    "aligned_chunk_score": anchored_topic.get("_aligned_chunk_score"),
                }
                if anchored_topic else None
            ),
        }