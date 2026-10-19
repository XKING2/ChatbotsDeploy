import re

from retrival.entity_retrieval import EntityRetrieval
from userquery.query_normalizer import normalize_query


ENTITY_MAX_CANDIDATES = 3


class ContextUnderstanding:
    """Resolve conversation context from retrieval results.

    This class does not create query embeddings and does not run semantic search.
    RetrievalPipeline owns those responsibilities.
    """

    def __init__(self):
        self.entity_retrieval = EntityRetrieval()
        self.entities = self.entity_retrieval.get_entities() or []
        self.aliases = self.entity_retrieval.get_aliases() or []

        self.aliases_by_entity = {}
        self.entity_ids_by_name = {}

        for item in self.entities:
            entity_id = item.get("entity_id")
            name = (item.get("name") or item.get("entity") or "").strip()
            if entity_id and name:
                self.entity_ids_by_name[name] = entity_id

        for item in self.aliases:
            entity_id = item.get("entity_id")
            alias = item.get("alias")
            if entity_id and alias:
                self.aliases_by_entity.setdefault(entity_id, []).append(alias)

        # Acronyms in entity names, e.g. "VISA ON ARRIVAL (VOA)", should be
        # recognized even if the alias collection does not contain them yet.
        for item in self.entities:
            entity_id = item.get("entity_id")
            name = (item.get("name") or item.get("entity") or "").strip()
            if not entity_id or not name:
                continue
            for acronym in re.findall(r"\(([^()]{2,15})\)", name):
                acronym = acronym.strip()
                if acronym and acronym not in self.aliases_by_entity.setdefault(entity_id, []):
                    self.aliases_by_entity[entity_id].append(acronym)

    # ------------------------------------------------------------
    # Text normalization used only for rule-based matching.
    # Query embedding normalization is performed by RetrievalPipeline.
    # ------------------------------------------------------------
    def normalize(self, text):
        text = str(text or "").lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def is_contextual_followup(self, question, previous_entity=None):
        """Detect references to prior turns without flagging ordinary wording.

        Words such as "itu" in "apa itu VOA?" are not follow-up signals by
        themselves. Very short questions ("berapa?", "bagaimana?") count as
        follow-ups only when a previous entity is available.
        """
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
        # A bare demonstrative is contextual only when the user has an active
        # prior entity and is clearly referring back to it.
        has_reference = any(token in {"ini", "itu", "tersebut", "tadi"} for token in tokens)
        if previous_entity and has_reference:
            # Do not misclassify definitional questions such as "apa itu VOA?"
            if not re.match(r"^(apa|apakah) itu\b", q):
                return True

        short_followups = {
            "berapa", "berapa biayanya", "bagaimana", "bagaimana caranya",
            "kapan", "di mana", "dimana", "siapa", "apakah",
            "mengapa", "kenapa", "syaratnya apa", "kalau begitu",
        }
        return bool(previous_entity and q in short_followups)

    def find_explicit_entity(self, question):
        q_tokens = self.normalize(question).split()
        matches = []

        for entity in self.entities:
            entity_id = entity.get("entity_id")
            name = (entity.get("name") or entity.get("entity") or "").strip()
            if not entity_id or not name:
                continue

            phrases = [name] + self.aliases_by_entity.get(entity_id, [])
            for raw_phrase in phrases:
                phrase = self.normalize(raw_phrase)
                if not phrase:
                    continue
                tokens = phrase.split()
                if len(tokens) > len(q_tokens):
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

    def find_explicit_subtopic(self, question, entity_id):
        if not entity_id:
            return None, None, 0.0, None

        q = self.normalize(question)
        query_tokens = set(q.split())
        subtopics = self.entity_retrieval.get_subtopics(entity_id=entity_id) or []
        matches = []

        # Ignore generic words for token-set matching so labels like "Jumlah
        # Pelaksana" do not match merely because one common word appears.
        generic_tokens = {
            "apa", "itu", "yang", "dan", "atau", "untuk", "dengan",
            "dari", "di", "ke", "pada", "adalah", "tentang", "bagaimana",
            "syarat", "persyaratan", "informasi", "layanan",
        }

        for item in subtopics:
            name = item.get("subtopic")
            subtopic_id = item.get("subtopic_id")
            if not name or not subtopic_id:
                continue

            phrase = self.normalize(name)
            if not phrase:
                continue

            if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", q):
                matches.append({
                    "subtopic": name,
                    "subtopic_id": subtopic_id,
                    "score": 1.0,
                    "match_type": "explicit",
                    "matched_phrase": phrase,
                })
                continue

            phrase_tokens = set(phrase.split())
            meaningful_tokens = phrase_tokens - generic_tokens
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
        return best["subtopic"], best["subtopic_id"], best["score"], best["match_type"]

    def resolve_entity(
        self,
        explicit_entity,
        ambiguous_candidates,
        semantic_entity,
        previous_entity,
        is_followup,
    ):
        if explicit_entity:
            return explicit_entity

        if ambiguous_candidates:
            if semantic_entity and semantic_entity in ambiguous_candidates:
                return semantic_entity
            if is_followup and previous_entity and previous_entity in ambiguous_candidates:
                return previous_entity
            return None

        if is_followup and previous_entity:
            return previous_entity

        # A repeated entity in a new, non-follow-up question is still valid;
        # do not discard it merely because it equals previous_entity.
        return semantic_entity

    def entity_id(self, entity):
        if not entity:
            return None
        return self.entity_ids_by_name.get(entity)

    @staticmethod
    def _candidate_entity_name(item):
        return item.get("entity") or item.get("name")

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
        """Resolve context from retrieval results supplied by RetrievalPipeline.

        This method does not create embeddings or run semantic search. It also
        avoids blindly assigning an entity to every out-of-domain question.
        """
        normalized_question = normalized_question or normalize_query(question)
        entity_results = entity_results or []
        subtopic_results = subtopic_results or []
        chunk_results = chunk_results or []

        is_followup = self.is_contextual_followup(normalized_question, previous_entity)
        explicit_entity, ambiguous = self.find_explicit_entity(normalized_question)

        top_entity = entity_results[0] if entity_results else None
        semantic_entity = self._candidate_entity_name(top_entity) if top_entity else None
        semantic_score = top_entity.get("score") if top_entity else None
        second_score = (
            entity_results[1].get("score")
            if len(entity_results) > 1
            else None
        )
        score_margin = (
            float(semantic_score) - float(second_score)
            if semantic_score is not None and second_score is not None
            else None
        )

        # Strong lexical evidence: the query contains a distinctive word from
        # the candidate entity name or one of its aliases. Generic question
        # words are deliberately excluded (e.g. "siapa", "apa", "sekarang").
        stopwords = {
            "apa", "apakah", "siapa", "kapan", "dimana", "mana", "di",
            "ke", "dari", "yang", "dan", "atau", "itu", "ini", "saya",
            "aku", "mau", "ingin", "tolong", "dong", "gimana",
            "bagaimana", "caranya", "cara", "sekarang", "tentang",
            "untuk", "dengan", "persyaratan", "syarat", "berapa",
            "adalah", "pengertian", "jelaskan", "please", "what",
            "are", "the", "for", "how", "do", "i", "my", "is",
            "can", "could", "would", "tell", "me", "requirements",
        }
        query_terms = set(self.normalize(normalized_question).split()) - stopwords

        candidate_terms = set()
        candidate_id = top_entity.get("entity_id") if top_entity else None
        if candidate_id:
            for item in self.entities:
                if item.get("entity_id") == candidate_id:
                    candidate_name = item.get("name") or item.get("entity") or ""
                    candidate_terms.update(self.normalize(candidate_name).split())
                    break
            for alias in self.aliases_by_entity.get(candidate_id, []):
                candidate_terms.update(self.normalize(alias).split())
        candidate_terms -= stopwords
        lexical_anchor = bool(query_terms & candidate_terms)

        # A semantic-only match needs either a domain-specific lexical anchor,
        # or sufficiently strong confidence plus a clear lead over rank 2.
        # Explicit entity mentions and valid contextual follow-ups bypass this
        # abstention check. Values are conservative starting points to tune
        # against a labelled test set, not universal similarity thresholds.
        semantic_confident = (
            semantic_score is not None
            and float(semantic_score) >= 0.65
            and (score_margin is None or score_margin >= 0.04)
        )
        semantic_accepted = bool(
            semantic_entity and (lexical_anchor or semantic_confident)
        )

        # A valid service/topic can be retrieved even when its parent entity
        # has a generic name. Example: "Apa itu GCI?" retrieves the subtopic
        # "Pengertian Global Citizen of Indonesia atau GCI", while the parent
        # entity is only "Layanan Keimigrasian bagi Warga Negara Asing".
        # Use aligned subtopic + chunk evidence instead of rejecting it based
        # only on the parent's entity similarity.
        generic_topic_tokens = {
            "apa", "apakah", "itu", "yang", "dan", "atau", "untuk", "dengan",
            "dari", "di", "ke", "pada", "adalah", "tentang", "pengertian",
            "jelaskan", "maksud", "syarat", "persyaratan", "informasi",
            "layanan", "sekarang", "siapa", "kapan", "berapa", "bagaimana",
            "indonesia",
        }
        query_tokens = set(self.normalize(normalized_question).split()) - generic_topic_tokens
        anchored_topic = None
        for topic in subtopic_results:
            topic_name = topic.get("subtopic") or ""
            topic_entity_id = topic.get("entity_id")
            topic_score = float(topic.get("score") or 0.0)
            if not topic_entity_id or not topic_name or topic_score < 0.60:
                continue

            topic_tokens = set(self.normalize(topic_name).split()) - generic_topic_tokens
            if not (query_tokens & topic_tokens):
                continue

            matching_chunks = [
                chunk for chunk in chunk_results
                if chunk.get("entity_id") == topic_entity_id
                and chunk.get("subtopic_id") == topic.get("subtopic_id")
                and float(chunk.get("score") or 0.0) >= 0.60
            ]
            if matching_chunks:
                anchored_topic = topic
                break

        if explicit_entity:
            resolved_semantic_entity = explicit_entity
        elif is_followup and previous_entity:
            resolved_semantic_entity = semantic_entity
        elif anchored_topic:
            resolved_semantic_entity = anchored_topic.get("entity")
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
        elif entity_id:
            matching = [
                item for item in subtopic_results
                if item.get("entity_id") == entity_id
            ]
            if matching:
                best_subtopic = max(
                    matching, key=lambda item: float(item.get("score") or 0.0)
                )
                selected_subtopic = best_subtopic.get("subtopic")
                selected_subtopic_id = best_subtopic.get("subtopic_id")
                selected_subtopic_score = best_subtopic.get("score")
                subtopic_match_type = "semantic"

        entity_result = {
            "found": bool(entity_results),
            "status": "found" if entity_results else "not_found",
            "results": entity_results,
            "top3": entity_results[:3],
        }

        # Domain relevance means context resolution accepted an entity; it is
        # intentionally distinct from whether ChromaDB returned any candidates.
        domain_relevant = bool(entity_id)
        resolution_reason = (
            "explicit_entity" if explicit_entity
            else "follow_up_context" if is_followup and previous_entity and entity
            else "subtopic_chunk_anchor" if anchored_topic and entity
            else "semantic_with_lexical_anchor" if semantic_accepted and lexical_anchor
            else "semantic_confident" if semantic_accepted
            else "ambiguous_entity" if ambiguous
            else "insufficient_relevance_evidence" if entity_results
            else "no_entity_candidates"
        )

        return {
            "question": question,
            "normalized_question": normalized_question,
            "entity": entity,
            "entity_id": entity_id,
            "detected_entity": semantic_entity,
            "detected_score": semantic_score,
            "entity_score_margin": score_margin,
            "explicit_entity": explicit_entity,
            "ambiguous_candidates": ambiguous,
            "previous_entity": previous_entity,
            "is_followup": is_followup,
            "domain_relevant": domain_relevant,
            "resolution_reason": resolution_reason,
            "entities": [self._candidate_entity_name(item) for item in entity_results],
            "entity_result": entity_result,
            "subtopic": selected_subtopic,
            "subtopic_id": selected_subtopic_id,
            "subtopic_score": selected_subtopic_score,
            "subtopic_match_type": subtopic_match_type,
            "query_embedding": query_embedding,
        }

