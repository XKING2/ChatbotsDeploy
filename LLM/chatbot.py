import json
import time

from retrival.context_understanding import ContextUnderstanding
from retrival.retrieval_pipeline import RetrievalPipeline
from retrival.language_switch import detect_language_switch_request
from LLM.context_formatter import format_context
from LLM.prompt_builder import build_prompt, LANGUAGE_NAMES
from LLM.generator import generate_answer
from LLM.rate_limiter import RateLimiter
from retrival.language_detector import LanguageDetector


OUT_OF_DOMAIN_MESSAGES = {
    "id": "Maaf, saya hanya dapat membantu pertanyaan yang informasinya tersedia dalam basis pengetahuan layanan keimigrasian.",
    "en": "Sorry, I can only help with questions covered by the immigration-service knowledge base.",
    "ru": "Извините, я могу отвечать только на вопросы, информация о которых есть в базе знаний иммиграционной службы.",
    "ja": "申し訳ありませんが、入国管理サービスのナレッジベースに情報がある質問のみお手伝いできます。",
    "ko": "죄송합니다. 출입국 서비스 지식 베이스에 정보가 있는 질문에 대해서만 답변할 수 있습니다.",
    "zh": "抱歉，我只能回答移民服务知识库中包含相关信息的问题。",
    "zh-cn": "抱歉，我只能回答移民服务知识库中包含相关信息的问题。",
    "zh-tw": "抱歉，我只能回答移民服務知識庫中包含相關資訊的問題。",
    "hi": "क्षमा करें, मैं केवल उन प्रश्नों में मदद कर सकता हूँ जिनकी जानकारी आव्रजन सेवा ज्ञान आधार में उपलब्ध है。",
}

# Dipakai saat pencarian gagal secara teknis (bukan pertanyaan di luar domain).
SEARCH_ERROR_MESSAGES = {
    "id": "Maaf, terjadi gangguan pada sistem pencarian. Silakan coba lagi beberapa saat lagi.",
    "en": "Sorry, the search system ran into a problem. Please try again in a moment.",
}


# Dipakai saat pengguna meminta ganti bahasa tetapi belum ada pertanyaan sebelumnya.
NO_PREVIOUS_QUESTION_MESSAGES = {
    "id": "Silakan ajukan pertanyaan Anda terlebih dahulu, nanti saya jawab dalam bahasa yang Anda pilih.",
    "en": "Please ask your question first, and I will answer it in the language you chose.",
    "ru": "Пожалуйста, сначала задайте вопрос, и я отвечу на выбранном вами языке.",
    "ja": "まずご質問をお知らせください。選ばれた言語でお答えします。",
    "ko": "먼저 질문을 해 주세요. 선택하신 언어로 답변해 드리겠습니다.",
    "zh": "请先提出您的问题，我会用您选择的语言回答。",
}


# True = baris "Score semantic" juga ditambahkan di akhir teks jawaban.
# False = skor hanya tampil di log dan ada di response["semantic_score"].
SHOW_SEMANTIC_SCORE_IN_ANSWER = False


def _semantic_scores(analysis):
    """Score semantic entity terpilih dan subtopic terpilih (None bila tidak ada)."""
    entity_id = analysis.get("entity_id")
    entity_score = next(
        (
            item.get("score")
            for item in (analysis.get("entity_result") or {}).get("results", [])
            if entity_id and item.get("entity_id") == entity_id
        ),
        None,
    )
    return {
        "entity": entity_score,
        "subtopic": analysis.get("subtopic_score"),
    }


def _fmt_score(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def _context_has_content(context):
    """format_context mengembalikan JSON; cek isi `found` dan `contexts`."""
    try:
        payload = json.loads(context)
    except (TypeError, ValueError):
        return bool((context or "").strip())
    return bool(payload.get("found")) and bool(payload.get("contexts"))


class ImmigrationChatbot:
    def __init__(self, retrieval_pipeline=None):
        # Satu detector dipakai bersama supaya model bahasa tidak dimuat dua kali.
        self.language_detector = LanguageDetector()
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(
            context_engine=ContextUnderstanding(
                language_detector=self.language_detector
            )
        )
        self.rate_limiter = RateLimiter(max_requests=5, period=60)

    @staticmethod
    def _language_code(language):
        code = (language or {}).get("language_code") or "id"
        return code.lower().strip()

    def ask(
        self,
        question,
        previous_entity=None,
        previous_question=None,
        evaluation=False,
    ):
        if not isinstance(question, str) or not question.strip():
            return {
                "question": question,
                "answer": "Silakan masukkan pertanyaan terlebih dahulu.",
                "found": False,
                "status": "empty_question",
            }

        forced_language_code = detect_language_switch_request(question)
        if forced_language_code is not None and not previous_question:
            # Tidak ada pertanyaan sebelumnya untuk diterjemahkan. Jangan
            # mencari "please use english" ke basis pengetahuan.
            return {
                "question": question,
                "effective_question": question,
                "answer": NO_PREVIOUS_QUESTION_MESSAGES.get(
                    forced_language_code, NO_PREVIOUS_QUESTION_MESSAGES["en"]
                ),
                "found": False,
                "status": "language_switch_without_context",
                "language_switch": forced_language_code,
            }

        if forced_language_code is not None:
            effective_question = previous_question
        else:
            effective_question = question

        if forced_language_code:
            language = {
                "language": LANGUAGE_NAMES.get(forced_language_code, forced_language_code),
                "language_code": forced_language_code,
                "confidence": 1.0,
            }
        else:
            language = self.language_detector.detect(effective_question) or {}
            language.setdefault("language_code", "id")
            language.setdefault("language", LANGUAGE_NAMES.get(language["language_code"], "Bahasa Indonesia"))
            language.setdefault("confidence", 0.0)

        t_start = time.perf_counter()
        retrieval_result = self.retrieval_pipeline.retrieve(
            effective_question,
            previous_entity=previous_entity,
            language_info=language,
        )
        t_retrieval = time.perf_counter() - t_start
        analysis = retrieval_result.get("context") or {}
        language_code = self._language_code(language)

        print("=" * 55)
        print("USER QUESTION")
        print("=" * 55)
        print("Pertanyaan       :", question)
        print("Effective query  :", effective_question)
        print("Retrieval status :", retrieval_result.get("status"))
        print("Found            :", retrieval_result.get("found"))
        print("Entity           :", analysis.get("entity") or "-")
        print("Subtopic         :", analysis.get("subtopic") or "-")
        semantic_score = _semantic_scores(analysis)
        print("Score semantic   : entity =", _fmt_score(semantic_score["entity"]),
              "| subtopic =", _fmt_score(semantic_score["subtopic"]))
        print("Raw retrieval    :", retrieval_result.get("retrieval_found", False))
        for error in retrieval_result.get("search_errors") or []:
            print("Search error     :", error)
        print("=" * 55)

        t_format = t_rate_wait = t_llm = 0.0
        found = bool(retrieval_result.get("found"))
        if not found:
            # Do not send unrelated/raw semantic candidates to the LLM.
            if retrieval_result.get("status") == "search_error":
                answer = SEARCH_ERROR_MESSAGES.get(
                    language_code, SEARCH_ERROR_MESSAGES["en"]
                    if language_code != "id" else SEARCH_ERROR_MESSAGES["id"]
                )
            else:
                answer = OUT_OF_DOMAIN_MESSAGES.get(
                    language_code,
                    OUT_OF_DOMAIN_MESSAGES["id"],
                )
            context = "Tidak ada informasi relevan yang lolos validasi retrieval."
            prompt = None
        else:
            t_mark = time.perf_counter()
            context = format_context(retrieval_result)
            t_format = time.perf_counter() - t_mark
            if not _context_has_content(context):
                answer = OUT_OF_DOMAIN_MESSAGES.get(language_code, OUT_OF_DOMAIN_MESSAGES["id"])
                found = False
                prompt = None
            else:
                prompt = build_prompt(
                    question=effective_question,
                    context=context,
                    language=language_code,
                )
                t_mark = time.perf_counter()
                self.rate_limiter.wait()
                t_rate_wait = time.perf_counter() - t_mark

                t_mark = time.perf_counter()
                answer = generate_answer(prompt)
                t_llm = time.perf_counter() - t_mark

        timings = dict(retrieval_result.get("timings") or {})
        timings.update({
            "rate_limiter_wait": t_rate_wait,
            "llm_generate": t_llm,
            "format_context": t_format,
            "total": time.perf_counter() - t_start,
        })
        print("TIMINGS (detik)  :", {k: round(v, 2) for k, v in timings.items()})
        print("Prompt           :", len(prompt or ""), "karakter")

        if found and SHOW_SEMANTIC_SCORE_IN_ANSWER:
            answer = (
                f"{answer}\n\nScore semantic: entity {_fmt_score(semantic_score['entity'])}"
                f" | subtopic {_fmt_score(semantic_score['subtopic'])}"
            )

        response = {
            "semantic_score": semantic_score,
            "timings": timings,
            "question": question,
            "effective_question": effective_question,
            "analysis": analysis,
            "retrieval": retrieval_result,
            "context": context,
            "prompt": prompt,
            "answer": answer,
            "language": language,
            "entity": analysis.get("entity"),
            "retrieved_entity": (retrieval_result.get("best_candidate") or {}).get("entity"),
            "retrieved_entity_id": (retrieval_result.get("best_candidate") or {}).get("entity_id"),
            "retrieved_subtopic": (retrieval_result.get("best_candidate") or {}).get("subtopic"),
            "retrieved_subtopic_id": (retrieval_result.get("best_candidate") or {}).get("subtopic_id"),
            "found": found,
            "status": retrieval_result.get("status"),
        }

        if evaluation:
            # Persis chunk yang dikirim ke LLM (bukan top-K pencarian mentah).
            response["evaluation_contexts"] = [
                item.get("content", "")
                for item in retrieval_result.get("relevant_chunks", [])
            ]
            response["retrieved_score"] = (
                retrieval_result.get("best_candidate") or {}
            ).get("score")
        return response