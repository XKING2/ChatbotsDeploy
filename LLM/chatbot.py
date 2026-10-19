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


class ImmigrationChatbot:
    def __init__(self, retrieval_pipeline=None):
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline()
        self.language_detector = LanguageDetector()
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
        if forced_language_code is not None and previous_question:
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

        retrieval_result = self.retrieval_pipeline.retrieve(
            effective_question,
            previous_entity=previous_entity,
        )
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
        print("Raw retrieval    :", retrieval_result.get("retrieval_found", False))
        print("=" * 55)

        found = bool(retrieval_result.get("found"))
        if not found:
            # Do not send unrelated/raw semantic candidates to the LLM.
            answer = OUT_OF_DOMAIN_MESSAGES.get(
                language_code,
                OUT_OF_DOMAIN_MESSAGES["id"],
            )
            context = "Tidak ada informasi relevan yang lolos validasi retrieval."
            prompt = None
        else:
            context = format_context(retrieval_result)
            if not context.strip() or context.startswith("Tidak ada informasi"):
                answer = OUT_OF_DOMAIN_MESSAGES.get(language_code, OUT_OF_DOMAIN_MESSAGES["id"])
                found = False
                prompt = None
            else:
                prompt = build_prompt(
                    question=effective_question,
                    context=context,
                    language=language_code,
                )
                self.rate_limiter.wait()
                answer = generate_answer(prompt)

        response = {
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
            response["evaluation_contexts"] = [
                item.get("content", "")
                for item in retrieval_result.get("chunk_results", [])
                if item.get("entity_id") == analysis.get("entity_id")
            ]
            response["retrieved_score"] = (
                retrieval_result.get("best_candidate") or {}
            ).get("score")
        return response
