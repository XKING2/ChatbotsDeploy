import json
import time

from retrival.retrieval_pipeline import RetrievalPipeline
from LLM.context_formatter import format_context
from LLM.prompt_builder import build_prompt
from LLM.generator import generate_answer
from LLM.rate_limiter import RateLimiter



OUT_OF_DOMAIN_MESSAGE = (
    "Maaf, saya hanya dapat membantu pertanyaan yang informasinya tersedia dalam "
    "basis pengetahuan layanan keimigrasian.\n\n"
    "Sorry, I can only help with questions covered by the immigration-service "
    "knowledge base.\n\n"
    "申し訳ありませんが、出入国管理サービスの知識ベースにある情報に関する質問のみお答えできます。\n\n"
    "Извините, я могу помочь только с вопросами, информация по которым есть в "
    "базе знаний иммиграционной службы.\n\n"
    "抱歉，我只能回答出入境服务知识库中所包含的问题。"
)

SEARCH_ERROR_MESSAGE = (
    "Maaf, terjadi gangguan pada sistem pencarian. Silakan coba lagi beberapa saat lagi.\n\n"
    "Sorry, the search system ran into a problem. Please try again in a moment.\n\n"
    "申し訳ありません、検索システムにエラーが発生しました。しばらく経ってからもう一度お試しください。\n\n"
    "Извините, в поисковой системе произошел сбой. Пожалуйста, попробуйте еще раз чуть позже.\n\n"
    "抱歉，搜索系统出现故障。请稍后再试。"
)



EMPTY_QUESTION_MESSAGE = (
    "Silakan masukkan pertanyaan terlebih dahulu.\n\n"
    "Please enter your question first.\n\n"
    "まずご質問を入力してください。\n\n"
    "Пожалуйста, сначала введите ваш вопрос.\n\n"
    "请先输入您的问题。"
)

SHOW_SEMANTIC_SCORE_IN_ANSWER = False


def _semantic_scores(analysis):
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
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline()
        self.rate_limiter = RateLimiter(max_requests=5, period=60)

    def ask(
        self,
        question,
        previous_entity=None,
        previous_question=None,
        previous_answer=None,
        evaluation=False,
    ):
        if not isinstance(question, str) or not question.strip():
            return {
                "question": question,
                "answer": EMPTY_QUESTION_MESSAGE,
                "found": False,
                "status": "empty_question",
            }

        
        effective_question = question

        t_start = time.perf_counter()
        retrieval_result = self.retrieval_pipeline.retrieve(
            effective_question,
            previous_entity=previous_entity,
        )
        analysis = retrieval_result.get("context") or {}

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
            answer = (
                SEARCH_ERROR_MESSAGE
                if retrieval_result.get("status") == "search_error"
                else OUT_OF_DOMAIN_MESSAGE
            )
            context = "Tidak ada informasi relevan yang lolos validasi retrieval."
            prompt = None
        else:
            t_mark = time.perf_counter()
            context = format_context(retrieval_result)
            t_format = time.perf_counter() - t_mark
            if not _context_has_content(context):
                answer = OUT_OF_DOMAIN_MESSAGE
                found = False
                prompt = None
            else:
                prompt = build_prompt(
                    question=effective_question,
                    context=context,
                    previous_question=previous_question,
                    previous_answer=previous_answer,
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