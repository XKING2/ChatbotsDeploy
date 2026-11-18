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
    """Skor cosine terbaik heading terpilih dan skor cosine subheading."""
    heading_id = analysis.get("heading_id")
    heading_score = next(
        (
            item.get("best_cosine")
            for item in (analysis.get("heading_result") or {}).get("results", [])
            if heading_id and item.get("heading_id") == heading_id
        ),
        None,
    )
    return {
        "heading": heading_score,
        "subheading": analysis.get("subheading_score"),
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
        previous_heading=None,
        previous_question=None,
        previous_answer=None,
        evaluation=False,
        previous_heading_id=None,
        previous_entity=None,  # nama lama, dipertahankan agar pemanggil lama tidak error
    ):
        previous_heading = previous_heading or previous_entity
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
            previous_heading=previous_heading,
            previous_heading_id=previous_heading_id,
        )
        analysis = retrieval_result.get("context") or {}

        print("=" * 55)
        print("USER QUESTION")
        print("=" * 55)
        print("Pertanyaan       :", question)
        print("Effective query  :", effective_question)
        print("Retrieval status :", retrieval_result.get("status"))
        print("Found            :", retrieval_result.get("found"))
        print("Heading          :", analysis.get("heading") or "-")
        print("Subheading       :", analysis.get("subheading") or "-")
        semantic_score = _semantic_scores(analysis)
        print("Score semantic   : heading =", _fmt_score(semantic_score["heading"]),
              "| subheading =", _fmt_score(semantic_score["subheading"]))
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
                f"{answer}\n\nScore semantic: heading {_fmt_score(semantic_score['heading'])}"
                f" | subheading {_fmt_score(semantic_score['subheading'])}"
            )

        best = retrieval_result.get("best_candidate") or {}

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
            "heading": analysis.get("heading"),
            "heading_id": analysis.get("heading_id"),
            "entity": analysis.get("heading"),  # alias nama lama
            "retrieved_heading": best.get("heading"),
            "retrieved_heading_id": best.get("heading_id"),
            "retrieved_subheading": best.get("subheading"),
            "retrieved_subheading_id": best.get("subheading_id"),
            "found": found,
            "status": retrieval_result.get("status"),
        }

        if evaluation:
            # Chunk yang dikirim ke LLM, diurutkan dari skor tertinggi. LLM
            # menerima urutan baca dokumen, tetapi metrik precision menilai
            # peringkat retriever, jadi daftar evaluasi harus berurutan skor.
            ranked = sorted(
                retrieval_result.get("relevant_chunks", []),
                key=lambda item: (item.get("score") is None,
                                  -(item.get("score") or 0.0)),
            )
            response["evaluation_contexts"] = [
                item.get("content", "") for item in ranked
            ]
            response["retrieved_score"] = (
                retrieval_result.get("best_candidate") or {}
            ).get("score")
        return response