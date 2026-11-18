
from __future__ import annotations

import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from openpyxl import Workbook, load_workbook
from openpyxl.styles import (
    Alignment,
    Border,
    Font,
    PatternFill,
    Side,
)
from openpyxl.utils import get_column_letter


# ============================================================
# PROJECT IMPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EMBEDDING_MODEL_1

# Menggunakan embedding service yang sama dengan sistem chatbot.
# Kuota, cache, dan failover API key tetap dikelola oleh service.
try:
    from database.embedding import create_embeddings_batch
except ImportError:
    from database.embedding import create_embeddings_batch


# ============================================================
# KONFIGURASI
# ============================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "Excel"
    / "dataset_cosine_test.xlsx"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "Excel"
    / "Hasil_Cosine_Similarity.xlsx"
)

SHEET_NAME = "Dataset Cosine"

# Jumlah teks maksimal dalam satu batch.
EMBEDDING_BATCH_SIZE = 20

# None berarti menggunakan konfigurasi default embedding service.
# Harus konsisten untuk seluruh pasangan jawaban yang dibandingkan.
EMBEDDING_TASK_TYPE = None

# Threshold penelitian sesuai dokumen pengujian.
COSINE_THRESHOLD = 0.75

# Hanya mengevaluasi jawaban yang berhasil dihasilkan.
ONLY_SUCCESSFUL_GENERATION = True

REQUIRED_COLUMNS = [
    "question_id",
    "language",
    "reference",
    "response",
]

OPTIONAL_COLUMNS = [
    "topic",
    "question",
    "generation_status",
    "found",
    "retrieval_status",
]


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(value) -> str:
    """Membersihkan nilai Excel menjadi string."""

    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def normalize_status(value: str) -> str:
    """Menormalkan status generasi."""

    return clean_text(value).lower()


# ============================================================
# DATASET LOADER
# ============================================================

def read_dataset(
    input_file: Path = INPUT_FILE,
    sheet_name: str = SHEET_NAME,
) -> tuple[list[dict], list[dict]]:

    if not input_file.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {input_file}"
        )

    workbook = load_workbook(
        input_file,
        read_only=True,
        data_only=True,
    )

    try:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(
                f"Sheet '{sheet_name}' tidak ditemukan. "
                f"Sheet tersedia: {workbook.sheetnames}"
            )

        worksheet = workbook[sheet_name]

        rows = worksheet.iter_rows(values_only=True)

        headers = next(rows, None)

        if headers is None:
            raise ValueError("Dataset kosong.")

        headers = [
            clean_text(header).lower()
            for header in headers
        ]

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in headers
        ]

        if missing:
            raise ValueError(
                f"Kolom wajib tidak ditemukan: {missing}"
            )

        index = {
            column: headers.index(column)
            for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
            if column in headers
        }

        records = []
        skipped = []

        for excel_row, row in enumerate(rows, start=2):

            def get(name: str) -> str:
                column_index = index.get(name)

                if column_index is None:
                    return ""

                if column_index >= len(row):
                    return ""

                return clean_text(row[column_index])

            question_id = get("question_id")
            language = get("language") or "Unknown"
            topic = get("topic")
            question = get("question")

            reference = get("reference")
            response = get("response")

            # Lewati baris yang benar-benar kosong.
            if not any(
                (
                    question_id,
                    language if language != "Unknown" else "",
                    topic,
                    question,
                    reference,
                    response,
                )
            ):
                continue

            reason = None

            if not question_id:
                reason = "Question ID kosong"

            elif not reference:
                reason = "Ground Truth kosong"

            elif not response:
                reason = "Generated Answer kosong"

            elif (
                ONLY_SUCCESSFUL_GENERATION
                and get("generation_status")
                and normalize_status(
                    get("generation_status")
                ) != "success"
            ):
                reason = (
                    "Generasi gagal: "
                    + get("generation_status")[:100]
                )

            if reason:
                skipped.append({
                    "source_row": excel_row,
                    "question_id": question_id,
                    "language": language,
                    "topic": topic,
                    "reason": reason,
                })
                continue

            records.append({
                "source_row": excel_row,
                "question_id": question_id,
                "language": language,
                "topic": topic,
                "question": question,
                "reference": reference,
                "response": response,
            })

    finally:
        workbook.close()

    if not records:
        raise ValueError(
            "Tidak ada pasangan Generated Answer "
            "dan Ground Truth yang dapat dievaluasi."
        )

    print(f"[DATASET] Data valid   : {len(records)}")
    print(f"[DATASET] Data dilewati: {len(skipped)}")

    return records, skipped


# ============================================================
# EMBEDDING
# ============================================================

def embed_records(
    records: list[dict],
) -> list[list[float] | None]:

    """
    Membuat embedding secara berpasangan:

    [response_1, reference_1,
     response_2, reference_2, ...]

    Menggunakan cache dan failover dari embedding service.
    """

    texts = []

    for record in records:
        texts.extend([
            record["response"],
            record["reference"],
        ])

    print("\n" + "=" * 65)
    print("EMBEDDING GENERATION")
    print("=" * 65)

    print(f"Model          : {EMBEDDING_MODEL_1}")
    print(f"Task Type      : {EMBEDDING_TASK_TYPE or 'Default'}")
    print(f"Total teks     : {len(texts)}")
    print(f"Teks unik      : {len(set(texts))}")
    print(f"Batch maksimum : {EMBEDDING_BATCH_SIZE}")

    vectors = create_embeddings_batch(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        task_type=EMBEDDING_TASK_TYPE,
        allow_partial=True,
    )

    if vectors is None or len(vectors) != len(texts):
        raise RuntimeError(
            "Layanan embedding tidak mengembalikan "
            "jumlah vektor yang sesuai."
        )

    return vectors


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:

    """
    Menghitung Cosine Similarity:

                  A . B
    cos(theta) = -------
                 ||A|| ||B||

    Menghasilkan nilai antara -1 sampai 1.
    """

    if not a or not b:
        raise ValueError("Embedding kosong.")

    if len(a) != len(b):
        raise ValueError(
            "Dimensi kedua embedding tidak sama."
        )

    dot_product = math.fsum(
        float(x) * float(y)
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        math.fsum(float(x) ** 2 for x in a)
    )

    norm_b = math.sqrt(
        math.fsum(float(y) ** 2 for y in b)
    )

    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            "Embedding memiliki norma nol."
        )

    score = dot_product / (norm_a * norm_b)

    if not math.isfinite(score):
        raise ValueError(
            "Hasil Cosine Similarity tidak valid."
        )

    # Menghindari kesalahan pembulatan floating point.
    return max(-1.0, min(1.0, score))


# ============================================================
# VALID / INVALID CLASSIFICATION
# ============================================================

def classify_similarity(score: float) -> str:
    """
    Klasifikasi berdasarkan threshold penelitian.

    Score >= 0.75 : Valid
    Score <  0.75 : Invalid
    """

    if score >= COSINE_THRESHOLD:
        return "Valid"

    return "Invalid"


# ============================================================
# CALCULATE ALL SCORES
# ============================================================

def calculate_scores(
    records: list[dict],
    vectors: list[list[float] | None],
) -> tuple[list[dict], list[dict]]:

    results = []
    skipped = []

    expected_vectors = len(records) * 2

    if len(vectors) != expected_vectors:
        raise ValueError(
            "Jumlah embedding tidak sesuai jumlah data."
        )

    for i, record in enumerate(records):

        response_vector = vectors[2 * i]
        reference_vector = vectors[2 * i + 1]

        if (
            response_vector is None
            or reference_vector is None
        ):
            skipped.append({
                "source_row": record["source_row"],
                "question_id": record["question_id"],
                "language": record["language"],
                "topic": record["topic"],
                "reason": (
                    "Embedding belum selesai. "
                    "Periksa kuota dan jalankan ulang."
                ),
            })
            continue

        try:
            score = cosine_similarity(
                response_vector,
                reference_vector,
            )

            status = classify_similarity(score)

        except (ValueError, TypeError, OverflowError) as error:
            skipped.append({
                "source_row": record["source_row"],
                "question_id": record["question_id"],
                "language": record["language"],
                "topic": record["topic"],
                "reason": (
                    f"Embedding tidak valid: {error}"
                ),
            })
            continue

        results.append({
            **record,
            "cosine_similarity": score,
            "threshold": COSINE_THRESHOLD,
            "status": status,
        })

    print("\n" + "=" * 65)
    print("COSINE SIMILARITY CALCULATION")
    print("=" * 65)

    print(
        f"Berhasil dihitung : {len(results)}"
    )

    print(
        f"Gagal / dilewati  : {len(skipped)}"
    )

    return results, skipped


# ============================================================
# STATISTICS
# ============================================================

def describe(scores: list[float]) -> dict:

    if not scores:
        return {
            "average": None,
            "median": None,
            "std": None,
            "maximum": None,
            "minimum": None,
        }

    return {
        "average": statistics.fmean(scores),
        "median": statistics.median(scores),
        "std": (
            statistics.pstdev(scores)
            if len(scores) > 1
            else 0.0
        ),
        "maximum": max(scores),
        "minimum": min(scores),
    }


def build_group_summary(
    results: list[dict],
    group_column: str,
) -> list[dict]:

    grouped = defaultdict(list)

    for item in results:
        group_name = (
            item.get(group_column) or "Tidak diketahui"
        )

        grouped[group_name].append(item)

    summary = []

    for group_name, items in grouped.items():

        scores = [
            item["cosine_similarity"]
            for item in items
        ]

        total = len(items)

        valid_count = sum(
            item["status"] == "Valid"
            for item in items
        )

        invalid_count = total - valid_count

        valid_percentage = (
            valid_count / total
            if total else 0.0
        )

        invalid_percentage = (
            invalid_count / total
            if total else 0.0
        )

        summary.append({
            "group": group_name,
            "total": total,
            "valid": valid_count,
            "invalid": invalid_count,
            "valid_percentage": valid_percentage,
            "invalid_percentage": invalid_percentage,
            **describe(scores),
        })

    return sorted(
        summary,
        key=lambda item: item["group"]
    )


def calculate_overall_summary(
    results: list[dict],
) -> dict:

    scores = [
        item["cosine_similarity"]
        for item in results
    ]

    total = len(results)

    valid_count = sum(
        item["status"] == "Valid"
        for item in results
    )

    invalid_count = total - valid_count

    return {
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "valid_percentage": (
            valid_count / total if total else 0.0
        ),
        "invalid_percentage": (
            invalid_count / total if total else 0.0
        ),
        **describe(scores),
    }


# ============================================================
# EXCEL FORMATTING
# ============================================================

def format_worksheet(ws) -> None:

    side = Side(
        style="thin",
        color="D9E2F3",
    )

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.sheet_view.showGridLines = False

    ws.row_dimensions[1].height = 32

    for cell in ws[1]:
        cell.fill = PatternFill(
            "solid",
            fgColor="1F4E78",
        )

        cell.font = Font(
            color="FFFFFF",
            bold=True,
        )

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

            cell.border = Border(
                bottom=side
            )

    for column in ws.columns:

        longest = max(
            (
                len(str(cell.value or ""))
                for cell in column[:100]
            ),
            default=12,
        )

        column_letter = get_column_letter(
            column[0].column
        )

        ws.column_dimensions[column_letter].width = min(
            max(longest + 2, 14),
            55,
        )


# ============================================================
# EXCEL DETAIL EXPORT
# ============================================================

def export_detail_sheet(
    wb: Workbook,
    results: list[dict],
) -> None:

    ws = wb.active
    ws.title = "Detail Hasil"

    ws.append([
        "question_id",
        "language",
        "topic",
        "question",
        "reference",
        "response",
        "cosine_similarity",
        "threshold",
        "status",
        "source_excel_row",
    ])

    for item in results:

        ws.append([
            item["question_id"],
            item["language"],
            item["topic"],
            item["question"],
            item["reference"],
            item["response"],
            item["cosine_similarity"],
            item["threshold"],
            item["status"],
            item["source_row"],
        ])

    for row in ws.iter_rows(min_row=2):

        row[6].number_format = "0.000000"
        row[7].number_format = "0.00"

        if row[8].value == "Valid":
            row[8].font = Font(
                color="008000",
                bold=True,
            )
        else:
            row[8].font = Font(
                color="FF0000",
                bold=True,
            )

    ws.column_dimensions["D"].width = 45
    ws.column_dimensions["E"].width = 60
    ws.column_dimensions["F"].width = 70


# ============================================================
# SUMMARY SHEET EXPORT
# ============================================================

def export_group_summary(
    wb: Workbook,
    sheet_name: str,
    summary: list[dict],
) -> None:

    ws = wb.create_sheet(sheet_name)

    ws.append([
        "kelompok",
        "jumlah_data",
        "valid",
        "invalid",
        "persentase_valid",
        "persentase_invalid",
        "rata_rata_similarity",
        "median",
        "standar_deviasi",
        "nilai_tertinggi",
        "nilai_terendah",
    ])

    for item in summary:

        ws.append([
            item["group"],
            item["total"],
            item["valid"],
            item["invalid"],
            item["valid_percentage"],
            item["invalid_percentage"],
            item["average"],
            item["median"],
            item["std"],
            item["maximum"],
            item["minimum"],
        ])

    for row in ws.iter_rows(min_row=2):

        row[4].number_format = "0.00%"
        row[5].number_format = "0.00%"

        for index in range(6, 11):
            row[index].number_format = "0.000000"


def export_overall_summary(
    wb: Workbook,
    overall: dict,
) -> None:

    ws = wb.create_sheet("Ringkasan Keseluruhan")

    ws.append([
        "parameter",
        "nilai",
    ])

    data = [
        ("Total Data Dievaluasi", overall["total"]),
        ("Jumlah Valid", overall["valid"]),
        ("Jumlah Invalid", overall["invalid"]),
        (
            "Persentase Valid",
            overall["valid_percentage"],
        ),
        (
            "Persentase Invalid",
            overall["invalid_percentage"],
        ),
        ("Threshold", COSINE_THRESHOLD),
        (
            "Rata-rata Cosine Similarity",
            overall["average"],
        ),
        ("Median", overall["median"]),
        (
            "Standar Deviasi",
            overall["std"],
        ),
        (
            "Nilai Similarity Tertinggi",
            overall["maximum"],
        ),
        (
            "Nilai Similarity Terendah",
            overall["minimum"],
        ),
        ("Embedding Model", EMBEDDING_MODEL_1),
        (
            "Embedding Task Type",
            EMBEDDING_TASK_TYPE or "Default",
        ),
        (
            "Tanggal Pengujian",
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        ),
    ]

    for item in data:
        ws.append(list(item))

    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 40

    for row in ws.iter_rows(min_row=2):

        parameter = row[0].value

        if parameter in (
            "Persentase Valid",
            "Persentase Invalid",
        ):
            row[1].number_format = "0.00%"

        elif isinstance(row[1].value, float):
            row[1].number_format = "0.000000"


# ============================================================
# SKIPPED DATA EXPORT
# ============================================================

def export_skipped_sheet(
    wb: Workbook,
    skipped: list[dict],
) -> None:

    if not skipped:
        return

    ws = wb.create_sheet("Data Dilewati")

    ws.append([
        "source_excel_row",
        "question_id",
        "language",
        "topic",
        "reason",
    ])

    for item in skipped:

        ws.append([
            item.get("source_row"),
            item.get("question_id"),
            item.get("language"),
            item.get("topic"),
            item.get("reason"),
        ])


# ============================================================
# EXPORT ALL RESULTS
# ============================================================

def export_results(
    results: list[dict],
    skipped: list[dict],
    language_summary: list[dict],
    topic_summary: list[dict],
    overall: dict,
) -> None:

    wb = Workbook()

    # Detail hasil setiap pertanyaan.
    export_detail_sheet(
        wb,
        results,
    )

    # Ringkasan berdasarkan bahasa.
    export_group_summary(
        wb,
        "Ringkasan Bahasa",
        language_summary,
    )

    # Ringkasan berdasarkan topik.
    export_group_summary(
        wb,
        "Ringkasan Topik",
        topic_summary,
    )

    # Ringkasan keseluruhan.
    export_overall_summary(
        wb,
        overall,
    )

    # Data yang tidak berhasil dievaluasi.
    export_skipped_sheet(
        wb,
        skipped,
    )

    # Format semua worksheet.
    for ws in wb.worksheets:
        format_worksheet(ws)

    # Atur ulang beberapa lebar kolom.
    detail = wb["Detail Hasil"]

    detail.column_dimensions["D"].width = 45
    detail.column_dimensions["E"].width = 60
    detail.column_dimensions["F"].width = 70

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wb.save(OUTPUT_FILE)

    print(
        f"\n[EXPORT] Hasil tersimpan: {OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("MULTILINGUAL RAG")
    print("COSINE SIMILARITY EVALUATION")
    print("=" * 70)

    print(f"Input     : {INPUT_FILE}")
    print(f"Output    : {OUTPUT_FILE}")
    print(f"Model     : {EMBEDDING_MODEL_1}")
    print(f"Threshold : {COSINE_THRESHOLD}")

    # --------------------------------------------------------
    # STEP 1: LOAD DATASET
    # --------------------------------------------------------

    print("\n[STEP 1] Membaca dataset...")

    records, skipped = read_dataset()

    print(
        f"[INFO] Siap dievaluasi: {len(records)}"
    )

    # --------------------------------------------------------
    # STEP 2: GENERATE EMBEDDINGS
    # --------------------------------------------------------

    print("\n[STEP 2] Membuat embedding...")

    vectors = embed_records(records)

    # --------------------------------------------------------
    # STEP 3: CALCULATE COSINE SIMILARITY
    # --------------------------------------------------------

    print("\n[STEP 3] Menghitung Cosine Similarity...")

    results, embedding_skipped = calculate_scores(
        records,
        vectors,
    )

    skipped.extend(embedding_skipped)

    if not results:
        raise RuntimeError(
            "Tidak ada data yang berhasil dihitung. "
            "Periksa embedding service dan kuota API."
        )

    # --------------------------------------------------------
    # STEP 4: BUILD SUMMARY
    # --------------------------------------------------------

    print("\n[STEP 4] Membuat ringkasan evaluasi...")

    language_summary = build_group_summary(
        results,
        "language",
    )

    topic_summary = build_group_summary(
        results,
        "topic",
    )

    overall = calculate_overall_summary(
        results,
    )

    # --------------------------------------------------------
    # STEP 5: EXPORT EXCEL
    # --------------------------------------------------------

    print("\n[STEP 5] Mengekspor hasil ke Excel...")

    export_results(
        results=results,
        skipped=skipped,
        language_summary=language_summary,
        topic_summary=topic_summary,
        overall=overall,
    )

    # --------------------------------------------------------
    # STEP 6: DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HASIL PENGUJIAN COSINE SIMILARITY")
    print("=" * 70)

    print(f"Threshold       : {COSINE_THRESHOLD}")
    print(f"Total dievaluasi: {overall['total']}")
    print(f"Valid           : {overall['valid']}")
    print(f"Invalid         : {overall['invalid']}")

    print(
        f"Persentase Valid   : "
        f"{overall['valid_percentage']:.2%}"
    )

    print(
        f"Persentase Invalid : "
        f"{overall['invalid_percentage']:.2%}"
    )

    print("-" * 70)

    print(
        f"Rata-rata Similarity : "
        f"{overall['average']:.6f}"
    )

    print(
        f"Median               : "
        f"{overall['median']:.6f}"
    )

    print(
        f"Standar Deviasi      : "
        f"{overall['std']:.6f}"
    )

    print(
        f"Nilai Tertinggi      : "
        f"{overall['maximum']:.6f}"
    )

    print(
        f"Nilai Terendah       : "
        f"{overall['minimum']:.6f}"
    )

    # --------------------------------------------------------
    # LANGUAGE SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RINGKASAN BERDASARKAN BAHASA")
    print("=" * 70)

    for item in language_summary:

        print(
            f"{item['group']:<15} "
            f"Total: {item['total']:<4} "
            f"Valid: {item['valid']:<4} "
            f"Invalid: {item['invalid']:<4} "
            f"Valid: {item['valid_percentage']:.2%}"
        )

    # --------------------------------------------------------
    # TOPIC SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RINGKASAN BERDASARKAN TOPIK")
    print("=" * 70)

    for item in topic_summary:

        print(
            f"{item['group']:<25} "
            f"Total: {item['total']:<4} "
            f"Valid: {item['valid']:<4} "
            f"Invalid: {item['invalid']:<4} "
            f"Valid: {item['valid_percentage']:.2%}"
        )

    print("\n" + "=" * 70)

    if skipped:
        print(
            f"[PERHATIAN] Ada {len(skipped)} data "
            "yang dilewati. Periksa sheet Data Dilewati."
        )

    print("[SUCCESS] Evaluasi Cosine Similarity selesai.")
    print(f"[OUTPUT] {OUTPUT_FILE}")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
