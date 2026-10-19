"""
evaluate_semantic_search.py

Evaluasi Semantic Search tanpa memanggil LLM.
"""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from retrival.entity_retrieval import EntityRetrieval
from retrival.semantic_search import semantic_search
from userquery.query_embedding import create_query_embedding


# =====================================================
# CONFIG
# =====================================================

DATASET_PATH = "data/excel/Dataset_Groundtruth_labeled.xlsx"
OUTPUT_PATH = "data/excel/semantic_search_result.xlsx"

TOP_K = 5


# =====================================================
# LOAD
# =====================================================

print("=" * 80)
print("MEMBACA DATASET")
print("=" * 80)

df = pd.read_excel(DATASET_PATH)

required = [
    "question",
    "ground_truth_entity"
]

for col in required:

    if col not in df.columns:
        raise Exception(f"Kolom {col} tidak ditemukan.")

print("Jumlah data :", len(df))


# =====================================================
# INIT
# =====================================================

entity_retrieval = EntityRetrieval()

truths = []
predictions = []

retrieved_heading = []
retrieved_score = []


print("\n")
print("=" * 80)
print("SEMANTIC SEARCH EVALUATION")
print("=" * 80)


# =====================================================
# LOOP
# =====================================================

for index, row in df.iterrows():

    question = str(row["question"]).strip()

    ground_truth = (
        str(row["ground_truth_entity"])
        .strip()
        .upper()
    )

    print("\n")
    print("=" * 80)
    print("QUESTION")
    print(question)

    # ------------------------------------
    # Embedding
    # ------------------------------------

    query_embedding = create_query_embedding(question)

    # ------------------------------------
    # Entity Retrieval
    # ------------------------------------

    entity_result = entity_retrieval.retrieve(
        query_embedding
    )

    if entity_result["found"]:

        entity = entity_result["results"][0]["name"]

    else:

        entity = "UNKNOWN"

    # ------------------------------------
    # Semantic Search
    # ------------------------------------

    results = semantic_search(
        query_embedding=query_embedding,
        disease=entity,
        top_k=TOP_K
    )

    # ------------------------------------
    # Ambil metadata hasil ranking pertama
    # ------------------------------------

    heading = "-"

    score = 0

    if len(results["metadatas"][0]) > 0:

        meta = results["metadatas"][0][0]

        heading = meta.get("heading", "-")

    if len(results["distances"][0]) > 0:

        score = results["distances"][0][0]

    truths.append(ground_truth)
    predictions.append(entity)

    retrieved_heading.append(heading)
    retrieved_score.append(score)

    print(f"[{index+1:03d}] {question}")

    print("GT        :", ground_truth)
    print("Prediction:", entity)
    print("Heading   :", heading)
    print("Distance  :", round(score,4))

    if ground_truth == entity:

        print("✓ Correct")

    else:

        print("✗ Wrong")


# =====================================================
# SAVE
# =====================================================

df["prediction"] = predictions
df["heading"] = retrieved_heading
df["distance"] = retrieved_score

df["correct"] = (

    df["ground_truth_entity"]
      .astype(str)
      .str.upper()

    ==

    df["prediction"]

)

df.to_excel(
    OUTPUT_PATH,
    index=False
)


# =====================================================
# METRIC
# =====================================================

accuracy = accuracy_score(
    truths,
    predictions
)

precision = precision_score(
    truths,
    predictions,
    average="micro",
    zero_division=0
)

recall = recall_score(
    truths,
    predictions,
    average="micro",
    zero_division=0
)

f1 = f1_score(
    truths,
    predictions,
    average="micro",
    zero_division=0
)


print("\n")
print("=" * 80)
print("HASIL")
print("=" * 80)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# =====================================================
# REPORT
# =====================================================

print("\n")
print("=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(

    classification_report(

        truths,

        predictions,

        zero_division=0

    )

)


# =====================================================
# CONFUSION MATRIX
# =====================================================

labels = sorted(

    list(

        set(truths + predictions)

    )

)

cm = confusion_matrix(

    truths,

    predictions,

    labels=labels

)

cm_df = pd.DataFrame(

    cm,

    index=labels,

    columns=labels

)

print("\n")
print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm_df)


with pd.ExcelWriter(

    OUTPUT_PATH,

    engine="openpyxl",

    mode="a",

    if_sheet_exists="replace"

) as writer:

    cm_df.to_excel(

        writer,

        sheet_name="Confusion Matrix"

    )


print("\n")
print("=" * 80)
print("HASIL DISIMPAN")
print("=" * 80)

print(OUTPUT_PATH)