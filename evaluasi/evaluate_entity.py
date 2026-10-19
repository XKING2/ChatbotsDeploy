"""
evaluate_entity.py

Evaluasi Entity Retrieval
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

from LLM.chatbot import HealthcareChatbot


# =====================================================
# CONFIG
# =====================================================

DATASET_PATH = "data/excel/data_pertanyaan.xlsx"

OUTPUT_PATH = "data/excel/entity_prediction_result.xlsx"


# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 80)
print("MEMBACA DATASET")
print("=" * 80)

df = pd.read_excel(DATASET_PATH)

required_columns = [
    "question",
    "ground_truth_entity"
]

for col in required_columns:

    if col not in df.columns:

        raise Exception(
            f"Kolom '{col}' tidak ditemukan pada dataset."
        )

print(f"Jumlah data : {len(df)}")

bot = HealthcareChatbot()

predictions = []
truths = []


print("\n")
print("=" * 80)
print("ENTITY RETRIEVAL EVALUATION")
print("=" * 80)


# =====================================================
# PREDIKSI
# =====================================================

for index, row in df.iterrows():

    question = str(row["question"]).strip()

    ground_truth = str(
        row["ground_truth_entity"]
    ).strip().upper()

    result = bot.retrieve_only(question)

    predicted = result["analysis"]["entity"]

    if predicted is None:
        predicted = "UNKNOWN"

    predicted = predicted.strip().upper()

    truths.append(ground_truth)
    predictions.append(predicted)

    print(f"[{index+1:03d}] {question}")
    print(f"GT   : {ground_truth}")
    print(f"PRED : {predicted}")

    if ground_truth == predicted:
        print("✓ Correct")
    else:
        print("✗ Wrong")

    print("-" * 60)


# =====================================================
# SIMPAN HASIL
# =====================================================

df["prediction"] = predictions

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
print("EVALUATION RESULT")
print("=" * 80)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# =====================================================
# CLASSIFICATION REPORT
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


# =====================================================
# SIMPAN CONFUSION MATRIX
# =====================================================

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