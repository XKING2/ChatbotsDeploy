import ast
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    ResponseRelevancy
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import GEMINI_API_KEY, LLM_MODEL, EMBEDDING_MODEL


# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("data/excel/evaluation_dataset.csv")
ground_truth = pd.read_excel("data/excel/Dataset_Groundtruth.xlsx")

df["ground_truth"] = ground_truth["ground_truth"]
df["contexts"] = df["contexts"].apply(ast.literal_eval)

dataset = Dataset.from_pandas(df)


# ==========================
# GEMINI LLM
# ==========================

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# ==========================
# EMBEDDINGS
# ==========================

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY
)


# ==========================
# METRICS
# ==========================

metrics = [
    Faithfulness(),
    ContextPrecision(),
    ContextRecall(),
    ResponseRelevancy(),
]


# ==========================
# EVALUATE
# ==========================

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings
)

print(result)