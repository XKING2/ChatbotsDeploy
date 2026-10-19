import pandas as pd

from LLM.chatbot import HealthcareChatbot

chatbot = HealthcareChatbot()

questions = pd.read_excel(
    "data/excel/data_pertanyaan.xlsx"
)

rows = []

for _, row in questions.iterrows():

    question = row["question"]

    result = chatbot.ask(
        question,
        evaluation=True
    )

    rows.append({

        "question": question,

        "contexts": result["contexts"],

        "answer": result["answer"]

    })

df = pd.DataFrame(rows)

df.to_csv(

    "data/excel/evaluation_dataset.csv",

    index=False,

    encoding="utf-8-sig"

)

print("Selesai")