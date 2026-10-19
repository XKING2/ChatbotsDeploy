
from userquery.query_normalizer import normalize_query


TEST_QUERIES = [
    "spill dong gv tu sbnrny apaan se-worth it itu kah??",
    "gv tuh sebenernya mahluk apa sih wkwk kepo bgt gwa jgn bilang gimmick doang",
    "cara extend visa gmn ya biar ga ketar ketir overstay help yaww",
    "syrt e-voa buat bule apaan aja? udh scroll tiktok tp blm nemu yg sreg",
    "eh spill dong syarat kitas yg gampang no ribet ribet club plisss!!",
    "gci ama golden visa tuh 11 12 ato kek gmn? jelasin dong sepuh masi awam nih",
]


if __name__ == "__main__":
    for question in TEST_QUERIES:
        normalized = normalize_query(question)

        print(f"Original   : {question}")
        print(f"Normalized : {normalized}")
        print("-" * 60)
