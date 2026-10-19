from userquery.query_embedding import create_query_embedding

query = input("Masukkan Pertanyaan : ")

vector = create_query_embedding(query)

if vector is None:

    print("Embedding gagal.")

else:

    print("\nEmbedding berhasil.")

    print("Dimensi :", len(vector))

    print("5 Nilai Pertama :")

    print(vector[:5])