"""
Bandingkan hasil precision pertanyaan YANG SAMA antar bahasa.

    python -m ragas_eval.diag_language

Memisahkan penyebab:
  - retrieval berbeda antar bahasa (chunk yang terambil tidak sama), atau
  - chunk sama tetapi penilaian juri berbeda.
"""
import json

import pandas as pd

try:
    from . import ragas_runner as base
except ImportError:
    import ragas_runner as base

BASE_LANG = "Indonesia"  # bahasa pembanding (sama dengan isi kolom language)


def main() -> None:
    data, _ = base.load_dataset(base.INPUT_FILE)
    res = pd.read_excel(base.OUTPUT_FILE, sheet_name="RAGAS Results", engine="openpyxl")
    res = res[res["status"].eq("success")]
    df = data.merge(res[["row_id", "context_precision", "relevant_contexts"]], on="row_id")
    df["precision"] = pd.to_numeric(df["context_precision"])
    df["n_contexts"] = df["retrieved_contexts"].apply(len)
    df["n_relevant"] = df["relevant_contexts"].apply(lambda s: len(json.loads(s)))
    df["ctx_key"] = df["retrieved_contexts"].apply(lambda c: tuple(sorted(c)))

    langs = list(df["language"].drop_duplicates())
    df = df.sort_values("row_id")
    df["pos"] = df.groupby("language").cumcount()
    topics = df.pivot(index="pos", columns="language", values="topic")
    aligned = (topics.nunique(axis=1) == 1).mean()
    print(f"\nBahasa: {langs}")
    print(f"Urutan baris sejajar antar bahasa (topik sama di posisi sama): {aligned:.0%}")
    if aligned < 0.95:
        print("PERINGATAN: urutan tidak sejajar; bagian pasangan di bawah tidak dapat dipercaya.")

    print("\n1) Per bahasa")
    print(df.groupby("language").agg(
        n=("precision", "count"), precision=("precision", "mean"),
        rata2_konteks=("n_contexts", "mean"), rata2_relevan=("n_relevant", "mean"),
    ).round(3).to_string())

    if BASE_LANG not in langs:
        print(f"\n(Ubah BASE_LANG; bahasa tersedia: {langs})")
        return
    ref = df[df["language"] == BASE_LANG].set_index("pos")
    print(f"\n2) Dibanding pertanyaan yang sama dalam bahasa {BASE_LANG}")
    rows = []
    for lang in langs:
        if lang == BASE_LANG:
            continue
        cur = df[df["language"] == lang].set_index("pos")
        idx = cur.index.intersection(ref.index)
        same = (cur.loc[idx, "ctx_key"] == ref.loc[idx, "ctx_key"])
        diff = cur.loc[idx, "precision"] - ref.loc[idx, "precision"]
        rows.append({
            "bahasa": lang,
            "konteks_sama_persen": same.mean() * 100,
            "selisih_precision_semua": diff.mean() * 100,
            "selisih_bila_konteks_SAMA": diff[same].mean() * 100 if same.any() else float("nan"),
            "selisih_bila_konteks_BEDA": diff[~same].mean() * 100 if (~same).any() else float("nan"),
            "lebih_rendah": int((diff < 0).sum()), "lebih_tinggi": int((diff > 0).sum()),
        })
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print("""
Cara membaca:
- konteks_sama_persen rendah  -> retrieval berbeda antar bahasa (penyebab di retrieval).
- selisih_bila_konteks_SAMA jauh dari 0 -> chunk sama tapi juri menilai beda (penyebab di penilaian).
- selisih_bila_konteks_BEDA jauh negatif -> chunk berbeda itulah yang menurunkan precision.""")


if __name__ == "__main__":
    main()