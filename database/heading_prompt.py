def _normalize_heading(heading):
    return " ".join((heading or "").strip().lower().split())


def build_heading_description(heading, entity=None):
    heading = _normalize_heading(heading)
    entity = (entity or "layanan ini").strip()

    descriptions = {
        "pengertian": f"Bagian ini menjelaskan pengertian dan gambaran umum {entity}, termasuk apa itu layanan tersebut, untuk siapa layanan digunakan, dan dalam kondisi apa layanan tersebut digunakan.",
        "persyaratan": f"Bagian ini menjelaskan dokumen dan persyaratan yang wajib disiapkan untuk mengajukan {entity}.",
        "negara penerima": f"Bagian ini menjelaskan negara yang warga negaranya berhak mendapatkan fasilitas atau kebijakan dari {entity}.",
        "sistem mekanisme dan prosedur": f"Bagian ini menjelaskan tahapan atau prosedur pengajuan {entity} secara berurutan.",
        "jangka waktu pelayanan": f"Bagian ini menjelaskan berapa lama proses pelayanan {entity} biasanya diselesaikan.",
        "biaya/tarif": f"Bagian ini menjelaskan biaya atau tarif resmi yang harus dibayarkan untuk {entity}.",
        "produk pelayanan": f"Bagian ini menjelaskan dokumen atau hasil akhir yang diterima pemohon setelah proses {entity} selesai.",
        "dasar hukum": f"Bagian ini menjelaskan peraturan yang menjadi dasar hukum penyelenggaraan {entity}.",
        "kompetensi pelaksana": f"Bagian ini menjelaskan kompetensi yang harus dimiliki petugas dalam melaksanakan {entity}.",
        "jumlah pelaksana": f"Bagian ini menjelaskan jumlah petugas yang menangani {entity}.",
        "jaminan pelayanan": f"Bagian ini menjelaskan jaminan dan komitmen kualitas pelayanan yang diberikan dalam {entity}.",
        "pengawasan internal": f"Bagian ini menjelaskan mekanisme pengawasan internal terhadap pelaksanaan {entity}.",
        "jaminan keamanan dan keselamatan pelayanan": f"Bagian ini menjelaskan jaminan keamanan dan keselamatan dalam pelaksanaan {entity}."
    }
    return descriptions.get(
        heading,
        f"Bagian ini menjelaskan informasi mengenai {heading} pada {entity}."
    )


def build_heading_examples(heading, entity=None):
    heading = _normalize_heading(heading)
    entity = (entity or "layanan ini").strip()

    examples = {
        "pengertian": [
            f"Apa itu {entity}?",
            f"Apa yang dimaksud dengan {entity}?",
            f"Jelaskan pengertian {entity}."
        ],
        "persyaratan": [
            f"Apa saja syarat {entity}?",
            f"Dokumen apa yang dibutuhkan untuk {entity}?",
            f"Berkas apa yang harus disiapkan untuk {entity}?"
        ],
        "negara penerima": [
            f"Negara mana saja yang bisa mendapatkan fasilitas {entity}?"
        ],
        "sistem mekanisme dan prosedur": [
            f"Bagaimana prosedur {entity}?",
            f"Apa saja tahapan {entity}?"
        ],
        "jangka waktu pelayanan": [
            f"Berapa lama proses {entity}?",
            f"Berapa hari sampai proses {entity} selesai?"
        ],
        "biaya/tarif": [
            f"Berapa biaya {entity}?",
            f"Berapa tarif {entity}?"
        ],
        "produk pelayanan": [
            f"Apa yang diterima setelah proses {entity} selesai?"
        ],
        "dasar hukum": [
            f"Apa dasar hukum {entity}?",
            f"{entity} berdasarkan peraturan apa?"
        ],
        "kompetensi pelaksana": [
            f"Apa kompetensi petugas untuk {entity}?"
        ],
        "jumlah pelaksana": [
            f"Berapa petugas yang menangani {entity}?"
        ],
        "jaminan pelayanan": [
            f"Apa jaminan pelayanan untuk {entity}?"
        ],
        "pengawasan internal": [
            f"Bagaimana pengawasan internal {entity}?"
        ],
        "jaminan keamanan dan keselamatan pelayanan": [
            f"Bagaimana jaminan keamanan dan keselamatan {entity}?"
        ]
    }
    return examples.get(
        heading,
        [
            f"Informasi mengenai {heading} {entity}.",
            f"Jelaskan {heading} {entity}."
        ]
    )
