
def _normalize_heading(text):
    return " ".join(
        str(text or "").strip().casefold().split()
    )


def build_heading_description(
    subheading,
    heading=None
):
    key = _normalize_heading(subheading)
    service = heading or "layanan keimigrasian"

    descriptions = {
        "pengertian": (
            f"Menjelaskan pengertian, definisi, tujuan, "
            f"dan gambaran umum dari layanan {service}."
        ),

        "persyaratan": (
            f"Menjelaskan seluruh persyaratan administrasi "
            f"dan dokumen yang harus dipenuhi pemohon "
            f"untuk memperoleh layanan {service}."
        ),

        "negara penerima": (
            f"Menjelaskan negara tujuan atau negara penerima "
            f"yang berkaitan dengan layanan {service}."
        ),

        "sistem mekanisme dan prosedur": (
            f"Menjelaskan sistem, mekanisme, tahapan, "
            f"dan prosedur yang harus dilalui pemohon "
            f"untuk memperoleh layanan {service}."
        ),

        "jangka waktu pelayanan": (
            f"Menjelaskan estimasi waktu, durasi proses, "
            f"dan lama penyelesaian layanan {service}."
        ),

        "biaya/tarif": (
            f"Menjelaskan biaya, tarif resmi, dan rincian "
            f"pembayaran yang dikenakan pada layanan {service}."
        ),

        "biaya": (
            f"Menjelaskan biaya dan tarif resmi "
            f"yang dikenakan pada layanan {service}."
        ),

        "tarif": (
            f"Menjelaskan tarif resmi dan rincian pembayaran "
            f"untuk memperoleh layanan {service}."
        ),

        "produk pelayanan": (
            f"Menjelaskan hasil atau produk pelayanan "
            f"yang diterima pemohon setelah proses "
            f"layanan {service} selesai."
        ),

        "dasar hukum": (
            f"Menjelaskan peraturan perundang-undangan, "
            f"ketentuan, dan landasan hukum yang menjadi "
            f"dasar pelaksanaan layanan {service}."
        ),

        "kompetensi pelaksana": (
            f"Menjelaskan kompetensi, kemampuan, "
            f"dan kualifikasi petugas yang melaksanakan "
            f"layanan {service}."
        ),

        "jumlah pelaksana": (
            f"Menjelaskan jumlah petugas atau personel "
            f"yang terlibat dalam pelaksanaan layanan {service}."
        ),

        "jaminan pelayanan": (
            f"Menjelaskan jaminan kualitas, kepastian, "
            f"dan komitmen pelayanan yang diberikan "
            f"kepada pemohon layanan {service}."
        ),

        "pengawasan internal": (
            f"Menjelaskan mekanisme pengawasan internal "
            f"terhadap pelaksanaan layanan {service}."
        ),

        "jaminan keamanan dan keselamatan pelayanan": (
            f"Menjelaskan jaminan keamanan dan keselamatan "
            f"pemohon selama proses memperoleh layanan {service}."
        )
    }

    return descriptions.get(
        key,
        (
            f"Menjelaskan informasi mengenai {subheading} "
            f"pada layanan {service}."
        )
    )


def build_heading_examples(
    subheading,
    heading=None
):
    key = _normalize_heading(subheading)
    service = heading or "layanan keimigrasian"

    examples = {
        "pengertian": [
            f"Apa yang dimaksud dengan {service}?",
            f"Apa tujuan dari layanan {service}?",
            f"Bisa jelaskan gambaran umum {service}?"
        ],

        "persyaratan": [
            f"Apa saja persyaratan {service}?",
            f"Dokumen apa yang harus disiapkan untuk {service}?",
            f"Apa saja berkas yang diperlukan untuk mengajukan {service}?"
        ],

        "negara penerima": [
            f"Negara mana saja yang menerima layanan {service}?",
            f"Ke negara mana layanan ini dapat digunakan?"
        ],

        "sistem mekanisme dan prosedur": [
            f"Bagaimana prosedur mengajukan {service}?",
            f"Apa tahapan untuk memperoleh {service}?",
            f"Bagaimana mekanisme pelayanan {service}?"
        ],

        "jangka waktu pelayanan": [
            f"Berapa lama proses {service}?",
            f"Berapa hari waktu penyelesaian {service}?",
            f"Kapan layanan {service} selesai?"
        ],

        "biaya/tarif": [
            f"Berapa biaya {service}?",
            f"Berapa tarif resmi {service}?",
            f"Apakah ada biaya tambahan untuk {service}?"
        ],

        "biaya": [
            f"Berapa biaya {service}?",
            f"Berapa biaya yang harus dibayar pemohon?"
        ],

        "tarif": [
            f"Berapa tarif resmi {service}?",
            f"Bagaimana rincian tarif {service}?"
        ],

        "produk pelayanan": [
            f"Apa hasil dari layanan {service}?",
            f"Apa produk pelayanan yang akan diterima pemohon?"
        ],

        "dasar hukum": [
            f"Apa dasar hukum {service}?",
            f"Peraturan apa yang mengatur {service}?"
        ],

        "kompetensi pelaksana": [
            f"Apa kompetensi petugas dalam melaksanakan {service}?",
            f"Siapa yang berwenang melaksanakan layanan ini?"
        ],

        "jumlah pelaksana": [
            f"Berapa jumlah petugas yang melayani {service}?",
            f"Berapa personel yang terlibat dalam layanan ini?"
        ],

        "jaminan pelayanan": [
            f"Apa jaminan pelayanan untuk {service}?",
            f"Bagaimana jaminan kualitas layanan ini?"
        ],

        "pengawasan internal": [
            f"Bagaimana pengawasan internal pada {service}?",
            f"Siapa yang mengawasi pelaksanaan layanan ini?"
        ],

        "jaminan keamanan dan keselamatan pelayanan": [
            f"Bagaimana jaminan keamanan dalam {service}?",
            f"Apa jaminan keselamatan bagi pemohon layanan ini?"
        ]
    }

    return examples.get(
        key,
        [
            f"Apa informasi mengenai {subheading} pada {service}?",
            f"Bagaimana ketentuan {subheading} untuk {service}?"
        ]
    )
