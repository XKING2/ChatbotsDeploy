import streamlit as st
import chromadb
import os

# Konfigurasi halaman
st.set_page_config(page_title="ChromaDB Local Viewer", layout="wide")
st.title("🔎 ChromaDB Local Viewer")

# Sesuai dengan konfigurasi PATH Anda
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

st.sidebar.info(f"📁 Memuat database dari:\n`{CHROMA_DB_PATH}`")

try:
    # Hubungkan ke database lokal Anda
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Ambil daftar semua koleksi
    collections = client.list_collections()
    collection_names = [col.name for col in collections]
    
    if not collection_names:
        st.warning("Database terhubung, tetapi belum ada Koleksi (Collection) yang dibuat.")
    else:
        # Pilih Koleksi lewat Sidebar
        selected_koleksi = st.sidebar.selectbox("Pilih Koleksi:", collection_names)
        
        # Ambil data dari koleksi terpilih
        collection = client.get_collection(name=selected_koleksi)
        data = collection.get() # Mengambil semua isi dokumen
        
        # Tampilkan Jumlah Data keseluruhan
        st.metric(label="Total Dokumen", value=len(data['ids']))
        
        # Susun data ke dalam tabel agar rapi
        list_data = []
        for i in range(len(data['ids'])):
            row = {
                "ID": data['ids'][i],
                "Isi Dokumen (Text Chunk)": data['documents'][i] if data['documents'] else "Tidak ada teks",
                "Metadata": str(data['metadatas'][i]) if data['metadatas'] else "{}"
            }
            list_data.append(row)
            
        # Tampilkan secara visual di browser
        st.subheader(f"Daftar Isi Koleksi: `{selected_koleksi}`")
        st.dataframe(list_data, use_container_width=True)

except Exception as e:
    st.error(f"Gagal menghubungkan ke ChromaDB: {e}")
