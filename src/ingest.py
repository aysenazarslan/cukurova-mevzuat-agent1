import os
import sys
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma

# --- AYARLAR ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Dosya Yolları
DATA_PATH = os.path.join(parent_dir, 'data', 'pdfs') 
CHROMA_PATH = os.path.join(parent_dir, 'data', 'chroma_db')
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def create_vector_db():
    print(f"🚀 Veri sindirme işlemi başlıyor... (KESKİN NİŞANCI MODU)")
    print(f"📂 Hedef Klasör: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ HATA: Klasör bulunamadı!")
        return

    # 1. PDF'leri Yükle
    documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("❌ HATA: PDF dosyası yok!")
        return

    for file in pdf_files:
        pdf_path = os.path.join(DATA_PATH, file)
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            print(f"✅ Okundu: {file}")
        except Exception as e:
            print(f"❌ Hata ({file}): {e}")

    if not documents:
        return

    # 2. OPTİMİZE EDİLMİŞ PARÇALAMA (SNIPER AYARI)
    # 512 Karakter: Embedding modelinin tam odaklanabildiği en net boyut.
    # 100 Karakter: Örtüşme (Overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,     # Çok daha küçük ve net parçalar
        chunk_overlap=100,  # Bağlam kopmasın
        separators=["\nMadde", "\nMADDE", "\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Metinler {len(chunks)} adet 'keskin' parçaya bölündü (512/100).")

    # 3. Veritabanını Temizle ve Kur
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            print("🧹 Eski veritabanı temizlendi.")
        except PermissionError:
            print("🚨 HATA: Dosya kilitli! Lütfen çalışan tüm terminalleri kapatıp tekrar dene.")
            return

    print("💾 Embeddings oluşturuluyor...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"🎉 İŞLEM BAŞARILI! Veritabanı hazır.")

if __name__ == "__main__":
    create_vector_db()