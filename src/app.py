import streamlit as st
import os
import sys
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --- AYARLAR ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME
except ImportError:
    CHROMA_DB_DIR = os.path.join(parent_dir, 'data', 'chroma_db')
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

# --- KAYNAKLAR ---
if not DEEPSEEK_API_KEY and __name__ == "__main__":
    st.error("API Key Eksik!")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
llm = ChatOpenAI(
    model=MODEL_NAME, 
    api_key=DEEPSEEK_API_KEY, 
    base_url=BASE_URL, 
    temperature=0.1 
)

# ---------------------------------------------------------
# ⭐ FINAL TURBO BEYİN (V5 - BIG CONTEXT) ⭐
# ---------------------------------------------------------
def get_response_from_deepseek(question):
    """
    RAG Stratejisi:
    - Küçük parçalar (512 char) kullanıyoruz ama SAYIYI ARTIRIYORUZ.
    - k=25: Yaklaşık 3-4 sayfalık yoğun bilgi verir.
    - lambda_mult=0.5: Çeşitliliği artırır, sadece benzerleri değil, 
      farklı yerlerdeki bilgileri de toplar.
    """
    # 1. MMR ARAMA (Genişletilmiş)
    docs = vector_db.max_marginal_relevance_search(
        question, 
        k=25,           # ARTTIRILDI: Modele daha fazla kanıt sunuyoruz.
        fetch_k=60,     # ARTTIRILDI: Daha geniş havuzdan seçiyor.
        lambda_mult=0.5 # DENGELENDİ: Hem benzerlik hem çeşitlilik (0.5 ideal).
    )
    
    chunk_texts = [d.page_content for d in docs]
    context_text = "\n---\n".join(chunk_texts)
    
    # 2. PROMPT (Analitik ve Esnek)
    system_prompt = ChatPromptTemplate.from_template("""
    Sen Çukurova Üniversitesi mevzuatlarında uzman, son derece dikkatli bir akademik asistansın.
    Aşağıdaki MEVZUAT PARÇALARINI (CONTEXT) incele ve soruya cevap ver.

    CONTEXT (KANITLAR):
    {context}
    
    SORU: {question}

    ANALİZ ADIMLARI:
    1. Sorudaki anahtar kelimelerin (örn: "GNO", "Mezuniyet", "Yaz okulu") eş anlamlılarını metinde ara.
    2. Cevabı parçaları birleştirerek oluştur. Tek bir maddede bulamayabilirsin.
    3. Sayısal verileri (gün, yüzde, not) kesinlikle metinden doğrula.
    4. Eğer metinde "Açılmaz" yazıyorsa bunu "Yoktur" olarak yorumla (Mantıksal Çıkarım).
    5. Cevabı kısa ve net ver.
    6. Bilgi kesinlikle yoksa "Yönetmeliklerde bu bilgi bulunmamaktadır" de.
    
    CEVAP:
    """)
    
    chain = system_prompt | llm
    response = chain.invoke({"context": context_text, "question": question})
    
    return response.content

# ---------------------------------------------------------
# STREAMLIT ARAYÜZÜ
# ---------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="ÇÜ Asistan", page_icon="🎓", layout="wide")
    st.title("🎓 Çukurova Mevzuat Asistanı (Turbo Mode)")
    
    with st.sidebar:
        st.success("Mod: Turbo (512 Chunk x 25)")
        st.info("Kapasite: Yüksek Bağlam")
        if st.button("Sıfırla"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Soruları bekliyorum Kingo! 👑"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.status("🧠 Geniş kapsamlı tarama yapılıyor..."):
                resp = get_response_from_deepseek(prompt)
            st.write(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})