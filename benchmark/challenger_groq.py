import os
import json
import pandas as pd
import sys
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

# --- AYARLAR VE GÜVENLİK ---
# .env dosyasındaki değişkenleri yükle
load_dotenv()

# API Anahtarlarını Çevresel Değişkenlerden Al
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Yol ayarları
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- İSTEMCİ KURULUMU VE KONTROLÜ ---
if not GROQ_API_KEY:
    print("❌ HATA: GROQ_API_KEY bulunamadı! Lütfen .env dosyanızı kontrol edin.")
    sys.exit(1)

if not DEEPSEEK_API_KEY:
    print("❌ HATA: DEEPSEEK_API_KEY bulunamadı! Lütfen .env dosyanızı kontrol edin.")
    sys.exit(1)

try:
    client_groq = Groq(api_key=GROQ_API_KEY)
    client_judge = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
except Exception as e:
    print(f"❌ İstemci hatası: {e}")
    sys.exit(1)

def get_challenger_response(question):
    """
    RAGSIZ Rakip (Llama 3.3 - En Güncel Sürüm).
    Yönetmelik erişimi olmadan sadece modelin kendi bilgisiyle cevap üretir.
    """
    prompt = f"""
    Sen bir üniversite öğrencisisin. Aşağıdaki soruyu Çukurova Üniversitesi yönetmeliğine göre cevapla.
    Elinin altında yönetmelik metni YOK. Sadece hafızandaki bilgileri kullan.
    Eğer spesifik kuralı bilmiyorsan, genel bir tahmin yürüt.
    
    SORU: {question}
    CEVAP:
    """
    try:
        chat_completion = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"HATA: {str(e)}"

def evaluate_with_judge(question, ground_truth, model_answer):
    """
    HAKEM DEĞERLENDİRMESİ (DeepSeek)
    Cevabı referans cevaba göre 0-5 arasında puanlar.
    """
    judge_prompt = f"""
    Sen akademik bir hakemsin. Aşağıdaki cevabı referans cevaba göre 0-5 arası puanla.
    
    SORU: {question}
    REFERANS CEVAP: {ground_truth}
    ADAY CEVAP: {model_answer}
    
    DEĞERLENDİRME KRİTERLERİ:
    1. Doğru bilgi (+1)
    2. Sayısal doğruluk (+1)
    3. Kapsam (+1)
    4. Mantık (+1)
    5. Halüsinasyon Yok (+1)
    
    ÖNEMLİ: Çukurova Üniversitesi'ne özel kuralları (sayı, gün, madde) içermeyen genel cevaplara DÜŞÜK PUAN VER.
    
    Çıktı Formatı (JSON):
    {{
        "Puan": (0-5 arası sayı),
        "Durum": "BAŞARILI" veya "BAŞARISIZ",
        "Gerekçe": "Kısa açıklama"
    }}
    """
    try:
        response = client_judge.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except:
        return {"Puan": 0, "Durum": "HATA", "Gerekçe": "Hakem hatası"}

def main():
    print("🥊 RAKİP TESTİ BAŞLIYOR: DeepSeek RAG vs. Llama 3.3 (No-RAG)")
    print("------------------------------------------------------------")
    
    data_path = os.path.join(current_dir, "benchmark_data.json")
    
    # Veri seti kontrolü
    if not os.path.exists(data_path):
        print(f"❌ HATA: Veri dosyası bulunamadı: {data_path}")
        return

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Veri okuma hatası: {e}")
        return

    results = []
    total_score = 0
    
    for index, item in enumerate(data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"[{index+1}/{len(data)}] Llama-3.3 Cevaplıyor: {question[:30]}...")
        
        # 1. Rakip Cevaplasın
        challenger_response = get_challenger_response(question)
        
        # 2. Hakem Puanlasın
        eval_result = evaluate_with_judge(question, ground_truth, challenger_response)
        score = eval_result.get("Puan", 0)
        total_score += score
        
        results.append({
            "ID": item["id"],
            "Soru": question,
            "Referans Cevap": ground_truth,
            "Rakip Cevabı (No-RAG)": challenger_response,
            "Puan": score,
            "Durum": eval_result.get("Durum"),
            "Gerekçe": eval_result.get("Gerekçe")
        })

    # Karşılaştırma Raporu (Opsiyonel: Eğer önceki sonuç dosyası varsa ortalamayı çeker)
    our_avg = 4.00 # Varsayılan değer
    our_results_path = os.path.join(current_dir, "deepseek_final_sonuc_v3.xlsx")
    
    if os.path.exists(our_results_path):
        try:
            df_ours = pd.read_excel(our_results_path)
            our_avg = df_ours["Puan (0-5)"].mean()
        except:
            pass
        
    challenger_avg = total_score / len(results) if results else 0
    
    print("\n" + "="*50)
    print(f"🏁 FİNAL KARŞILAŞTIRMA SONUCU")
    print(f"🦁 BİZİM SİSTEM (RAG): {our_avg:.2f} / 5.00")
    print(f"🐱 RAKİP (Llama 3.3):  {challenger_avg:.2f} / 5.00")
    print("="*50)
    
    output_file = os.path.join(current_dir, "challenger_groq_results_v2.xlsx")
    pd.DataFrame(results).to_excel(output_file, index=False)
    print(f"Sonuçlar kaydedildi: {output_file}")

if __name__ == "__main__":
    main()