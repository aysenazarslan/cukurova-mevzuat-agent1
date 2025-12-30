import json
import pandas as pd
import time
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# --- YOL AYARI (PATH FIX) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Modül import kontrolü
try:
    from src.app import get_response_from_deepseek
    print("✅ BAŞARILI: src.app dosyasına erişildi.")
except ImportError as e:
    print("\n🚨 KRİTİK HATA: src.app dosyası bulunamadı!")
    print(f"Hata Detayı: {e}")
    sys.exit(1)

load_dotenv()

# DeepSeek Hakem Yapılandırması
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def evaluate_with_judge(question, ground_truth, model_answer):
    """
    Hakem Yapay Zeka cevabı 5 kriter üzerinden değerlendirir.
    Her başarılı kriter 1 puandır. Toplam 5 üzerinden puanlanır.
    """
    judge_prompt = f"""
    Sen akademik bir değerlendirme uzmanısın. Aşağıdaki öğrenci sorusuna verilen Yapay Zeka cevabını, 
    Referans Cevap (Ground Truth) ile karşılaştırarak değerlendireceksin.
    
    SORU: {question}
    REFERANS CEVAP: {ground_truth}
    YZ CEVABI: {model_answer}
    
    Lütfen cevabı şu 5 kriter üzerinden analiz et. Her kriter 1 puandır:
    1. Erişim (Retrieval): Doğru bilgi bulunmuş mu? (+1 Puan)
    2. Doğruluk (Precision): Sayısal veriler ve tarihler doğru mu? (+1 Puan)
    3. Kapsam (Completeness): Cevap eksiksiz mi? (+1 Puan)
    4. Mantık (Reasoning): Koşullu durumlar doğru yorumlanmış mı? (+1 Puan)
    5. Dürüstlük (Honesty): Halüsinasyon (uydurma) yok mu? (+1 Puan)
    
    DEĞERLENDİRME KURALI:
    - Toplam puan 0 ile 5 arasında bir tam sayı olmalıdır.
    - 3 ve üzeri puanlar "BAŞARILI", 3'ün altı "BAŞARISIZ" sayılır.
    
    Çıktı Formatı (Sadece bu JSON formatını döndür):
    {{
        "Puan": (0-5 arası tam sayı),
        "Gerekçe": "Kısa bir değerlendirme cümlesi",
        "Durum": "BAŞARILI" veya "BAŞARISIZ"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Sen adil bir hakemsin. Sadece JSON döndür."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Hakem Hatası: {e}")
        return {"Puan": 0, "Gerekçe": "Hata oluştu", "Durum": "HATA"}

def main():
    print("🚀 Benchmark Testi Başlatılıyor... (5'lik Sistem)")
    
    # Veri Setini Yükle
    data_path = os.path.join(current_dir, "benchmark_data.json")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"📂 {len(data)} adet soru yüklendi.")
    except FileNotFoundError:
        print("❌ HATA: 'benchmark_data.json' bulunamadı!")
        return

    results = []
    total_score = 0
    passed_count = 0
    
    for index, item in enumerate(data):
        q_id = item.get("id", index+1)
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"\n[{index+1}/{len(data)}] Soru İşleniyor: {question[:40]}...")
        
        try:
            start_time = time.time()
            model_response = get_response_from_deepseek(question) 
            duration = time.time() - start_time
        except Exception as e:
            model_response = f"HATA: {str(e)}"
            duration = 0

        eval_result = evaluate_with_judge(question, ground_truth, model_response)
        
        score = eval_result.get("Puan", 0)
        total_score += score
        status = eval_result.get("Durum", "BELİRSİZ")
        
        # Kod tarafında da garanti kontrol (3 altı kalır)
        if score < 3:
            status = "BAŞARISIZ ❌"
        else:
            status = "BAŞARILI ✅"
            passed_count += 1
            
        results.append({
            "ID": q_id,
            "Soru": question,
            "Referans Cevap": ground_truth,
            "Model Cevabı": model_response,
            "Puan (0-5)": score,
            "Durum": status,
            "Gerekçe": eval_result.get("Gerekçe", ""),
            "Süre (sn)": round(duration, 2)
        })

    if len(results) > 0:
        avg_score = total_score / len(results)
        success_rate = (passed_count / len(results)) * 100
        
        df = pd.DataFrame(results)
        output_file = os.path.join(current_dir, "deepseek_final_sonuc_v3.xlsx")
        df.to_excel(output_file, index=False)
        
        print("\n" + "="*50)
        print(f"🎉 TEST TAMAMLANDI! (5 Üzerinden)")
        print(f"📊 Ortalama Puan: {avg_score:.2f} / 5.00")
        print(f"📈 Başarı Oranı: %{success_rate:.1f} (Geçen Soru Sayısı: {passed_count}/{len(results)})")
        print(f"📄 Rapor Kaydedildi: {output_file}")
        print("="*50)
        
        if avg_score >= 4.5:
            print("🏆 MÜKEMMEL! We killed it! 🔥")
        elif avg_score >= 3.5:
            print("✅ GAYET İYİ. Hoca beğenir.")
        else:
            print("⚠️ KRİTİK. Ortalamayı yükseltmemiz lazım.")

if __name__ == "__main__":
    main()