# -*- coding: utf-8 -*-
"""
Raporlama Modülü

Bu modül, sistem performansını ve test sonuçlarını tarih damgalı raporlar
halinde kaydetmek için fonksiyonlar içerir.
"""

import os
from datetime import datetime

REPORTS_DIR = "reports"

def generate_report(report_data):
    """
    Verilen verilerle bir test raporu oluşturur ve kaydeder.

    Args:
        report_data (dict): Rapor içeriğini içeren bir sözlük. 
                              Örnek:
                              {
                                  "title": "ASR Model Test Sonuçları",
                                  "metrics": {
                                      "Veri Sayısı": 1500,
                                      "Kelime Hata Oranı (WER)": 0.15,
                                      "Cümle Hata Oranı (SER)": 0.25
                                  },
                                  "notes": "Bu test, genişletilmiş veri seti üzerinde yapılmıştır."
                              }
    """
    # Raporlar dizininin var olduğundan emin ol
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    # Dosya adını o günün tarihinden oluştur (örn: 2023-10-27.txt)
    today_str = datetime.now().strftime("%Y-%m-%d")
    file_name = f"{today_str}.txt"
    file_path = os.path.join(REPORTS_DIR, file_name)

    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            # Rapor başlığını ve zaman damgasını yaz
            f.write("="*60 + "\n")
            f.write(f"Rapor Başlığı: {report_data.get('title', 'Genel Rapor')}\n")
            f.write(f"Oluşturma Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")

            # Metrikleri yaz
            if 'metrics' in report_data and isinstance(report_data['metrics'], dict):
                f.write("--- Performans Metrikleri ---\
")
                for key, value in report_data['metrics'].items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")

            # Notları veya açıklamaları yaz
            if 'notes' in report_data:
                f.write("--- Notlar ---\
")
                f.write(f"{report_data['notes']}\n\n")
            
            f.write("\n" + "-"*60 + "\n\n")

        print(f"✅ Rapor başarıyla '{file_path}' dosyasına eklendi.")

    except IOError as e:
        print(f"❌ Hata: Rapor dosyası yazılırken bir sorun oluştu: {e}")

if __name__ == '__main__':
    # Örnek bir rapor oluşturma
    print("Örnek bir rapor oluşturuluyor...")
    example_data = {
        "title": "Örnek ASR Model Değerlendirmesi",
        "metrics": {
            "Test Veri Seti Boyutu": 500,
            "Kelime Hata Oranı (WER)": "12.5%",
            "Model Versiyonu": "v1.2-beta"
        },
        "notes": "Bu rapor, 'test_utils' içinden örnek bir çağrımı göstermektedir."
    }
    generate_report(example_data)
