# -*- coding: utf-8 -*-
import os
import json
from collections import Counter
import config

# Uyarı için minimum örnek sayısı eşiği
MIN_SAMPLES_THRESHOLD = 10

def analyze_distribution():
    """
    meta.json dosyasını okur ve her bir etiket (kelime) için
    kaç adet örnek olduğunu sayar, sonuçları sıralı bir şekilde yazdırır.
    """
    meta_path = os.path.join(config.BASE_PATH, config.USER_ID, "meta.json")
    if not os.path.exists(meta_path):
        print(f"Hata: Analiz edilecek meta.json dosyası bulunamadı: {meta_path}")
        print("Lütfen önce 01_calibrate.py script'ini çalıştırdığınızdan emin olun.")
        return

    print(f"--- '{meta_path}' dosyası analiz ediliyor ---")

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    labels = [item['etiket'] for item in meta_data.get('kelimeler', [])]

    if not labels:
        print("Meta dosyasında hiç etiket bulunamadı.")
        return

    # Her bir etiketin kaç kere tekrarlandığını say
    label_counts = Counter(labels)

    # Sayım sonuçlarını, en azdan en çoğa doğru sırala
    sorted_counts = sorted(label_counts.items(), key=lambda item: item[1])

    print("\n--- Veri Dağılım Raporu ---")
    print(f"Toplam {len(labels)} adet ses örneği bulundu.")
    print(f"Toplam {len(sorted_counts)} adet benzersiz kelime (etiket) bulundu.")
    print("-" * 30)

    low_data_warnings = 0
    for label, count in sorted_counts:
        warning = ""
        if count < MIN_SAMPLES_THRESHOLD:
            warning = f"  <-- UYARI: Çok az örnek ({MIN_SAMPLES_THRESHOLD}'den az)"
            low_data_warnings += 1
        print(f"Kelime: '{label}' \t\t Örnek Sayısı: {count}{warning}")

    print("-" * 30)
    print("\n--- Analiz Özeti ---")
    if low_data_warnings > 0:
        print(f"'{MIN_SAMPLES_THRESHOLD}' adetten daha az örneğe sahip {low_data_warnings} kelime tespit edildi.")
        print("Modelin bu kelimeleri öğrenmesi çok zordur ve genel doğruluğu düşürebilir.")
        print("Tavsiye: Bu kelimeler için daha fazla veri toplayın veya veri setinden çıkarmayı düşünün.")
    else:
        print("Tüm kelimeler için yeterli sayıda örnek var gibi görünüyor. Harika!")

if __name__ == "__main__":
    analyze_distribution()