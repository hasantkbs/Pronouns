# -*- coding: utf-8 -*-

import os
import json
import re
import config  # Ayar dosyasını import et


def main():
    """
    Ana kalibrasyon fonksiyonu.
    Mevcut .wav dosyalarından meta.json oluşturur.
    """
    user_path = os.path.join(os.getcwd(), config.BASE_PATH, config.USER_ID)
    if not os.path.exists(user_path):
        print(f"Hata: Kullanıcı dizini bulunamadı: {user_path}")
        return

    meta_dosya_yolu = os.path.join(user_path, "meta.json")
    meta_data = {"user_id": config.USER_ID, "kelimeler": []}

    print(f"--- Kullanıcı '{config.USER_ID}' için meta.json oluşturuluyor ---")

    for dosya_adi in os.listdir(user_path):
        if dosya_adi.endswith(".wav"):
            # Dosya adından etiketi çıkar (ör: "elma_1.wav" -> "elma")
            etiket_match = re.match(r"(.+)_\d+\.wav", dosya_adi)
            if etiket_match:
                etiket = etiket_match.group(1)
                meta_data["kelimeler"].append({"etiket": etiket, "dosya": dosya_adi})
                print(f"  '{dosya_adi}' için etiket: '{etiket}'")

    if not meta_data["kelimeler"]:
        print("Uyarı: Hiç .wav dosyası bulunamadı veya dosyalar isimlendirme formatına uymuyor (kelime_sayı.wav).")
        return

    with open(meta_dosya_yolu, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)

    print(f"\nKalibrasyon tamamlandı! '{meta_dosya_yolu}' dosyası oluşturuldu/güncellendi.")


if __name__ == "__main__":
    main()
