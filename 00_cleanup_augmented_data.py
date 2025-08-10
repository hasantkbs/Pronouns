# -*- coding: utf-8 -*-
import os
import glob
import config

def main():
    """
    Veri artırma (augmentation) ile oluşturulmuş sentetik ses dosyalarını temizler.
    """
    user_path = os.path.join(config.BASE_PATH, config.USER_ID)
    if not os.path.exists(user_path):
        print(f"Hata: Kullanıcı dizini bulunamadı: {user_path}")
        return

    # Silinecek dosyaları bulmak için bir arama deseni (pattern) oluştur
    # İçinde '_aug' geçen tüm .wav dosyalarını hedefler
    search_pattern = os.path.join(user_path, "*_aug*.wav")

    # Desenle eşleşen tüm dosyaların bir listesini al
    files_to_delete = glob.glob(search_pattern)

    if not files_to_delete:
        print("Temizlenecek artırılmış (_aug) ses dosyası bulunamadı.")
        return

    print(f"--- {len(files_to_delete)} adet artırılmış ses dosyası silinecek ---")

    deleted_count = 0
    for file_path in files_to_delete:
        try:
            print(f"Siliniyor: {file_path}")
            os.remove(file_path)
            deleted_count += 1
        except OSError as e:
            print(f"Hata: {file_path} silinirken bir sorun oluştu: {e}")
    
    print(f"\n--- Temizlik Tamamlandı ---")
    print(f"Toplam {deleted_count} adet dosya başarıyla silindi.")

if __name__ == "__main__":
    main()