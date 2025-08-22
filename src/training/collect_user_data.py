
# -*- coding: utf-8 -*-
import os
import csv
from utils import record_audio

# --- Ayarlar ---
CUMLELER_FILE = "cumleler.txt"
OUTPUT_DIR = "users"

def main():
    """Kullanıcıya özel ses verisi toplama script'i."""
    print("--- Kullanıcı Sesi Veri Toplama Aracı ---")

    # 1. Cümle dosyasını kontrol et
    if not os.path.exists(CUMLELER_FILE):
        print(f"HATA: '{CUMLELER_FILE}' dosyası bulunamadı.")
        # Örnek bir dosya oluştur
        with open(CUMLELER_FILE, "w", encoding="utf-8") as f:
            f.write("bu birinci örnek cümledir\n")
            f.write("bu da ikinci örnek cümledir\n")
        print(f"Lütfen kayıt yapmak istediğiniz cümleleri bu dosyaya ekleyin.")
        print(f"Örnek bir '{CUMLELER_FILE}' dosyası oluşturuldu. Lütfen içini doldurup script'i tekrar çalıştırın.")
        return

    # 2. Kullanıcı bilgilerini al
    user_name = input("Lütfen kullanıcı adınızı girin (örn: hasan): ").strip()
    if not user_name:
        print("Geçerli bir kullanıcı adı girmelisiniz.")
        return

    user_path = os.path.join(OUTPUT_DIR, user_name)
    audio_path = os.path.join(user_path, "audio")
    os.makedirs(audio_path, exist_ok=True)
    print(f"Verileriniz '{user_path}' klasörüne kaydedilecek.")

    # 3. Cümleleri oku
    with open(CUMLELER_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    if not sentences:
        print(f"'{CUMLELER_FILE}' dosyasında okunacak cümle bulunamadı.")
        return

    # 4. Metadata dosyasını hazırla
    metadata_path = os.path.join(user_path, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "transcription"]) # Başlık satırı

        print(f"\nToplam {len(sentences)} adet cümle kaydedilecek.")
        
        # 5. Kayıt döngüsünü başlat
        for i, sentence in enumerate(sentences):
            print("\n------------------------------------------")
            print(f"Cümle {i+1}/{len(sentences)}: '{sentence}'")
            
            file_name = f"audio/{i+1}.wav"
            output_file_path = os.path.join(user_path, file_name)
            
            prompt = f"▶️  Yukarıdaki cümleyi okumak için ENTER'a basın..."
            record_audio(file_path=output_file_path, record_seconds=5, prompt=prompt)

            # Metadata'ya kaydet
            writer.writerow([file_name, sentence])

    print("\n=========================================")
    print("🎉 Veri toplama işlemi başarıyla tamamlandı!")
    print(f"Tüm kayıtlarınız ve '{metadata_path}' dosyanız oluşturuldu.")
    print("Artık bu veri setini kullanarak modeli yeniden eğitebilirsiniz.")

if __name__ == "__main__":
    main()
