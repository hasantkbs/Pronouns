# -*- coding: utf-8 -*-
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import glob
import config

def get_next_file_index(user_path, word):
    """
    Belirli bir kelime için mevcut dosya sayısını bulur ve bir sonraki dosya numarasını döndürür.
    Örn: elma_1.wav, elma_2.wav varsa 3 döndürür.
    """
    search_pattern = os.path.join(user_path, f"{word}_*.wav")
    existing_files = glob.glob(search_pattern)
    if not existing_files:
        return 1

    max_index = 0
    for f in existing_files:
        try:
            # Dosya adından indeksi çıkar (örn: elma_1.wav -> 1)
            index_str = os.path.splitext(f)[0].split('_')[-1]
            index = int(index_str)
            if index > max_index:
                max_index = index
        except (ValueError, IndexError):
            # Format uymuyorsa veya sayı değilse görmezden gel
            continue

    return max_index + 1

def record_audio(duration, sample_rate):
    """Belirtilen süre boyunca ses kaydı yapar."""
    print("Kayıt başladı...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Kaydın bitmesini bekle
    print("Kayıt tamamlandı.")
    return recording.flatten()

def main():
    user_path = os.path.join(config.BASE_PATH, config.USER_ID)
    os.makedirs(user_path, exist_ok=True)

    print("--- Veri Toplama Yardımcısı ---")
    print("Bu script, belirlediğiniz kelimeler için yeni ses kayıtları yapmanızı sağlar.")
    print("Çıkmak için kelime sorulduğunda 'q' yazıp Enter'a basın.")

    while True:
        word_to_record = input("\n>>> Veri eklemek istediğiniz kelimeyi yazın: ").strip().lower()

        if word_to_record == 'q':
            print("Veri toplama sonlandırıldı. Hoşça kalın!")
            break

        if not word_to_record:
            print("Lütfen geçerli bir kelime girin.")
            continue

        # Bu kelime için kaçıncı örneği kaydedeceğimizi bul
        next_index = get_next_file_index(user_path, word_to_record)
        new_filename = f"{word_to_record}_{next_index}.wav"
        new_filepath = os.path.join(user_path, new_filename)

        print(f"\n'{word_to_record}' kelimesi için kayıt yapılacak.")
        print(f"Dosya adı '{new_filename}' olacak.")

        # Geri sayım
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)

        # Kaydı yap
        recorded_audio = record_audio(config.KAYIT_SURESI_SN, config.ORNEKLEME_ORANI)

        # Kaydı dosyaya yaz
        try:
            sf.write(new_filepath, recorded_audio, config.ORNEKLEME_ORANI)
            print(f"Başarılı: '{new_filepath}' dosyası kaydedildi.")
        except Exception as e:
            print(f"Hata: Dosya kaydedilirken bir sorun oluştu: {e}")
if __name__ == "__main__":
    main()