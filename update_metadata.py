import pandas as pd
import os
import glob
from pathlib import Path
import re

def update_metadata_words(user_id):
    user_data_path = Path("data/users") / user_id
    words_audio_path = user_data_path / "words"
    metadata_path = user_data_path / "metadata_words.csv"

    if not words_audio_path.exists():
        print(f"❌ Hata: {words_audio_path} klasörü bulunamadı.")
        return

    # 1. Parse existing metadata_words.csv to get word mappings
    word_to_transcription = {}
    if metadata_path.exists():
        existing_df = pd.read_csv(metadata_path)
        for _, row in existing_df.iterrows():
            filename = Path(row["file_path"]).name
            match = re.match(r"Furkan_kelime_(\d+)_rep\d+\.wav", filename)
            if match:
                kelime_id = f"Furkan_kelime_{match.group(1)}"
                word_to_transcription[kelime_id] = row["transcription"]
    
    if not word_to_transcription:
        print("❌ Hata: Mevcut metadata_words.csv dosyasından kelime-transkripsiyon eşleşmeleri çıkarılamadı.")
        print("Lütfen metadata_words.csv dosyasının doğru formatta olduğundan emin olun.")
        return

    # 2. List all .wav files
    wav_files = list(words_audio_path.glob("*.wav"))
    if not wav_files:
        print(f"❌ Hata: {words_audio_path} içinde hiç .wav dosyası bulunamadı.")
        return

    # 3. Generate new data
    new_records = []
    for wav_file in wav_files:
        filename = wav_file.name
        match = re.match(r"Furkan_kelime_(\d+)_rep(\d+)\.wav", filename)
        if match:
            kelime_id = f"Furkan_kelime_{match.group(1)}"
            repetition = int(match.group(2))
            transcription = word_to_transcription.get(kelime_id)
            
            if transcription is not None:
                new_records.append({
                    "file_path": str(wav_file),
                    "transcription": transcription,
                    "repetition": repetition
                })
            else:
                print(f"⚠️  Uyarı: {kelime_id} için transkripsiyon bulunamadı. Dosya atlanıyor: {filename}")
        else:
            print(f"⚠️  Uyarı: Beklenmeyen dosya adı formatı. Dosya atlanıyor: {filename}")

    if not new_records:
        print("❌ Hata: Yeni metadata kaydı oluşturulamadı.")
        return

    # 4. Write new metadata_words.csv
    new_df = pd.DataFrame(new_records)
    new_df = new_df.sort_values(by=["file_path"]).reset_index(drop=True)
    new_df.to_csv(metadata_path, index=False)
    print(f"✅ {metadata_path} başarıyla güncellendi. Toplam {len(new_df)} kayıt.")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Kullanıcıya özel metadata_words.csv dosyasını günceller.")
    parser.add_argument("user_id", type=str, help="Metadata dosyası güncellenecek kullanıcının kimliği.")
    args = parser.parse_args()
    update_metadata_words(args.user_id)

if __name__ == "__main__":
    main()
