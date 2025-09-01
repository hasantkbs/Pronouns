# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Sentetik Veri Oluşturma Aracı

Bu script, `collect_word_data.py` ile toplanan tekil kelime kayıtlarını kullanarak
sentetik cümleler oluşturur ve bunları ana veri setine ekler.
Bu bir veri artırma (data augmentation) tekniğidir.

Kullanım:
- python augment_from_words.py user_001 --num-sentences 100
"""

import os
import pandas as pd
from pathlib import Path
import argparse
import random

try:
    from pydub import AudioSegment
except ImportError:
    print("❌ Hata: pydub kütüphanesi bulunamadı.")
    print("Lütfen `pip install pydub` komutu ile kurun.")
    exit()

# --- Yapılandırma ---
BASE_DATA_PATH = "data/users"

def get_last_sentence_index(df):
    """Mevcut metadata'daki son 'sentence' index'ini bulur."""
    sentence_files = df[df['file_path'].str.contains("_sentence_")]
    if sentence_files.empty:
        return 0
    last_index = sentence_files['file_path'].str.extract(r'_sentence_(\d+).wav').astype(int).max().iloc[0]
    return last_index if pd.notna(last_index) else 0

def main():
    """Ana veri artırma fonksiyonu."""
    parser = argparse.ArgumentParser(description="Tekil kelimelerden sentetik veri oluşturma aracı.")
    parser.add_argument("user_id", type=str, help="Veri artırma yapılacak kullanıcının kimliği.")
    parser.add_argument("--num-sentences", type=int, required=True, help="Oluşturulacak sentetik cümle sayısı.")
    parser.add_argument("--min-words", type=int, default=3, help="Bir cümlede olacak minimum kelime sayısı.")
    parser.add_argument("--max-words", type=int, default=8, help="Bir cümlede olacak maksimum kelime sayısı.")
    args = parser.parse_args()

    # --- Dosya Yollarını Ayarla ---
    user_path = Path(BASE_DATA_PATH) / args.user_id
    words_metadata_path = user_path / "metadata_words.csv"
    main_metadata_path = user_path / "metadata.csv"
    main_audio_path = user_path / "audio"

    # --- Gerekli Dosyaları Kontrol Et ---
    if not words_metadata_path.exists():
        print(f"❌ Hata: '{words_metadata_path}' bulunamadı.")
        print("Önce `collect_word_data.py` script'ini çalıştırarak kelime verisi toplayın.")
        return

    main_audio_path.mkdir(parents=True, exist_ok=True)

    # --- Verileri Yükle ---
    print("Kelime verileri okunuyor...")
    word_df = pd.read_csv(words_metadata_path)
    
    if main_metadata_path.exists():
        main_df = pd.read_csv(main_metadata_path)
    else:
        main_df = pd.DataFrame(columns=["file_path", "transcription"])

    print(f"Toplam {len(word_df)} adet tekil kelime kaydı bulundu.")

    # --- Sentetik Cümleleri Oluştur ---
    print(f"\n{args.num_sentences} adet sentetik cümle oluşturuluyor...")
    
    new_metadata_rows = []
    last_augmented_index = main_df[main_df['file_path'].str.contains("_augmented_")]\
        .file_path.str.extract(r'_augmented_(\d+).wav').astype(int).max().iloc[0] if not main_df.empty and main_df[main_df['file_path'].str.contains("_augmented_")].any().any() else 0
    last_augmented_index = last_augmented_index if pd.notna(last_augmented_index) else 0

    for i in range(1, args.num_sentences + 1):
        num_words_to_combine = random.randint(args.min_words, args.max_words)
        
        # Rastgele kelimeler seç
        sampled_words = word_df.sample(n=num_words_to_combine, replace=True)
        
        combined_audio = AudioSegment.empty()
        combined_transcription = []
        
        # Sesleri ve metinleri birleştir
        for _, row in sampled_words.iterrows():
            word_audio_path = Path(row['file_path'])
            if word_audio_path.exists():
                segment = AudioSegment.from_wav(word_audio_path)
                combined_audio += segment
                combined_transcription.append(row['transcription'])
            else:
                print(f"⚠️ Uyarı: '{word_audio_path}' bulunamadı, atlanıyor.")

        if not combined_audio:
            continue

        # Yeni dosyayı ve metni oluştur
        new_transcription = " ".join(combined_transcription)
        new_index = last_augmented_index + i
        new_filename = f"{args.user_id}_augmented_{new_index}.wav"
        new_filepath = main_audio_path / new_filename
        
        # Sesi dışa aktar
        combined_audio.export(new_filepath, format="wav")
        
        new_metadata_rows.append({
            "file_path": str(new_filepath.absolute()),
            "transcription": new_transcription
        })
        
        print(f"Oluşturuldu ({i}/{args.num_sentences}): {new_filename} -> '{new_transcription}'")

    # --- Ana Metadata Dosyasını Güncelle ---
    if new_metadata_rows:
        new_rows_df = pd.DataFrame(new_metadata_rows)
        updated_df = pd.concat([main_df, new_rows_df], ignore_index=True)
        updated_df.to_csv(main_metadata_path, index=False, encoding='utf-8')

    print("\n" + "="*50)
    print("🎉 Veri artırma işlemi başarıyla tamamlandı!")
    print(f"Ana veri setine {len(new_metadata_rows)} adet sentetik cümle eklendi.")
    print(f"Güncel metadata dosyanız: {main_metadata_path}")

if __name__ == "__main__":
    main()
