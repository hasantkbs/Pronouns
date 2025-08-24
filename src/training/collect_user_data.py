# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Kişisel Veri Toplama Aracı

Bu script, belirli bir kullanıcıdan kişiselleştirilmiş model eğitimi için 
veri toplamak amacıyla kullanılır. Kullanıcıya bir dizi cümle okutur, 
sesini kaydeder ve bir metadata dosyası oluşturur.

Kullanım:
- python collect_user_data.py
- python collect_user_data.py --file /path/to/sentences.txt
"""

import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# --- Yapılandırma ---
TARGET_SAMPLING_RATE = 16000
BASE_DATA_PATH = "data/users"

# Varsayılan cümleler (eğer dosya belirtilmezse kullanılır)
DEFAULT_SENTENCES = [
    "Merhaba, nasılsın?",
    "Bugün hava çok güzel.",
    "Yarın ne yapacaksın?",
    "Bu sistemi test ediyorum.",
    "Lütfen bana yardım et.",
    "Ankara Türkiye'nin başkentidir.",
    "Kitap okumayı çok severim.",
    "Saat kaç?",
    "Alışverişe gitmem gerekiyor.",
    "İyi günler dilerim."
]

def get_sentences_from_file(file_path):
    """Verilen txt dosyasından cümleleri okur."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        print(f"✅ '{file_path}' dosyasından {len(sentences)} cümle başarıyla okundu.")
        return sentences
    except FileNotFoundError:
        print(f"❌ Hata: '{file_path}' dosyası bulunamadı.")
        return None
    except Exception as e:
        print(f"❌ Hata: Dosya okunurken bir sorun oluştu: {e}")
        return None

def get_user_id():
    """Kullanıcıdan bir kimlik alır veya varsayılanı kullanır."""
    user_id = input("Lütfen bir kullanıcı kimliği girin (örn: user_001): ").strip()
    if not user_id:
        user_id = "default_user"
        print(f"Giriş yapılmadı, varsayılan kimlik kullanılıyor: {user_id}")
    return user_id

def record_audio(duration, samplerate):
    """Belirtilen sürede ses kaydı yapar."""
    print("Kayıt başladı...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Kaydın bitmesini bekle
    print("Kayıt tamamlandı.")
    return recording

def main():
    """Ana veri toplama fonksiyonu."""
    parser = argparse.ArgumentParser(description="Kullanıcı verisi toplama aracı.")
    parser.add_argument("--file", type=str, help=".txt formatındaki cümleleri içeren dosyanın yolu.")
    args = parser.parse_args()

    if args.file:
        sentences_to_read = get_sentences_from_file(args.file)
        if sentences_to_read is None:
            return # Dosya okuma hatası durumunda çık
    else:
        sentences_to_read = DEFAULT_SENTENCES

    user_id = get_user_id()
    
    user_path = Path(BASE_DATA_PATH) / user_id
    audio_path = user_path / "audio"
    
    audio_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nHoş geldiniz, {user_id}!")
    print("Kişiselleştirilmiş model için veri toplama süreci başlıyor.")
    print(f"Lütfen aşağıdaki {len(sentences_to_read)} cümleyi okuyun.")
    print("Her cümleden sonra kayıt otomatik olarak başlayacak ve duracaktır.")
    
    metadata = []

    for i, sentence in enumerate(sentences_to_read):
        print("\n" + "="*50)
        print(f"Cümle {i+1}/{len(sentences_to_read)}: '{sentence}'")
        input("Hazır olduğunuzda ENTER tuşuna basın...")
        
        recording = record_audio(duration=5, samplerate=TARGET_SAMPLING_RATE)
        
        file_name = f"{user_id}_sentence_{i+1}.wav"
        file_path = audio_path / file_name
        
        sf.write(file_path, recording, TARGET_SAMPLING_RATE)
        print(f"Ses dosyası kaydedildi: {file_path}")
        
        metadata.append({
            "file_path": str(file_path.absolute()),
            "transcription": sentence
        })

    metadata_df = pd.DataFrame(metadata)
    metadata_file_path = user_path / "metadata.csv"
    metadata_df.to_csv(metadata_file_path, index=False, encoding='utf-8')
    
    print("\n" + "="*50)
    print("🎉 Veri toplama işlemi başarıyla tamamlandı!")
    print(f"Toplam {len(metadata)} adet ses kaydı ve transkript oluşturuldu.")
    print(f"Metadata dosyanız: {metadata_file_path}")
    print("\nŞimdi bu verileri kullanarak modeli kişiselleştirebilirsiniz.")

if __name__ == "__main__":
    main()
