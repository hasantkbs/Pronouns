# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Birleşik Veri Toplama Aracı

Bu interaktif script, hem cümle hem de kelime bazlı veri toplamayı yönetir.
Kullanıcıya ne tür bir kayıt yapmak istediğini sorar, ilgili dosya setlerinden
birini seçtirir ve kayıt işlemini başlatır.

Kullanım:
- python collect_data.py
"""

import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path

# --- Yapılandırma ---
TARGET_SAMPLING_RATE = 16000
BASE_DATA_PATH = "data/users"

def select_from_list(items, prompt):
    """Verilen listeden bir öğe seçmek için kullanıcıya bir menü gösterir."""
    print(prompt)
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item.name}")
    
    while True:
        try:
            choice = int(input("Lütfen seçiminizi yapın (sayı olarak): "))
            if 1 <= choice <= len(items):
                return items[choice - 1]
            else:
                print("Geçersiz seçim, lütfen listedeki bir sayıyı girin.")
        except ValueError:
            print("Lütfen bir sayı girin.")

def get_files_from_dir(directory_path):
    """Belirtilen dizindeki .txt dosyalarını bulur."""
    path = Path(directory_path)
    if not path.exists():
        print(f"❌ Hata: '{directory_path}' dizini bulunamadı.")
        return []
    return sorted(list(path.glob("*.txt")))

def get_lines_from_file(file_path):
    """Verilen txt dosyasından satırları (cümle/kelime) okur."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"✅ '{file_path.name}' dosyasından {len(lines)} satır başarıyla okundu.")
        return lines
    except Exception as e:
        print(f"❌ Hata: Dosya okunurken bir sorun oluştu: {e}")
        return None

def get_user_id():
    """Kullanıcıdan bir kimlik alır."""
    user_id = input("Lütfen bir kullanıcı kimliği girin (örn: user_001): ").strip()
    if not user_id:
        raise ValueError("Kullanıcı kimliği boş bırakılamaz.")
    return user_id

def record_audio(duration, samplerate):
    """Belirtilen sürede ses kaydı yapar."""
    print("▶️  Kayıt başladı...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Kaydın bitmesini bekle
    print("⏹️  Kayıt tamamlandı.")
    return recording

def run_recording_session(user_id, items_to_record, save_path, metadata_path, item_type):
    """Cümle veya kelime kayıt oturumunu yürütür."""
    save_path.mkdir(parents=True, exist_ok=True)
    metadata = []
    quit_session = False

    # Üzerine yazmayı önlemek için başlangıç indeksini belirle
    start_index = 0
    if save_path.exists():
        existing_indices = []
        for f in save_path.glob(f"{user_id}_{item_type}_*.wav"):
            try:
                # Dosya adının sonundaki sayıyı al
                index_str = f.stem.split('_')[-1]
                existing_indices.append(int(index_str))
            except (ValueError, IndexError):
                continue # Formatla uyuşmayan dosyaları atla
        if existing_indices:
            start_index = max(existing_indices)
    
    print(f"\n{len(items_to_record)} adet {item_type} kaydedilecek.")
    if start_index > 0:
        print(f"Bilgi: Mevcut kayıtlar bulundu. Numaralandırma {start_index + 1}'den başlayacak.")

    try:
        for i, item in enumerate(items_to_record):
            current_index = start_index + i + 1
            print("\n" + "="*50)
            print(f"{item_type.capitalize()} {i+1}/{len(items_to_record)} (Dosya No: {current_index}): '{item}'")
            
            for rep_num in range(1, 4): # Record 3 times
                print(f"   ➡️ Tekrar {rep_num}/3: '{item}' için kayıt...")
                
                user_input = input("   Hazır olduğunuzda ENTER'a basın (çıkmak için 'q' yazıp ENTER'a basın): ")
                if user_input.lower() == 'q':
                    quit_session = True
                    break

                duration = 20 if item_type == "cümle" else 3 # Still use 3 for words
                rec = record_audio(duration=duration, samplerate=TARGET_SAMPLING_RATE)
                
                file_name = f"{user_id}_{item_type}_{current_index}_rep{rep_num}.wav"
                file_path = save_path / file_name
                
                sf.write(file_path, rec, TARGET_SAMPLING_RATE)
                print(f"   ✅ Ses dosyası kaydedildi: {file_path}")
                
                metadata.append({
                    "file_path": str(file_path.absolute()),
                    "transcription": item,
                    "repetition": rep_num # Add repetition info
                })
            
            if quit_session:
                break
        
        if not quit_session:
            print("\n" + "="*50)
            print(f"🎉 {item_type.capitalize()} toplama işlemi başarıyla tamamlandı!")

    finally:
        if metadata:
            print("\n🛑 Kayıt durduruluyor. Toplanan veriler CSV dosyasına yazılıyor...")
            # Mevcut metadata dosyasını oku ve yeni verileri ekle
            if metadata_path.exists():
                try:
                    existing_df = pd.read_csv(metadata_path)
                    new_df = pd.DataFrame(metadata)
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                except pd.errors.EmptyDataError:
                    updated_df = pd.DataFrame(metadata) # Eğer dosya boşsa
            else:
                updated_df = pd.DataFrame(metadata)
                
            updated_df.to_csv(metadata_path, index=False, encoding='utf-8')
            
            print(f"✅ Metadata dosyanız güncellendi: {metadata_path}")
        else:
            print("\n🛑 Kayıt durduruldu. Yazılacak yeni veri bulunmuyor.")

def main():
    """Ana veri toplama menüsü."""
    print("=======================================")
    print("  Birleşik Veri Toplama Aracına Hoş Geldiniz ")
    print("=======================================")
    
    # 1. Kayıt Türünü Seç
    print("Ne tür bir kayıt yapmak istersiniz?")
    print("  1. Cümle Kaydı")
    print("  2. Kelime Kaydı")
    
    choice = ""
    while choice not in ["1", "2"]:
        choice = input("Seçiminiz (1 veya 2): ")

    # 2. Dosya Seç
    if choice == '1':
        record_type = "cümle"
        sets_dir = "sentence_sets"
    else:
        record_type = "kelime"
        sets_dir = "words_set"

    available_files = get_files_from_dir(sets_dir)
    if not available_files:
        print(f"'{sets_dir}' dizininde okunacak .txt dosyası bulunamadı.")
        return

    file_to_process = select_from_list(available_files, f"Lütfen bir {record_type} dosyası seçin:")
    lines = get_lines_from_file(file_to_process)
    if not lines:
        return

    # 3. Kullanıcı Kimliğini Al ve Kaydı Başlat
    try:
        user_id = get_user_id()
        user_path = Path(BASE_DATA_PATH) / user_id
        
        if record_type == "cümle":
            save_path = user_path / "audio"
            metadata_path = user_path / "metadata.csv"
        else:
            save_path = user_path / "words"
            metadata_path = user_path / "metadata_words.csv"
            
        run_recording_session(user_id, lines, save_path, metadata_path, record_type)

    except ValueError as e:
        print(f"❌ Hata: {e}")
        return
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")
        return

if __name__ == "__main__":
    main()
