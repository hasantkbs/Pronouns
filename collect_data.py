# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Birleşik Veri Toplama Aracı

Bu interaktif script, hem cümle hem de kelime bazlı veri toplamayı yönetir.
Kullanıcıya ne tür bir kayıt yapmak istediğini sorar, ilgili dosya setlerinden
birini seçtirir ve kayıt işlemini başlatır.

Kullanım:
- python collect_data.py
- python collect_data.py --re-record
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

def run_recording_session(user_id, items_to_record, save_path, metadata_path, item_type, repetitions=3, re_record=False):
    """Cümle, kelime veya harf kayıt oturumunu yürütür."""
    save_path.mkdir(parents=True, exist_ok=True)
    metadata = []
    quit_session = False

    # Mevcut kayıtları CSV'den kontrol et
    already_recorded = set()
    if metadata_path.exists() and not re_record:
        try:
            existing_df = pd.read_csv(metadata_path)
            if 'transcription' in existing_df.columns:
                # CSV'deki transkripsiyonları kümeye ekle
                already_recorded = set(existing_df['transcription'].dropna().unique())
                print(f"\nBilgi: Mevcut {len(already_recorded)} kayıtlı {item_type} bulundu.")
        except (pd.errors.EmptyDataError, KeyError):
            print(f"Bilgi: Mevcut metadata dosyası ({metadata_path.name}) boş veya geçersiz. Yeni bir dosya oluşturulacak.")
            pass # Dosya boşsa veya sütun yoksa devam et

    # Kaydedilecek yeni öğeleri filtrele (daha önce kaydedilmemiş olanlar)
    if not re_record:
        items_to_record_new = [item for item in items_to_record if item not in already_recorded]
    else:
        items_to_record_new = items_to_record

    if not items_to_record_new:
        print(f"\n🎉 Tebrikler! Bu setteki tüm {item_type}ler zaten kaydedilmiş.")
        return

    # Orijinal listedeki indeksleri aramak için bir sözlük oluştur
    # Bu, dosya adlarının tutarlı olmasını sağlar
    original_indices = {item: i for i, item in enumerate(items_to_record)}
    
    # "Genel No" için başlangıç sayısını belirle
    num_already_recorded = len(already_recorded)

    print(f"\n-> Toplam {len(items_to_record)} {item_type} içinden {len(items_to_record_new)} adet yeni {item_type} kaydedilecek.")

    try:
        # Sadece yeni (kaydedilmemiş) öğeler üzerinde döngü yap
        for i, item in enumerate(items_to_record_new):
            # Dosya adlandırması için orijinal indeksi bul (tutarlılık için)
            original_index = original_indices.get(item)
            
            if original_index is None and not re_record:
                print(f"⚠️ Uyarı: '{item}' kelimesi orijinal listede bulunamadı. Atlanıyor.")
                continue

            # Dosya adı için numara (orijinal sıraya göre)
            file_number = original_index + 1 if original_index is not None else i + 1
            
            # Ekranda gösterilecek Genel No (toplam kayıt sayısı)
            genel_no = num_already_recorded + i + 1

            print("\n" + "="*50)
            # İlerleme durumunu göster: yeni listedeki sıra / toplam yeni sayısı
            print(f"Kayıt {i+1}/{len(items_to_record_new)} (Genel No: {genel_no}): -> '{item}'")
            
            for rep_num in range(1, repetitions + 1):
                print(f"   -> Tekrar {rep_num}/{repetitions}: '{item}' için kayıt...")
                
                user_input = input("   Hazır olduğunuzda ENTER'a basın (çıkmak için 'q' yazıp ENTER'a basın): ")
                if user_input.lower() == 'q':
                    quit_session = True
                    break

                # Kayıt süresini türe göre ayarla
                if item_type == "cümle":
                    duration = 20
                elif item_type == "kelime":
                    duration = 4
                else: # Harf için
                    duration = 2
                
                rec = record_audio(duration=duration, samplerate=TARGET_SAMPLING_RATE)
                
                # Tutarlı dosya adı oluştur (orijinal indeksi kullanarak)
                file_name = f"{user_id}_{item_type}_{file_number}_rep{rep_num}.wav"
                file_path = save_path / file_name
                
                sf.write(file_path, rec, TARGET_SAMPLING_RATE)
                print(f"   ✅ Ses dosyası kaydedildi: {file_path}")
                
                metadata.append({
                    "file_path": str(file_path.absolute()),
                    "transcription": item,
                    "repetition": rep_num
                })
            
            if quit_session:
                print("\nKullanıcı isteğiyle oturum sonlandırılıyor...")
                break
        
        if not quit_session:
            print("\n" + "="*50)
            print(f"🎉 {item_type.capitalize()} toplama işlemi başarıyla tamamlandı!")

    finally:
        if metadata:
            print("\n🛑 Kayıt durduruluyor. Toplanan veriler CSV dosyasına yazılıyor...")
            # Mevcut metadata dosyasını oku ve yeni verileri ekle
            if metadata_path.exists() and metadata_path.stat().st_size > 0 and not re_record:
                try:
                    existing_df = pd.read_csv(metadata_path)
                    new_df = pd.DataFrame(metadata)
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                except pd.errors.EmptyDataError:
                    # Bu blok, dosya varsa ama tamamen boşsa çalışır
                    updated_df = pd.DataFrame(metadata)
            else:
                # Dosya hiç yoksa veya boşsa, yeni bir DataFrame oluştur
                updated_df = pd.DataFrame(metadata)
                
            # Yinelenen satırları temizle (güvenlik önlemi)
            if not re_record:
                updated_df.drop_duplicates(subset=['file_path', 'transcription', 'repetition'], inplace=True)
            
            updated_df.to_csv(metadata_path, index=False, encoding='utf-8')
            
            print(f"✅ Metadata dosyanız güncellendi: {metadata_path}")
        else:
            print("\n🛑 Kayıt durduruldu. Yazılacak yeni veri bulunmuyor.")


def main():
    """Ana veri toplama menüsü."""
    parser = argparse.ArgumentParser(description="Birleşik Veri Toplama Aracı")
    parser.add_argument("--re-record", action="store_true", help="datasets/tekrar_kayit.txt dosyasındaki verileri yeniden kaydeder.")
    args = parser.parse_args()

    if args.re_record:
        print("=======================================")
        print("     Yeniden Kayıt Modu Başlatıldı     ")
        print("=======================================")
        try:
            user_id = get_user_id()
            rerecord_file_path = Path("datasets/tekrar_kayit.txt")
            lines = get_lines_from_file(rerecord_file_path)
            if not lines:
                print("Yeniden kaydedilecek veri bulunamadı.")
                return

            # Determine record type (word or letter)
            # This is a simple heuristic, assuming single characters are letters
            if all(len(line) == 1 for line in lines):
                record_type = "harf"
                repetitions = 5
                save_path = Path(BASE_DATA_PATH) / user_id / "letters"
                metadata_path = Path(BASE_DATA_PATH) / user_id / "metadata_letters.csv"
            else:
                record_type = "kelime"
                repetitions = 3
                save_path = Path(BASE_DATA_PATH) / user_id / "words"
                metadata_path = Path(BASE_DATA_PATH) / user_id / "metadata_words.csv"

            run_recording_session(user_id, lines, save_path, metadata_path, record_type, repetitions, re_record=True)

        except ValueError as e:
            print(f"❌ Hata: {e}")
            return
        except Exception as e:
            print(f"Beklenmedik bir hata oluştu: {e}")
            return
        return

    print("=======================================")
    print("  Birleşik Veri Toplama Aracına Hoş Geldiniz ")
    print("=======================================")
    
    # 1. Kayıt Türünü Seç
    print("Ne tür bir kayıt yapmak istersiniz?")
    print("  1. Cümle Kaydı")
    print("  2. Kelime Kaydı")
    print("  3. Harf Kaydı")
    
    choice = ""
    while choice not in ["1", "2", "3"]:
        choice = input("Seçiminiz (1, 2 veya 3): ")

    # 2. Dosya Seç
    if choice == '1':
        record_type = "cümle"
        sets_dir = "datasets/sentence_sets"
        repetitions = 3
    elif choice == '2':
        record_type = "kelime"
        sets_dir = "datasets/words_set"
        repetitions = 3
    else:
        record_type = "harf"
        sets_dir = "datasets/letters_set"
        repetitions = 5

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
        elif record_type == "kelime":
            save_path = user_path / "words"
            metadata_path = user_path / "metadata_words.csv"
        else: # Harf için
            save_path = user_path / "letters"
            metadata_path = user_path / "metadata_letters.csv"
            
        run_recording_session(user_id, lines, save_path, metadata_path, record_type, repetitions)

    except ValueError as e:
        print(f"❌ Hata: {e}")
        return
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")
        return

if __name__ == "__main__":
    main()