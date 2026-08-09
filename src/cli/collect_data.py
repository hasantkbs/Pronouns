# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Birleşik Veri Toplama Aracı
"""

import os
import sys
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import pandas as pd
import argparse
import platform

from src.utils.utils import record_audio, calculate_audio_quality, play_audio, check_consistency, normalize_path_for_cross_platform
from src.services.recording_service import RecordingService
from src.services.model_service import ModelService
from src.services.reporting_service import ReportingService
from src.data.repository import UserDataRepository
from src.constants import (
    RECORD_TYPE_WORD, RECORD_TYPE_SENTENCE, RECORD_TYPE_LETTER,
    DEFAULT_REPETITIONS, DATASET_DIRS, METADATA_FILENAMES, USER_DATA_SUBDIRS
)
import config



# --- Yapılandırma ---
# TARGET_SAMPLING_RATE artık config.py'den alınacak (record_audio fonksiyonu içinde)
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

# record_audio function is now imported from src.utils.utils (VAD-enabled)



# normalize_path_for_cross_platform artık RecordingService içinde

def run_recording_session(user_id, items_to_record, save_path, metadata_path, item_type, repetitions=3, re_record=False):
    """Cümle, kelime veya harf kayıt oturumunu yürütür."""
    save_path.mkdir(parents=True, exist_ok=True)
    metadata = []
    quit_session = False
    
    # Initialize reporting service
    reporting_service = ReportingService()
    
    # Path yolu kontrolü ve uyarı
    import platform
    current_platform = platform.system()
    print("\n" + "="*70)
    print("📁 DOSYA YOLU BİLGİSİ")
    print("="*70)
    print(f"   🖥️  Platform: {current_platform}")
    print(f"   📂 Kayıt Dizini: {save_path.absolute()}")
    print(f"   📄 Metadata Dosyası: {metadata_path.absolute()}")
    
    if current_platform == "Darwin":  # macOS
        print(f"   ⚠️  UYARI: MacBook'ta kayıt yapıyorsunuz.")
        print(f"   💡 Linux server'da eğitim için relative path kullanılacak.")
        print(f"   💡 Dosya yolları otomatik olarak normalize edilecek.")
    elif current_platform == "Linux":
        print(f"   ✅ Linux platformunda kayıt yapıyorsunuz.")
    print("="*70)
    
    # İstatistikler için sayaçlar
    stats = {
        "total_recordings": 0,
        "successful_recordings": 0,
        "failed_recordings": 0,
        "rerecorded": 0,
        "avg_quality_score": 0.0,
        "quality_scores": [],
        "items_completed": 0,  # Tamamlanan item sayısı
        "items_total": 0
    }

    # Mevcut kayıtları CSV'den kontrol et
    already_recorded = set()
    already_recorded_details = {}  # Her item için kaç tekrar kaydedilmiş
    fully_recorded = set()  # Yeterli kaydı olan itemler (IDEAL_REPETITIONS kadar)
    
    if metadata_path.exists() and not re_record:
        try:
            existing_df = pd.read_csv(metadata_path)
            if 'transcription' in existing_df.columns:
                # CSV'deki transkripsiyonları kümeye ekle
                all_recorded = set(existing_df['transcription'].dropna().unique())
                
                # Her item için tekrar sayısını hesapla
                for transcription in all_recorded:
                    item_records = existing_df[existing_df['transcription'] == transcription]
                    rep_count = len(item_records)
                    already_recorded_details[transcription] = rep_count
                    
                    # Eğer yeterli kayıt varsa (IDEAL_REPETITIONS kadar), tam kayıtlı olarak işaretle
                    if rep_count >= config.IDEAL_REPETITIONS:
                        fully_recorded.add(transcription)
                        already_recorded.add(transcription)  # Bu item'i atla
                    elif rep_count > 0:
                        # Kısmen kayıtlı - eksik kayıtları tamamla
                        already_recorded.add(transcription)
                
                print(f"\n📊 Mevcut Kayıt Durumu:")
                print(f"   • Toplam kayıtlı {item_type}: {len(all_recorded)}")
                print(f"   • Tam kayıtlı (≥{config.IDEAL_REPETITIONS} kayıt): {len(fully_recorded)}")
                print(f"   • Kısmen kayıtlı: {len(already_recorded) - len(fully_recorded)}")
                if already_recorded_details:
                    avg_reps = sum(already_recorded_details.values()) / len(already_recorded_details)
                    print(f"   • Ortalama tekrar sayısı: {avg_reps:.1f}")
        except (pd.errors.EmptyDataError, KeyError):
            print(f"Bilgi: Mevcut metadata dosyası ({metadata_path.name}) boş veya geçersiz. Yeni bir dosya oluşturulacak.")
            pass # Dosya boşsa veya sütun yoksa devam et

    # Kaydedilecek yeni öğeleri filtrele
    if not re_record:
        # Tam kayıtlı olanları atla, kısmen kayıtlı olanları dahil et (eksik kayıtları tamamlamak için)
        items_to_record_new = [item for item in items_to_record if item not in fully_recorded]
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
    stats["items_total"] = len(items_to_record_new)

    # Detaylı başlangıç istatistikleri
    print("\n" + "="*70)
    print("📊 KAYIT İSTATİSTİKLERİ")
    print("="*70)
    print(f"   📝 Kaynak Dosya: {len(items_to_record)} {item_type}")
    print(f"   ✅ Zaten Kayıtlı: {len(already_recorded)} {item_type}")
    print(f"   🆕 Yeni Kaydedilecek: {len(items_to_record_new)} {item_type}")
    print(f"   🔄 Her {item_type} için ideal tekrar sayısı: {config.IDEAL_REPETITIONS}")
    # Toplam kayıt hesaplama - kısmen kayıtlı olanlar için eksik kayıtları da dahil et
    total_records_needed = 0
    for item in items_to_record_new:
        current_count = already_recorded_details.get(item, 0)
        needed = max(0, config.IDEAL_REPETITIONS - current_count)
        total_records_needed += needed
    print(f"   📦 Toplam kayıt sayısı: ~{total_records_needed} kayıt (eksik kayıtlar dahil)")
    print(f"   📂 Kayıt Dizini: {save_path.absolute()}")
    print("="*70)

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

            print("\n" + "="*70)
            # Detaylı ilerleme durumunu göster
            progress_percent = ((i) / len(items_to_record_new)) * 100
            remaining_items = len(items_to_record_new) - i
            print(f"📝 İlerleme: {i+1}/{len(items_to_record_new)} {item_type} ({progress_percent:.1f}%)")
            print(f"   • Şu anki: '{item}' (Genel No: {genel_no})")
            print(f"   • Kalan: {remaining_items} {item_type}")
            print(f"   • Tamamlanan: {stats['items_completed']} {item_type}")
            print("="*70)
            
            # Kelime kayıtları için özel klasör yapısı: words/kelime/rep1.wav
            if item_type == "kelime":
                # Her kelime için ayrı klasör oluştur
                word_dir = save_path / item
                word_dir.mkdir(parents=True, exist_ok=True)
            
            recorded_files_for_item = []
            durations_for_item = []  # Tutarlılık kontrolü için süreleri sakla
            
            # Bu item için mevcut kayıt sayısını kontrol et
            current_rep_count = already_recorded_details.get(item, 0)
            remaining_reps = max(0, config.IDEAL_REPETITIONS - current_rep_count)
            
            # Eğer yeterli kayıt varsa, bu item'i atla
            if remaining_reps == 0 and not re_record:
                print(f"   ✅ '{item}' için zaten {current_rep_count} kayıt mevcut (yeterli). Atlanıyor...")
                stats["items_completed"] += 1
                continue
            
            # Eksik kayıtları tamamla
            if current_rep_count > 0 and not re_record:
                print(f"   ℹ️  '{item}' için {current_rep_count}/{config.IDEAL_REPETITIONS} kayıt mevcut. {remaining_reps} kayıt daha yapılacak.")
                # Eksik kayıtlar için rep_num'ı ayarla
                start_rep = current_rep_count + 1
                end_rep = config.IDEAL_REPETITIONS
            else:
                # Yeni kayıt - baştan başla
                start_rep = 1
                end_rep = config.IDEAL_REPETITIONS
            
            for rep_num in range(start_rep, end_rep + 1):
                print(f"   -> Tekrar {rep_num}/{repetitions}: '{item}' için kayıt...")
                
                # Önceki kayıtlar varsa, ortalama süreyi göster (tutarlılık için rehber)
                if durations_for_item and config.CONSISTENCY_CHECK_ENABLED:
                    avg_duration = sum(durations_for_item) / len(durations_for_item)
                    print(f"   💡 Önceki kayıtların ortalama süresi: {avg_duration:.2f}s (tutarlılık için rehber)")
                
                user_input = input("   Hazır olduğunuzda ENTER'a basın (çıkmak için 'q' yazıp ENTER'a basın): ")
                if user_input.lower() == 'q':
                    quit_session = True
                    break

                # Kayıt süresini türe göre ayarla (konuşma bozukluğu için optimize)
                if item_type == "cümle":
                    record_duration = 20
                elif item_type == "kelime":
                    record_duration = 4  # Konuşma bozukluğu için 4 saniye yeterli
                else: # Harf için
                    record_duration = 2
                
                # Dosya yolu ve adını belirle
                if item_type == "kelime":
                    # Kelime için: words/kelime/rep1.wav formatı
                    file_name = f"rep{rep_num}.wav"
                    file_path = word_dir / file_name
                elif item_type == "cümle":
                    # Cümle için: user_id_cümle_file_number_rep1.wav
                    file_name = f"{user_id}_{item_type}_{file_number}_rep{rep_num}.wav"
                    file_path = save_path / file_name
                else:  # Harf için
                    # Harf için: user_id_harf_file_number_rep1.wav
                    file_name = f"{user_id}_{item_type}_{file_number}_rep{rep_num}.wav"
                    file_path = save_path / file_name
                
                # VAD-enabled record_audio kullan (dosyayı otomatik kaydeder)
                recorded_file = record_audio(file_path=str(file_path), record_seconds=record_duration)
                
                if recorded_file:
                    # Ses kalitesi kontrolü
                    quality_info = calculate_audio_quality(recorded_file)
                    
                    # Kalite bilgilerini göster
                    print(f"   📊 Kalite Skoru: {quality_info['quality_score']:.1f}/100")
                    print(f"   📊 RMS: {quality_info['rms']:.0f}, SNR: {quality_info['snr_db']:.1f}dB, Süre: {quality_info['duration']:.2f}s")
                    
                    # Kalite kontrolü ve yeniden kayıt önerisi
                    should_rerecord = False
                    if not quality_info['is_valid']:
                        print(f"   ⚠️  Düşük kalite tespit edildi (skor: {quality_info['quality_score']:.1f} < {config.QUALITY_THRESHOLD})")
                        if config.AUTO_RERECORD_ENABLED:
                            should_rerecord = True
                    
                    # Kayıt önizleme seçeneği
                    if quality_info['is_valid'] or not config.AUTO_RERECORD_ENABLED:
                        preview = input("   🎧 Kaydı dinlemek ister misiniz? (e/h): ").strip().lower()
                        if preview == 'e':
                            print("   ▶️  Kayıt oynatılıyor...")
                            play_audio(recorded_file)
                            keep_recording = input("   💾 Bu kaydı tutmak ister misiniz? (e/h): ").strip().lower()
                            if keep_recording != 'e':
                                should_rerecord = True
                                os.remove(recorded_file)  # Kötü kaydı sil
                                print("   🗑️  Kayıt silindi.")
                    
                    # Yeniden kayıt gerekli mi?
                    if should_rerecord:
                        print(f"   🔄 Yeniden kayıt yapılıyor...")
                        retry_count = 0
                        max_retries = 2
                        
                        while retry_count < max_retries:
                            retry_file = record_audio(file_path=str(file_path), record_seconds=record_duration)
                            if retry_file:
                                retry_quality = calculate_audio_quality(retry_file)
                                print(f"   📊 Yeni Kalite Skoru: {retry_quality['quality_score']:.1f}/100")
                                
                                if retry_quality['quality_score'] > quality_info['quality_score']:
                                    quality_info = retry_quality
                                    recorded_file = retry_file
                                    print(f"   ✅ Daha iyi kalite elde edildi!")
                                    break
                                else:
                                    retry_count += 1
                                    if retry_count < max_retries:
                                        print(f"   ⚠️  Kalite iyileşmedi. Tekrar denenecek...")
                                    else:
                                        print(f"   ⚠️  Maksimum deneme sayısına ulaşıldı. Mevcut kayıt kullanılacak.")
                            else:
                                retry_count += 1
                    
                    # Path normalizasyonu (cross-platform uyumluluk için)
                    # Relative path kullan (Mac'te kayıt, Linux'ta eğitim için)
                    relative_file_path = normalize_path_for_cross_platform(str(file_path.absolute()), save_path.parent)
                    
                    # Başarılı kayıt
                    if quality_info['is_valid'] or not config.AUTO_RERECORD_ENABLED:
                        print(f"   ✅ Ses dosyası kaydedildi: {file_path.name}")
                        print(f"   📁 Relative Path: {relative_file_path}")
                        recorded_files_for_item.append(str(file_path.absolute()))
                        durations_for_item.append(quality_info['duration'])
                        metadata.append({
                            "file_path": relative_file_path,  # Relative path kullan
                            "transcription": item,
                            "repetition": rep_num,
                            "quality_score": quality_info['quality_score'],
                            "rms": quality_info['rms'],
                            "snr_db": quality_info['snr_db'],
                            "duration": quality_info['duration']
                        })
                        stats["successful_recordings"] += 1
                        stats["quality_scores"].append(quality_info['quality_score'])
                    else:
                        print(f"   ⚠️  Kayıt düşük kalitede ama kaydedildi: {file_path.name}")
                        print(f"   📁 Relative Path: {relative_file_path}")
                        recorded_files_for_item.append(str(file_path.absolute()))
                        durations_for_item.append(quality_info['duration'])
                        metadata.append({
                            "file_path": relative_file_path,  # Relative path kullan
                            "transcription": item,
                            "repetition": rep_num,
                            "quality_score": quality_info['quality_score'],
                            "rms": quality_info['rms'],
                            "snr_db": quality_info['snr_db'],
                            "duration": quality_info['duration']
                        })
                        stats["successful_recordings"] += 1
                        stats["quality_scores"].append(quality_info['quality_score'])
                    
                    # Her tekrar sonrası ilerleme göster
                    current_item_total = current_rep_count + len(recorded_files_for_item)
                    item_progress = f"{current_item_total}/{config.IDEAL_REPETITIONS}"
                    # Toplam kayıt hesaplama (her item için IDEAL_REPETITIONS kadar)
                    total_expected = len(items_to_record_new) * config.IDEAL_REPETITIONS
                    overall_progress = stats["successful_recordings"] / total_expected * 100 if total_expected > 0 else 0
                    print(f"   📊 İlerleme: '{item}' {item_progress} | Genel: {stats['successful_recordings']}/{total_expected} kayıt ({overall_progress:.1f}%)")
                    
                    # Tutarlılık kontrolü (2 veya daha fazla kayıt varsa)
                    if len(durations_for_item) >= 2 and config.CONSISTENCY_CHECK_ENABLED:
                        consistency_info = check_consistency(durations_for_item, config.CONSISTENCY_TOLERANCE)
                        
                        if not consistency_info['is_consistent']:
                            print(f"   ⚠️  Tutarlılık Uyarısı: Süre farkı {consistency_info['max_diff']:.2f}s (tolerans: {consistency_info['tolerance']:.2f}s)")
                            print(f"   💡 Ortalama süre: {consistency_info['avg_duration']:.2f}s, Standart sapma: {consistency_info['std_deviation']:.2f}s")
                            print(f"   💡 Sonraki kayıtlarda {consistency_info['avg_duration']:.2f}s civarında söylemeye çalışın.")
                        else:
                            print(f"   ✅ Tutarlılık: Tüm kayıtlar benzer sürede ({consistency_info['avg_duration']:.2f}s ± {consistency_info['std_deviation']:.2f}s)")
                    
                    if should_rerecord:
                        stats["rerecorded"] += 1
                    stats["total_recordings"] += 1
                else:
                    print(f"   ❌ Ses kaydı başarısız oldu veya ses algılanmadı.")
                    stats["failed_recordings"] += 1
                    stats["total_recordings"] += 1
            


            # Item tamamlandı kontrolü
            total_reps_for_item = current_rep_count + len(recorded_files_for_item)
            if total_reps_for_item >= config.IDEAL_REPETITIONS:
                stats["items_completed"] += 1
                print(f"\n   ✅ '{item}' tamamlandı! ({total_reps_for_item}/{config.IDEAL_REPETITIONS} kayıt) | ({stats['items_completed']}/{stats['items_total']} {item_type})")
            
            if quit_session:
                print("\n" + "="*70)
                print("⏸️  Kullanıcı isteğiyle oturum sonlandırılıyor...")
                print("="*70)
                break
        
        if not quit_session:
            print("\n" + "="*70)
            print(f"🎉 {item_type.capitalize()} toplama işlemi başarıyla tamamlandı!")
            print("="*70)
            
            # Detaylı istatistikleri göster
            if stats["total_recordings"] > 0:
                if stats["quality_scores"]:
                    stats["avg_quality_score"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])
                
                print(f"\n📊 DETAYLI OTURUM İSTATİSTİKLERİ")
                print("="*70)
                print(f"   📝 Kaynak Dosya: {len(items_to_record)} {item_type}")
                print(f"   ✅ Tamamlanan: {stats['items_completed']}/{stats['items_total']} {item_type}")
                print(f"   📦 Toplam Kayıt: {stats['total_recordings']}")
                print(f"   ✅ Başarılı: {stats['successful_recordings']}")
                print(f"   ❌ Başarısız: {stats['failed_recordings']}")
                print(f"   🔄 Yeniden Kayıt: {stats['rerecorded']}")
                if stats["avg_quality_score"] > 0:
                    print(f"   ⭐ Ortalama Kalite Skoru: {stats['avg_quality_score']:.1f}/100")
                print(f"   📂 Kayıt Dizini: {save_path.absolute()}")
                print(f"   📄 Metadata Dosyası: {metadata_path.absolute()}")
                print("="*70)

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
            
            # Create recording report
            stats["recorded_items"] = stats["items_completed"]
            stats["skipped_items"] = stats["items_total"] - stats["items_completed"]
            stats["total_items"] = len(items_to_record)
            
            report_file = reporting_service.log_recording_session(
                user_id=user_id,
                record_type=item_type,
                stats=stats
            )
            print(f"\n📊 Recording report saved: {report_file}")
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
        repetitions = config.IDEAL_REPETITIONS  # Konuşma bozukluğu için ideal tekrar sayısı
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