# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Veri Seti Analizi
downloaded_data klasöründeki verileri analiz eder.
"""

import os
import glob
from pathlib import Path

def analyze_data_structure():
    """Veri seti yapısını analiz eder."""
    print("📊 Veri Seti Analizi")
    print("=" * 50)
    
    base_path = "downloaded_data"
    
    if not os.path.exists(base_path):
        print(f"❌ {base_path} klasörü bulunamadı!")
        return
    
    # Klasörleri listele
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    print("📁 Mevcut Klasörler:")
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        # Dosya sayılarını hesapla
        mp3_files = glob.glob(os.path.join(folder_path, "*.mp3"))
        parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))
        tar_files = glob.glob(os.path.join(folder_path, "*.tar"))
        
        print(f"\n📂 {folder}/")
        print(f"   🎵 MP3 dosyaları: {len(mp3_files)}")
        print(f"   📄 Parquet dosyaları: {len(parquet_files)}")
        print(f"   📦 TAR dosyaları: {len(tar_files)}")
        
        # Klasör boyutunu hesapla
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        size_mb = total_size / (1024 * 1024)
        print(f"   💾 Toplam boyut: {size_mb:.1f} MB")

def analyze_audio_files():
    """Ses dosyalarını analiz eder."""
    print("\n🎵 Ses Dosyası Analizi")
    print("=" * 50)
    
    # MP3 dosyalarını bul
    mp3_files = glob.glob("downloaded_data/**/*.mp3", recursive=True)
    
    if not mp3_files:
        print("❌ MP3 dosyası bulunamadı!")
        return
    
    print(f"📊 Toplam MP3 dosyası: {len(mp3_files)}")
    
    # İlk birkaç dosyayı incele
    print("\n📋 Örnek Dosya İsimleri:")
    for i, file_path in enumerate(mp3_files[:10]):
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"   {i+1}. {file_name} ({file_size:.1f} KB)")
    
    if len(mp3_files) > 10:
        print(f"   ... ve {len(mp3_files) - 10} dosya daha")

def analyze_parquet_files():
    """Parquet dosyalarını analiz eder."""
    print("\n📄 Parquet Dosyası Analizi")
    print("=" * 50)
    
    parquet_files = glob.glob("downloaded_data/**/*.parquet", recursive=True)
    
    if not parquet_files:
        print("❌ Parquet dosyası bulunamadı!")
        return
    
    print(f"📊 Toplam Parquet dosyası: {len(parquet_files)}")
    
    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"   📄 {file_name} ({file_size:.1f} MB)")
        
        # Pandas varsa detaylı analiz
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            print(f"      📈 Kayıt sayısı: {len(df)}")
            print(f"      📋 Sütunlar: {df.columns.tolist()}")
            
            # İlk birkaç kaydı göster
            if len(df) > 0:
                print(f"      🔍 İlk kayıt örneği:")
                first_row = df.iloc[0]
                for col in df.columns:
                    if col in ['sentence', 'path']:
                        value = str(first_row[col])[:50] + "..." if len(str(first_row[col])) > 50 else str(first_row[col])
                        print(f"         {col}: {value}")
                break  # Sadece ilk dosyayı detaylı analiz et
                
        except ImportError:
            print("      ⚠️  Pandas yüklü değil, detaylı analiz yapılamadı.")
        except Exception as e:
            print(f"      ❌ Hata: {e}")

def get_training_recommendations():
    """Eğitim önerileri verir."""
    print("\n💡 Eğitim Önerileri")
    print("=" * 50)
    
    mp3_count = len(glob.glob("downloaded_data/**/*.mp3", recursive=True))
    parquet_count = len(glob.glob("downloaded_data/**/*.parquet", recursive=True))
    
    print(f"📊 Veri Seti Durumu:")
    print(f"   🎵 Ses dosyaları: {mp3_count}")
    print(f"   📄 Metadata dosyaları: {parquet_count}")
    
    if mp3_count > 1000 and parquet_count > 0:
        print("\n✅ Eğitim için yeterli veri mevcut!")
        print("🚀 Önerilen adımlar:")
        print("   1. train_model.py scriptini çalıştırın")
        print("   2. Eğitim tamamlandıktan sonra config.py'de MODEL_NAME'i güncelleyin")
        print("   3. Yeni modeli test edin")
    else:
        print("\n⚠️  Eğitim için daha fazla veri gerekebilir.")
        print("📝 Öneriler:")
        print("   - Daha fazla ses verisi toplayın")
        print("   - Veri kalitesini kontrol edin")
        print("   - Farklı konuşma bozukluğu türleri için veri ekleyin")

def main():
    """Ana analiz fonksiyonu."""
    print("🔍 Konuşma Bozukluğu Veri Seti Analizi")
    print("=" * 60)
    
    # Veri yapısını analiz et
    analyze_data_structure()
    
    # Ses dosyalarını analiz et
    analyze_audio_files()
    
    # Parquet dosyalarını analiz et
    analyze_parquet_files()
    
    # Önerileri ver
    get_training_recommendations()
    
    print("\n🎯 Sonuç:")
    print("Bu veri seti konuşma bozukluğu olan bireyler için")
    print("özel bir model eğitmek için kullanılabilir.")

if __name__ == "__main__":
    main()
