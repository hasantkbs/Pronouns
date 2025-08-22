# -*- coding: utf-8 -*-
import os
from src.core.asr import ASRSystem
from src.utils.utils import record_audio

# --- Ayarlar ---
TEMP_WAV_FILE = "temp_recording.wav"

def main():
    """Konuşma Bozukluğu Ses Tanıma Sistemi - Ana uygulama döngüsü."""
    # 1. ASR Sistemini Başlat
    try:
        asr_system = ASRSystem()
    except Exception as e:
        print(f"ASR Sistemi başlatılırken kritik bir hata oluştu: {e}")
        return

    print("\n=========================================")
    print("   Konuşma Bozukluğu Ses Tanıma Sistemi   ")
    print("=========================================")
    print("Bu sistem konuşma bozukluğu olan bireylerin")
    print("seslerini tanıyıp metne dönüştürür.")
    print("Çıkmak için 'çık' veya 'exit' deyin.\n")

    # 2. Ana Ses Tanıma Döngüsü
    while True:
        # a. Kullanıcıdan ses al
        prompt = "\n------------------------------------------\n🎤 Konuşmak için ENTER'a basın ve konuşun..."
        audio_file = record_audio(file_path=TEMP_WAV_FILE, record_seconds=5, prompt=prompt)

        # b. Sesi metne çevir (ASR)
        print("\n🧠 Sesiniz analiz ediliyor...")
        recognized_text = asr_system.transcribe(audio_file)

        if not recognized_text:
            print("❌ Sessizlik algılandı veya bir hata oluştu. Lütfen tekrar deneyin.")
            continue

        # c. Sonucu ekrana yazdır
        print(f"\n📝 Tanınan Metin:")
        print(f"   '{recognized_text}'")
        
        # d. Çıkış kontrolü
        if recognized_text.lower() in ['çık', 'exit', 'kapat', 'durdur']:
            print("\n👋 Sistem kapatılıyor...")
            break
    
    # Geçici ses dosyasını sil
    if os.path.exists(TEMP_WAV_FILE):
        os.remove(TEMP_WAV_FILE)

if __name__ == "__main__":
    main()