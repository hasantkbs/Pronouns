# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(__file__)) # Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__))) # Add parent directory for src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # Add project root for config import

import config
from src.core.asr import ASRSystem
from src.utils.utils import record_audio
from src.core.nlu import NLU_System # Added
from src.core.actions import run_action # Added

def main():
    """Konuşma Bozukluğu Ses Tanıma Sistemi - Ana uygulama döngüsü."""
    # 1. Sistemleri Başlat
    try:
        asr_system = ASRSystem()
        nlu_system = NLU_System() # Added
    except Exception as e:
        print(f"Sistem başlatılırken kritik bir hata oluştu: {e}")
        return

    print("\n=========================================")
    print("   Konuşma Bozukluğu Ses Tanıma Sistemi   ")
    print("=========================================")
    print("Bu sistem konuşma bozukluğu olan bireylerin")
    print("seslerini tanıyıp metne dönüştürür.")
    print("Çıkmak için 'çık' veya 'exit' deyin.\n")

    # 2. Ana Ses Tanıma ve Anlama Döngüsü
    while True:
        # a. Kullanıcıdan ses al
        prompt = "\n------------------------------------------\n🎤 Konuşmak için ENTER'a basın ve konuşun..."
        audio_file = record_audio(file_path=config.GECICI_DOSYA_YOLU, record_seconds=config.KAYIT_SURESI_SN, prompt=prompt) # Used config

        if not audio_file:
            print("❌ Ses kaydı alınamadı. Lütfen tekrar deneyin.")
            continue

        # b. Sesi metne çevir (ASR)
        print("\n🧠 Sesiniz analiz ediliyor...")
        recognized_text = asr_system.transcribe(audio_file)

        if not recognized_text:
            print("❌ Sessizlik algılandı veya bir hata oluştu. Lütfen tekrar deneyin.")
            continue

        print(f"\n📝 Tanınan Metin:\n   '{recognized_text}'")

        # c. Metni işle (NLU) ve eylemi çalıştır
        intent, entities = nlu_system.process_text(recognized_text)
        
        # Eylemi çalıştır ve sonucu yazdır
        action_response = run_action(intent, entities)
        print(f"🤖 {action_response}")

        # d. Çıkış kontrolü (NLU'dan gelen intent'e göre)
        if intent == 'exit':
            print("\n👋 Sistem kapatılıyor...")
            break
    
    # Geçici ses dosyasını sil
    if os.path.exists(config.GECICI_DOSYA_YOLU): # Used config
        os.remove(config.GECICI_DOSYA_YOLU)

if __name__ == "__main__":
    main()
