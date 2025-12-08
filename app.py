# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path



import config
from src.core.asr import ASRSystem
from src.utils.utils import record_audio
from src.core.nlu import NLU_System # Added
from src.core.actions import run_action # Added

def get_user_id():
    """Kullanıcıdan bir kimlik alır."""
    return input("Lütfen kullanıcı kimliğinizi girin (örn: hasan): ").strip()

def main():
    """Konuşma Bozukluğu Ses Tanıma Sistemi - Ana uygulama döngüsü."""
    
    # 0. Kullanıcı Kimliğini Al ve Kişiselleştirilmiş Modeli Kontrol Et
    user_id = get_user_id()
    personalized_model_path = Path("data/models/personalized_models") / user_id
    model_to_load = None

    if personalized_model_path.exists():
        print(f"✅ {user_id} için kişiselleştirilmiş model bulundu!")
        model_to_load = str(personalized_model_path)
    else:
        print("ℹ️  Kişiselleştirilmiş model bulunamadı. Varsayılan model kullanılacak.")
        model_to_load = config.MODEL_NAME

    # 1. Sistemleri Başlat
    try:
        asr_system = ASRSystem(model_name=model_to_load)
        nlu_system = NLU_System() # Added
    except Exception as e:
        print(f"Sistem başlatılırken kritik bir hata oluştu: {e}")
        return

    print("\n=========================================")
    print("   Konuşma Bozukluğu Ses Tanıma Sistemi   ")
    print("=========================================")
    print(f"Hoş geldin, {user_id}!")
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
