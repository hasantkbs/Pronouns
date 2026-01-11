# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path



import config
from src.core.asr import ASRSystem
from src.utils.utils import record_audio
from src.core.nlu import NLU_System
from src.core.actions import run_action
from src.services.model_service import ModelService
from src.constants import EXIT_COMMANDS

def get_user_id():
    """Kullanıcıdan bir kimlik alır."""
    return input("Lütfen kullanıcı kimliğinizi girin (örn: hasan): ").strip()

def main():
    """Konuşma Bozukluğu Ses Tanıma Sistemi - Ana uygulama döngüsü."""
    
    # 0. Kullanıcı Kimliğini Al ve Kişiselleştirilmiş Modeli Kontrol Et
    user_id = get_user_id()
    if not user_id:
        print("❌ Kullanıcı kimliği girilmedi. Sistem kapatılıyor.")
        return
    
    # Model servisi ile model bulma
    model_to_load = ModelService.find_personalized_model(user_id)
    
    if model_to_load:
        print(f"✅ {user_id} için kişiselleştirilmiş model bulundu! ({model_to_load})")
    else:
        print(f"ℹ️  {user_id} için kişiselleştirilmiş model bulunamadı.")
        print(f"   Varsayılan model kullanılacak: {config.MODEL_NAME}")
        model_to_load = None  # None geçildiğinde ASRSystem config'deki modeli kullanır

    # 1. Sistemleri Başlat
    try:
        print("\n🔄 ASR sistemi başlatılıyor...")
        asr_system = ASRSystem(model_name=model_to_load)
        nlu_system = NLU_System()
    except Exception as e:
        print(f"❌ Sistem başlatılırken kritik bir hata oluştu: {e}")
        import traceback
        traceback.print_exc()
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

        # d. Çıkış kontrolü (NLU'dan gelen intent'e göre veya metin kontrolü)
        if intent == 'exit' or recognized_text.lower().strip() in EXIT_COMMANDS:
            print("\n👋 Sistem kapatılıyor...")
            break
    
    # Geçici ses dosyasını sil
    if os.path.exists(config.GECICI_DOSYA_YOLU): # Used config
        os.remove(config.GECICI_DOSYA_YOLU)

if __name__ == "__main__":
    main()
