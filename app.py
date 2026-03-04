# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import config
from src.core.asr import ASRSystem
from src.core.synthesizer import WordSynthesizer
from src.utils.utils import record_audio
from src.core.nlu import NLU_System
from src.core.actions import run_action
from src.services.model_service import ModelService
from src.constants import EXIT_COMMANDS


def get_user_id():
    return input("Lutfen kullanici kimliginizi girin (orn: hasan): ").strip()


def main():
    """Konusma Bozuklugu Ses Tanima Sistemi - Ana uygulama dongusu."""

    user_id = get_user_id()
    if not user_id:
        print("Kullanici kimligi girilmedi. Sistem kapatiliyor.")
        return

    model_to_load = ModelService.find_personalized_model(user_id)
    if model_to_load:
        print(f"{user_id} icin kisisellestirilmis model bulundu: {model_to_load}")
    else:
        print(f"{user_id} icin kisisellestirilmis model bulunamadi.")
        print(f"Varsayilan model kullanilacak: {config.MODEL_NAME}")
        model_to_load = None

    try:
        print("\nASR sistemi baslatiliyor...")
        asr_system = ASRSystem(model_name=model_to_load)
        nlu_system = NLU_System()

        print("Sentez sistemi baslatiliyor...")
        try:
            synthesizer = WordSynthesizer(user_id=user_id)
        except Exception as e:
            print(f"Sentez sistemi baslatılamadı (sadece metin modu aktif): {e}")
            synthesizer = None

    except Exception as e:
        print(f"Sistem baslatilirken kritik hata: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n=========================================")
    print("  Konusma Bozuklugu Ses Tanima Sistemi   ")
    print("=========================================")
    print(f"Hos geldin, {user_id}!")
    print("Cikis icin 'cik' veya 'exit' deyin.\n")

    while True:
        prompt = "\n------------------------------------------\nKonusmak icin ENTER'a basin ve konusun..."
        audio_file = record_audio(
            file_path=config.GECICI_DOSYA_YOLU,
            record_seconds=config.KAYIT_SURESI_SN,
            prompt=prompt,
        )

        if not audio_file:
            print("Ses kaydi alinamadi. Lutfen tekrar deneyin.")
            continue

        print("\nSesiniz analiz ediliyor...")
        recognized_text, confidence = asr_system.transcribe(audio_file)

        if not recognized_text:
            print("Sessizlik algilandi veya bir hata olustu. Lutfen tekrar deneyin.")
            continue

        conf_pct = f"{confidence:.0%}"
        print(f"\nTanınan Metin : '{recognized_text}'  (guven: {conf_pct})")

        # Dusuk guvenli tanımada kullaniciya uyar
        if confidence < 0.30:
            print("Not: Guven skoru dusuk — lutfen daha net konusun.")

        intent, entities = nlu_system.process_text(recognized_text)
        action_response = run_action(intent, entities)
        print(f"Yanit: {action_response}")

        # Sentez: aksiyon yanıtını kullanicinin kendi sesiyle oynat
        if synthesizer is not None and action_response:
            response_words = action_response.split()
            out_path, missing = synthesizer.synthesize(response_words, strategy="best")
            if out_path:
                if missing:
                    print(f"Sentezde eksik kelimeler (metin olarak gosterildi): {missing}")
                synthesizer.play(out_path)
                try:
                    os.remove(out_path)
                except OSError:
                    pass

        if intent == "exit" or recognized_text.lower().strip() in EXIT_COMMANDS:
            print("\nSistem kapatiliyor...")
            break

    if os.path.exists(config.GECICI_DOSYA_YOLU):
        os.remove(config.GECICI_DOSYA_YOLU)


if __name__ == "__main__":
    main()
