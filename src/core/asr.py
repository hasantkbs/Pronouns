

# -*- coding: utf-8 -*-
import torch
import whisper
import librosa

class ASRSystem:
    """Otomatik Konuşma Tanıma (ASR) sistemi."""

    def __init__(self, model_name="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=self.device)
        print(f"ASR Sistemi başlatıldı. Model: Whisper-{model_name}, Cihaz: {self.device}")

    def transcribe(self, audio_path):
        """Verilen ses dosyasını metne dönüştürür."""
        try:
            # Whisper modeli doğrudan dosya yolunu kabul eder
            result = self.model.transcribe(str(audio_path))
            return result["text"]
        except Exception as e:
            print(f"ASR transkripsiyon hatası: {e}")
            return None

if __name__ == '__main__':
    asr_system = ASRSystem()
    # Örnek kullanım için geçerli bir .wav dosyası yolu belirtin
    test_file = "users/speech_commands_test/on.wav" 
    if os.path.exists(test_file):
        print(f"\n--- ASR Testi Başlatılıyor ---")
        print(f"Test dosyası: {test_file}")
        recognized_text = asr_system.transcribe(test_file)
        print(f"\nTest tamamlandı.")
        print(f"Tanınan metin: '{recognized_text}'")
    else:
        print(f"Test dosyası bulunamadı: {test_file}")

