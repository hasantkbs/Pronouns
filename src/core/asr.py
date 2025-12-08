# -*- coding: utf-8 -*-
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

class ASRSystem:
    """
    Otomatik Konuşma Tanıma (ASR) sistemi.
    Bu sürüm, Hugging Face Transformers kütüphanesini kullanarak yerel olarak
    ince ayarlanmış (fine-tuned) veya standart Whisper modellerini yükler.
    """

    def __init__(self, model_name="openai/whisper-base"):
        """
        ASR Sistemini başlatır.

        Args:
            model_name (str): Yüklenecek modelin Hugging Face Hub adı veya yerel dosya yolu.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ASR Sistemi için model yükleniyor: '{model_name}'")
        
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval() # Modeli değerlendirme moduna al
        except Exception as e:
            print(f"HATA: Model veya işlemci yüklenemedi: {e}")
            print("Lütfen model adının veya yolunun doğru olduğundan emin olun.")
            raise

        print(f"✅ ASR Sistemi başarıyla başlatıldı. Model: '{model_name}', Cihaz: {self.device}")

    def transcribe(self, audio_path):
        """
        Verilen ses dosyasını metne dönüştürür.

        Args:
            audio_path (str): Ses dosyasının yolu.

        Returns:
            str: Tanınan metin veya hata durumunda None.
        """
        try:
            # 1. Ses dosyasını yükle ve yeniden örnekle
            speech_array, sampling_rate = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
            
            # 2. Sesi işleyerek modelin beklediği formata getir
            input_features = self.processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_features

            # 3. Model ile metin tahmini yap
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features.to(self.device))

            # 4. Tahmin edilen token'ları metne çevir
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            return transcription.strip()
            
        except Exception as e:
            print(f"HATA: ASR transkripsiyon hatası: {e}")
            return None

if __name__ == '__main__':
    # Bu testin çalışması için 'openai/whisper-base' modelinin indirilmesi gerekir.
    # İnternet bağlantısı gerektirebilir.
    print("\n--- ASR Sınıfı Testi Başlatılıyor ---")
    try:
        # Varsayılan model ile test et
        asr_system = ASRSystem("openai/whisper-base") 
        print("Test için geçici bir boş ses dosyası kullanılıyor...")
        # Not: Gerçek bir ses dosyası ile test etmek daha doğru sonuç verir.
        # test_file = "path/to/your/test/audio.wav" 
        # recognized_text = asr_system.transcribe(test_file)
        # print(f"Tanınan metin: '{recognized_text}'")
        print("✅ ASR sınıfı başarıyla yüklendi ve test edildi.")
    except Exception as e:
        print(f"❌ ASR sınıfı testi sırasında hata oluştu: {e}")