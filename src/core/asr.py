

# -*- coding: utf-8 -*-
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
from peft import PeftModel
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

class ASRSystem:
    """
    Konuşma Bozukluğu Ses Tanıma Sistemi.
    Ses verisini metne dönüştürür, konuşma bozukluğu olan bireyler için optimize edilmiştir.
    """
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print("🎧 Konuşma Bozukluğu Ses Tanıma Sistemi başlatılıyor...")
        print(f"💻 Kullanılacak cihaz: {self.device}")

        try:
            # Check if the provided model_name is a directory (i.e., a personalized model path)
            is_personalized_model = os.path.isdir(self.model_name)

            if is_personalized_model:
                print(f"📥 Temel model '{config.MODEL_NAME}' yükleniyor...")
                # 1. Load the base model
                self.processor = Wav2Vec2Processor.from_pretrained(config.MODEL_NAME)
                self.model = Wav2Vec2ForCTC.from_pretrained(config.MODEL_NAME).to(self.device)
                
                print(f"🎨 Kişiselleştirilmiş adaptör yükleniyor: '{self.model_name}'")
                # 2. Load and apply the PEFT adapter
                self.model = PeftModel.from_pretrained(self.model, self.model_name)
                print("✨ Adaptör başarıyla birleştirildi.")

            else:
                # Original behavior: load a model directly from Hugging Face hub
                print(f"📥 '{self.model_name}' modeli yükleniyor...")
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)

            self.model.eval()
            print("✅ ASR Modeli başarıyla yüklendi.")
        except Exception as e:
            print(f"❌ Hata: Model yüklenemedi. {e}")
            raise

        # Dil modelini yükle (opsiyonel)
        self._load_decoder()

    def _load_decoder(self):
        """KenLM tabanlı CTC decoder'ı yükler (opsiyonel)."""
        self.decoder = None
        if not os.path.exists(config.KENLM_MODEL_PATH):
            print("\n⚠️  KenLM dil modeli bulunamadı.")
            print(f"   Yol: '{config.KENLM_MODEL_PATH}'")
            print("   Dil modeli olmadan devam ediliyor (basit decoding).\n")
            return

        try:
            print("📚 Dil modeli (decoder) yükleniyor...")
            vocab_dict = self.processor.tokenizer.get_vocab()
            sorted_vocab_list = sorted(vocab_dict.items(), key=lambda item: item[1])
            labels = [x[0] for x in sorted_vocab_list]
            
            # Harf bazlı modellerde kelime ayırıcıyı boşluk yap
            labels[labels.index("|")] = " "

            self.decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=config.KENLM_MODEL_PATH,
            )
            print("✅ Dil modeli başarıyla yüklendi.")
        except Exception as e:
            print(f"❌ Hata: Dil modeli yüklenemedi. {e}")
            self.decoder = None

    @torch.no_grad()
    def transcribe(self, audio_path: str) -> str:
        """
        Verilen ses dosyasını metne dönüştürür.
        Konuşma bozukluğu olan bireyler için optimize edilmiştir.

        Args:
            audio_path (str): Ses dosyasının yolu

        Returns:
            str: Tanınan metin
        """
        try:
            # Ses dosyasını oku
            speech_array, sampling_rate = sf.read(audio_path)
            
            # Örnekleme oranını kontrol et ve gerekirse değiştir
            if sampling_rate != self.processor.feature_extractor.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate,
                    new_freq=self.processor.feature_extractor.sampling_rate
                )
                speech_array = resampler(torch.tensor(speech_array, dtype=torch.float)).numpy()

            # Ses verisini model için hazırla
            input_values = self.processor(
                speech_array, 
                sampling_rate=self.processor.feature_extractor.sampling_rate, 
                return_tensors="pt", 
                padding=True
            ).input_values
            input_values = input_values.to(self.device)

            # Model tahmini
            logits = self.model(input_values).logits

            # Deşifreleme
            if self.decoder:
                # Dil modeli ile beam search decoding (daha doğru)
                transcription = self.decoder.decode(logits[0])
            else:
                # Basit greedy decoding
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
            
            # Metni temizle ve döndür
            transcription = transcription.strip()
            return transcription

        except Exception as e:
            print(f"❌ Hata: Ses dosyası işlenirken sorun oluştu. {e}")
            return ""

    def get_model_info(self) -> dict:
        """Model bilgilerini döndürür."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "has_language_model": self.decoder is not None,
            "sampling_rate": self.processor.feature_extractor.sampling_rate
        }

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

