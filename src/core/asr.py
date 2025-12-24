# -*- coding: utf-8 -*-
import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from peft import PeftModel
import config

class ASRSystem:
    """Otomatik Konuşma Tanıma (ASR) sistemi - Wav2Vec2 tabanlı."""

    def __init__(self, model_name=None):
        """
        ASR sistemi başlatır.
        
        Args:
            model_name: Model yolu veya Hugging Face model ID'si. 
                       None ise config'deki varsayılan model kullanılır.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Eğer model_name belirtilmemişse, config'deki varsayılan modeli kullan
        if model_name is None:
            model_name = config.MODEL_NAME
        
        # Kişiselleştirilmiş model kontrolü
        if os.path.exists(model_name) and os.path.isdir(model_name):
            # Kişiselleştirilmiş model (PEFT/LoRA adapter)
            base_model_name = config.MODEL_NAME
            peft_model_path = model_name
            
            try:
                print(f"📥 Temel model yükleniyor: {base_model_name}")
                self.processor = Wav2Vec2Processor.from_pretrained(base_model_name)
                base_model = Wav2Vec2ForCTC.from_pretrained(
                    base_model_name,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    vocab_size=len(self.processor.tokenizer)
                )
                
                print(f"📥 Kişiselleştirilmiş adapter yükleniyor: {peft_model_path}")
                self.model = PeftModel.from_pretrained(base_model, peft_model_path)
                self.model.to(self.device)
                self.model.eval()  # Inference modu
                print(f"✅ ASR Sistemi başlatıldı. Kişiselleştirilmiş Model: {peft_model_path}")
                print(f"   Cihaz: {self.device}")
            except Exception as e:
                print(f"❌ Kişiselleştirilmiş model yüklenirken hata: {e}")
                print(f"⚠️  Varsayılan model kullanılıyor: {base_model_name}")
                self._load_base_model(base_model_name)
        else:
            # Standart model (Hugging Face hub'dan veya yerel)
            self._load_base_model(model_name)

    def _load_base_model(self, model_name):
        """Temel modeli yükler."""
        try:
            print(f"📥 Model yükleniyor: {model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Inference modu
            print(f"✅ ASR Sistemi başlatıldı. Model: {model_name}")
            print(f"   Cihaz: {self.device}")
        except Exception as e:
            print(f"❌ Model yüklenirken kritik hata: {e}")
            raise

    def transcribe(self, audio_path):
        """
        Verilen ses dosyasını metne dönüştürür.
        
        Args:
            audio_path: Ses dosyasının yolu
            
        Returns:
            str: Tanınan metin veya None (hata durumunda)
        """
        try:
            # Ses dosyasını yükle
            speech, sr = librosa.load(audio_path, sr=config.ORNEKLEME_ORANI)
            
            # Boş ses kontrolü
            if len(speech) == 0 or np.max(np.abs(speech)) < 0.001:
                print("⚠️  Sessizlik algılandı veya ses dosyası çok kısa.")
                return None
            
            # Processor ile özellik çıkarımı
            input_values = self.processor(
                speech, 
                sampling_rate=sr, 
                return_tensors="pt",
                padding=True
            ).input_values
            
            input_values = input_values.to(self.device)
            
            # Model ile tahmin
            with torch.no_grad():
                logits = self.model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
            
            # Metne dönüştür
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Boş sonuç kontrolü
            if not transcription or transcription.strip() == "":
                return None
                
            return transcription.strip()
            
        except FileNotFoundError:
            print(f"❌ Ses dosyası bulunamadı: {audio_path}")
            return None
        except Exception as e:
            print(f"❌ ASR transkripsiyon hatası: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == '__main__':
    """ASR sistemi test fonksiyonu."""
    import sys
    
    # Test için kullanıcı ID'si
    user_id = "Furkan"
    personalized_model_dir = f"data/models/personalized_models/{user_id}"
    
    print("=" * 50)
    print("ASR Sistemi Test Modu")
    print("=" * 50)
    
    # Model yükleme
    if os.path.exists(personalized_model_dir):
        print(f"✅ Kişiselleştirilmiş model bulundu: {personalized_model_dir}")
        asr_system = ASRSystem(model_name=personalized_model_dir)
    else:
        print(f"ℹ️  Kişiselleştirilmiş model bulunamadı, varsayılan model kullanılıyor.")
        asr_system = ASRSystem()

    # Test dosyası
    test_file = "data/users/Furkan/words/Furkan_kelime_1_rep1.wav"
    
    # Komut satırından dosya yolu verilmişse onu kullan
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    if os.path.exists(test_file):
        print(f"\n--- ASR Testi Başlatılıyor ---")
        print(f"📁 Test dosyası: {test_file}")
        recognized_text = asr_system.transcribe(test_file)
        
        if recognized_text:
            print(f"\n✅ Test tamamlandı.")
            print(f"📝 Tanınan metin: '{recognized_text}'")
        else:
            print(f"\n❌ Tanıma başarısız veya sessizlik algılandı.")
    else:
        print(f"\n❌ Test dosyası bulunamadı: {test_file}")
        print(f"💡 Kullanım: python src/core/asr.py [ses_dosyasi_yolu]")
