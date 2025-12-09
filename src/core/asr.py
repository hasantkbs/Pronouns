
# -*- coding: utf-8 -*-
import os
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import config

class ASRSystem:
    """Otomatik Konuşma Tanıma (ASR) sistemi."""

    def __init__(self, model_name="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if model_name is a path to a personalized model
        if os.path.exists(model_name):
            # It's a personalized model
            base_model_name = config.MODEL_NAME
            peft_model_path = model_name
            
            self.processor = WhisperProcessor.from_pretrained(base_model_name, language="tr", task="transcribe")
            
            base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
            
            self.model = PeftModel.from_pretrained(base_model, peft_model_path)
            self.model.to(self.device)
            print(f"ASR Sistemi başlatıldı. Kişiselleştirilmiş Model: {peft_model_path}, Cihaz: {self.device}")
        else:
            # It's a standard model from the hub
            self.processor = WhisperProcessor.from_pretrained(model_name, language="tr", task="transcribe")
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            print(f"ASR Sistemi başlatıldı. Model: {model_name}, Cihaz: {self.device}")

    def transcribe(self, audio_path):
        """Verilen ses dosyasını metne dönüştürür."""
        try:
            speech, sr = librosa.load(audio_path, sr=16000)
            
            input_features = self.processor(speech, sampling_rate=sr, return_tensors="pt").input_features
            input_features = input_features.to(self.device)
            
            predicted_ids = self.model.generate(input_features)
            
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            return transcription
        except Exception as e:
            print(f"ASR transkripsiyon hatası: {e}")
            return None

if __name__ == '__main__':
    # This part is for testing the ASRSystem class directly.
    # It requires a valid audio file.
    
    # Example with a personalized model (if available)
    user_id = "Furkan"
    personalized_model_dir = f"data/models/personalized_models/{user_id}"
    
    if os.path.exists(personalized_model_dir):
        print("Kişiselleştirilmiş model ile test ediliyor...")
        asr_system = ASRSystem(model_name=personalized_model_dir)
    else:
        print("Varsayılan 'base' modeli ile test ediliyor...")
        asr_system = ASRSystem(model_name="base")

    # You need to provide a path to a test audio file.
    # For example:
    test_file = "data/users/Furkan/words/Furkan_kelime_1_rep1.wav" 
    
    if os.path.exists(test_file):
        print(f"\n--- ASR Testi Başlatılıyor ---")
        print(f"Test dosyası: {test_file}")
        recognized_text = asr_system.transcribe(test_file)
        print(f"\nTest tamamlandı.")
        print(f"Tanınan metin: '{recognized_text}'")
    else:
        print(f"Test dosyası bulunamadı: {test_file}")
