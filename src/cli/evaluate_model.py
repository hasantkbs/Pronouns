# evaluate_model.py

import os
import sys
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import json
import pandas as pd
import torch
import librosa
import numpy as np
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor,
    WhisperForConditionalGeneration, 
    WhisperProcessor
)
from datasets import Dataset, Audio
import evaluate
from pathlib import Path
import config
from tqdm import tqdm
from src.services.reporting_service import ReportingService
from peft import PeftModel, PeftConfig

class ModelEvaluator:
    def __init__(self, user_id):
        self.user_id = user_id
        self.personalized_model_dir = Path("data/models/personalized_models") / self.user_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        
        self.load_model()

    def load_model(self):
        """Modeli ve processor'ı mimariye göre yükler."""
        if not self.personalized_model_dir.exists():
            print(f"⚠️  Kişiselleştirilmiş model bulunamadı: {self.personalized_model_dir}")
            print(f"   Varsayılan model yükleniyor: {config.MODEL_NAME}")
            self.model_type = "wav2vec2"
            self.base_model_name = config.MODEL_NAME
            self.processor = Wav2Vec2Processor.from_pretrained(self.base_model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.base_model_name).to(self.device)
            return

        # Mimariyi tespit et
        adapter_config_path = self.personalized_model_dir / "adapter_config.json"
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        self.base_model_name = adapter_config.get("base_model_name_or_path")
        
        # Whisper tespiti
        if "whisper" in self.base_model_name.lower() or adapter_config.get("auto_mapping", {}).get("base_model_class") == "WhisperForConditionalGeneration":
            self.model_type = "whisper"
            print(f"✅ Whisper mimarisi tespit edildi: {self.base_model_name}")
            self.processor = WhisperProcessor.from_pretrained(self.base_model_name, language="tr", task="transcribe")
            base_model = WhisperForConditionalGeneration.from_pretrained(self.base_model_name)
        else:
            self.model_type = "wav2vec2"
            print(f"✅ Wav2Vec2 mimarisi tespit edildi: {self.base_model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(self.base_model_name)
            base_model = Wav2Vec2ForCTC.from_pretrained(self.base_model_name)

        print(f"📥 Kişiselleştirilmiş adapter yükleniyor: {self.personalized_model_dir}")
        # Safetensors desteği için path'i tam yol olarak veriyoruz
        model_path = os.path.abspath(str(self.personalized_model_dir))
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()

    def prepare_dataset(self, max_samples=None):
        """Değerlendirme veri setini hazırlar."""
        eval_csv = Path(config.BASE_PATH) / self.user_id / "eval.csv"
        if not eval_csv.exists():
            print(f"❌ Hata: {eval_csv} bulunamadı. Lütfen önce 'prepare_training_data.py' çalıştırın.")
            return None

        df = pd.read_csv(eval_csv, encoding='utf-8')
        
        if max_samples and len(df) > max_samples:
            df = df.head(max_samples)

        # Dosya yollarının varlığını kontrol et
        df = df[df["file_path"].apply(os.path.exists)]
        
        if len(df) == 0:
            print(f"❌ Hata: Hiç geçerli ses dosyası bulunamadı!")
            return None

        dataset = Dataset.from_pandas(df)
        print(f"📊 Değerlendirme seti: {len(dataset)} örnek")
        return dataset

    def evaluate_model(self, dataset):
        """Modeli değerlendirir ve WER/CER metriklerini hesaplar."""
        if dataset is None:
            return
        
        predictions = []
        references = []

        print(f"\n🚀 {self.user_id} modeli değerlendiriliyor ({self.model_type})...")
        
        for item in tqdm(dataset, desc="Değerlendirme"):
            try:
                audio_path = item['file_path']
                reference_text = item['transcript']
                
                # Ses yükleme
                speech, sr = librosa.load(audio_path, sr=config.ORNEKLEME_ORANI)
                
                if self.model_type == "whisper":
                    # Whisper çıkarımı
                    input_features = self.processor(speech, sampling_rate=sr, return_tensors="pt").input_features.to(self.device)
                    with torch.no_grad():
                        generated_ids = self.model.generate(input_features, language="tr", task="transcribe")
                    prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                else:
                    # Wav2Vec2 çıkarımı
                    inputs = self.processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
                    input_values = inputs.input_values.to(self.device)
                    with torch.no_grad():
                        logits = self.model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    prediction = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                predictions.append(prediction.strip())
                references.append(reference_text.strip())
                
            except Exception as e:
                print(f"⚠️  Örnek işlenirken hata: {e}")
                continue

        if len(predictions) == 0:
            print("❌ Hiç tahmin yapılamadı!")
            return

        # Metrikleri hesapla
        wer = self.wer_metric.compute(predictions=predictions, references=references)
        cer = self.cer_metric.compute(predictions=predictions, references=references)

        print("\n" + "="*50)
        print(f"✅ Değerlendirme Tamamlandı!")
        print(f"   İşlenen örnek: {len(predictions)}")
        print(f"   Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
        print(f"   Character Error Rate (CER): {cer:.4f} ({cer*100:.2f}%)")
        print("="*50)
        
        # Örnek tahminler
        print("\n📝 Örnek Tahminler:")
        for i in range(min(5, len(predictions))):
            print(f"   {i+1}. Gerçek: '{references[i]}'")
            print(f"      Tahmin: '{predictions[i]}'")
            print()

        # Rapor kaydet
        reporting_service = ReportingService()
        evaluation_data = {
            "wer": wer,
            "cer": cer,
            "total_samples": len(dataset),
            "evaluated_samples": len(predictions),
            "model_path": str(self.personalized_model_dir),
            "base_model": self.base_model_name,
            "model_type": self.model_type,
            "sample_predictions": [
                {"reference": references[i], "prediction": predictions[i]} 
                for i in range(min(10, len(predictions)))
            ]
        }
        report_file = reporting_service.log_evaluation_session(self.user_id, evaluation_data)
        print(f"\n📊 Evaluation report saved: {report_file}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Kişiselleştirilmiş ASR modelini değerlendirir.")
    parser.add_argument("user_id", type=str, help="Kullanıcı kimliği")
    parser.add_argument("--max_samples", type=int, default=None, help="Maksimum örnek sayısı")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(user_id=args.user_id)
    dataset = evaluator.prepare_dataset(max_samples=args.max_samples)
    if dataset:
        evaluator.evaluate_model(dataset)

if __name__ == "__main__":
    main()
