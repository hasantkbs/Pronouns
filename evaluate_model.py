# evaluate_model.py

import os
import pandas as pd
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Dataset, Audio
import evaluate
from pathlib import Path
import config
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, user_id, model_path=None):
        self.user_id = user_id
        self.personalized_model_dir = Path("data/models/personalized_models") / self.user_id
        self.data_path = Path(config.BASE_PATH) / self.user_id / "metadata_words.csv"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.base_model_name = config.MODEL_NAME

        if self.personalized_model_dir.exists():
            self.model_to_load = str(self.personalized_model_dir)
            print(f"✅ {self.user_id} için kişiselleştirilmiş model yüklenecek: {self.model_to_load}")
        else:
            self.model_to_load = model_path or self.base_model_name
            print(f"ℹ️  Kişiselleştirilmiş model bulunamadı. Varsayılan model kullanılacak: {self.model_to_load}")

        self.processor = Wav2Vec2Processor.from_pretrained(self.base_model_name)
        
        if self.personalized_model_dir.exists():
            from peft import PeftModel
            base_model = Wav2Vec2ForCTC.from_pretrained(self.base_model_name)
            self.model = PeftModel.from_pretrained(base_model, str(self.personalized_model_dir))
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_to_load)
            
        self.model.to(self.device)
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def prepare_dataset(self):
        if not self.data_path.exists():
            print(f"❌ Hata: {self.data_path} bulunamadı. Lütfen doğru yolu sağlayın.")
            return None

        df = pd.read_csv(self.data_path).head(500)

        df["file_path"] = df["file_path"].apply(
            lambda x: str(Path(config.BASE_PATH) / self.user_id / "words" / os.path.basename(x))
        )
        # Filter out non-existent files
        df = df[df["file_path"].apply(os.path.exists)]

        dataset = Dataset.from_pandas(df).cast_column("file_path", Audio(sampling_rate=config.ORNEKLEME_ORANI, decode=False))
        return dataset

    def evaluate_model(self, dataset):
        if dataset is None:
            return
        
        from torch.utils.data import DataLoader

        def collate_fn(batch):
            audio_arrays = [librosa.load(item['file_path']['path'], sr=config.ORNEKLEME_ORANI)[0] for item in batch]
            reference_texts = [item['transcription'] for item in batch]
            input_features = self.processor(audio_arrays, sampling_rate=config.ORNEKLEME_ORANI, return_tensors="pt", padding=True, truncation=True, max_length=96000).input_values
            return {"input_features": input_features, "reference_texts": reference_texts}

        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)


        predictions = []
        references = []

        print(f"🚀 {self.user_id} modeli değerlendiriliyor... (Toplam {len(dataset)} kayıt)")
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_features = batch["input_features"].to(self.device)

                logits = self.model(input_features).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)

                predictions.extend(transcription)
                references.extend(batch["reference_texts"])

        wer = self.wer_metric.compute(predictions=predictions, references=references)
        cer = self.cer_metric.compute(predictions=predictions, references=references)

        print("\n=========================================")
        print(f"✅ Değerlendirme Tamamlandı!")
        print(f"   Word Error Rate (WER): {wer:.4f}")
        print(f"   Character Error Rate (CER): {cer:.4f}")
        print("=========================================")
        print("\n⚠️  Not: Bu değerlendirme eğitim verisi üzerinde yapılmıştır ve modelin gerçek performansını yansıtmayabilir.")

        self.provide_suggestions(wer, cer)

    def provide_suggestions(self, wer, cer):
        print("\n💡 Geliştirme Önerileri:")
        if wer > 0.3 or cer > 0.15: # Arbitrary thresholds for "poor" performance
            print("   - Daha fazla ve çeşitli veri toplayın.")
            print("   - `train_adapter.py` içindeki eğitim parametrelerini (epoch, öğrenme oranı) ayarlamayı deneyin.")
        elif wer > 0.15 or cer > 0.05:
            print("   - Model performansı iyi görünüyor. Daha fazla veri ile daha da iyileştirilebilir.")
        else:
            print("   - Model performansı oldukça başarılı!")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Kişiselleştirilmiş ASR modelini değerlendirir.")
    parser.add_argument("user_id", type=str, help="Değerlendirilecek kullanıcının kimliği.")
    args = parser.parse_args()

    evaluator = ModelEvaluator(user_id=args.user_id)
    dataset = evaluator.prepare_dataset()
    if dataset:
        evaluator.evaluate_model(dataset)

if __name__ == "__main__":
    main()