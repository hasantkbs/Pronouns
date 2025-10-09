import os
import pandas as pd
import torch
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio
import evaluate
from pathlib import Path
import config

class ModelEvaluator:
    def __init__(self, user_id, model_path=None):
        self.user_id = user_id
        self.personalized_model_dir = Path("data/models/personalized_models") / self.user_id
        self.data_path = Path(config.BASE_PATH) / self.user_id / "metadata_words.csv"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.personalized_model_dir.exists():
            self.model_to_load = str(self.personalized_model_dir)
            print(f"✅ {self.user_id} için kişiselleştirilmiş model yüklenecek: {self.model_to_load}")
        else:
            self.model_to_load = model_path or "openai/whisper-base" # Default Whisper base model
            print(f"ℹ️  Kişiselleştirilmiş model bulunamadı. Varsayılan model kullanılacak: {self.model_to_load}")

        self.processor = WhisperProcessor.from_pretrained(self.model_to_load if self.personalized_model_dir.exists() else "openai/whisper-base", language="tr", task="transcribe")
        
        if self.personalized_model_dir.exists():
            from peft import PeftModel
            base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            self.model = PeftModel.from_pretrained(base_model, str(self.personalized_model_dir))
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_to_load)
            
        self.model.to(self.device)
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def prepare_dataset(self):
        if not self.data_path.exists():
            print(f"❌ Hata: {self.data_path} bulunamadı. Lütfen doğru yolu sağlayın.")
            return None

        df = pd.read_csv(self.data_path)

        def audio_loader(path):
            filename = os.path.basename(path)
            filepath = Path(config.BASE_PATH) / self.user_id / "words" / filename
            speech, sample_rate = librosa.load(filepath, sr=config.ORNEKLEME_ORANI)
            return speech

        df["audio"] = df["file_path"].apply(audio_loader)
        dataset = Dataset.from_pandas(df)
        return dataset

    def evaluate_model(self, dataset):
        if dataset is None:
            return

        predictions = []
        references = []

        print(f"🚀 {self.user_id} modeli değerlendiriliyor... (Toplam {len(dataset)} kayıt)")
        self.model.eval()
        with torch.no_grad():
            for i, item in enumerate(dataset):
                audio_input = item["audio"]
                reference_text = item["transcription"]

                # Preprocess audio
                input_features = self.processor(audio_input, sampling_rate=config.ORNEKLEME_ORANI, return_tensors="pt").input_features
                input_features = input_features.to(self.device)

                # Generate transcription
                generated_ids = self.model.generate(inputs=input_features, language="tr", task="transcribe")
                transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                predictions.append(transcription)
                references.append(reference_text)
                
                print(f"[{i+1}/{len(dataset)}] Ref: '{reference_text}' | Pred: '{transcription}'")

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
            print("   - Daha fazla ve çeşitli veri toplayın. Özellikle modelin zorlandığı kelimeler/cümleler üzerinde yoğunlaşın.")
            print("   - Kayıt ortamının kalitesini artırın (daha az gürültü, daha iyi mikrofon).")
            print("   - `personalize_model.py` içindeki eğitim parametrelerini (epoch sayısı, öğrenme oranı) ayarlamayı deneyin.")
            print("   - Eğer model hala kötü performans gösteriyorsa, daha güçlü bir temel ASR modeli araştırmayı düşünebilirsiniz.")
        elif wer > 0.15 or cer > 0.05: # Arbitrary thresholds for "moderate" performance
            print("   - Model performansı iyi görünüyor. Daha fazla veri ile daha da iyileştirilebilir.")
            print("   - Özellikle yanlış tanınan kelimeleri analiz ederek bu kelimeler için ek veri toplayabilirsiniz.")
            print("   - Dil modeli (KenLM) entegrasyonunu kontrol edin veya daha güçlü bir dil modeli kullanmayı düşünün.")
        else:
            print("   - Model performansı oldukça başarılı! Daha fazla iyileştirme için çok spesifik hatalara odaklanılabilir.")
            print("   - Farklı konuşma hızları veya tonlamalar için ek veri toplayarak genellenebilirliği artırabilirsiniz.")

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
    # Set environment variable for MPS fallback if on Apple Silicon
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    main()
