# evaluate_model.py

import os
import pandas as pd
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
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
        
        # Whisper modeline göre base_model_name'i sabitliyoruz.
        self.base_model_name = "openai/whisper-large-v2"

        print(f"✅ {self.user_id} için kişiselleştirilmiş model yüklenecek: {self.personalized_model_dir}")
        
        # Whisper Processor'ı yüklüyoruz
        self.processor = WhisperProcessor.from_pretrained(self.base_model_name, language="tr", task="transcribe")
        
        # Temel Whisper modelini ve üzerine eğitilmiş PEFT adaptörünü yüklüyoruz
        from peft import PeftModel
        base_model = WhisperForConditionalGeneration.from_pretrained(self.base_model_name)
        self.model = PeftModel.from_pretrained(base_model, str(self.personalized_model_dir))
            
        self.model.to(self.device)
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def prepare_dataset(self, max_samples=None):
        """
        Değerlendirme veri setini hazırlar.
        
        Args:
            max_samples: Maksimum örnek sayısı (None ise tümü)
        
        Returns:
            Dataset veya None (hata durumunda)
        """
        # Önce eval.csv'yi kontrol et
        eval_csv = Path(config.BASE_PATH) / self.user_id / "eval.csv"
        if eval_csv.exists():
            print(f"   ✅ eval.csv bulundu, kullanılıyor.")
            df = pd.read_csv(eval_csv, encoding='utf-8')
        elif self.data_path.exists():
            print(f"   ⚠️  eval.csv bulunamadı, metadata_words.csv kullanılıyor.")
            df = pd.read_csv(self.data_path, encoding='utf-8')
            df = df[['file_path', 'transcription']].copy()
            df.rename(columns={'transcription': 'transcript'}, inplace=True)
        else:
            print(f"❌ Hata: Ne eval.csv ne de {self.data_path} bulunamadı.")
            return None

        # Maksimum örnek sayısı sınırı
        if max_samples and len(df) > max_samples:
            print(f"   ⚠️  {len(df)} örnek var, {max_samples} ile sınırlandırılıyor.")
            df = df.head(max_samples)

        # Dosya yollarını düzelt
        words_dir = Path(config.BASE_PATH) / self.user_id / "words"
        df["file_path"] = df["file_path"].apply(
            lambda x: str(words_dir / os.path.basename(str(x)))
        )
        
        # Var olmayan dosyaları filtrele
        original_size = len(df)
        df = df[df["file_path"].apply(os.path.exists)]
        if len(df) < original_size:
            print(f"   ⚠️  {original_size - len(df)} adet bulunamayan ses dosyası atlandı.")

        if len(df) == 0:
            print(f"❌ Hata: Hiç geçerli ses dosyası bulunamadı!")
            return None

        # Transcript sütununu kontrol et
        transcript_col = 'transcript' if 'transcript' in df.columns else 'transcription'
        df = df[df[transcript_col].notna() & (df[transcript_col].str.strip() != '')]

        dataset = Dataset.from_pandas(df).cast_column(
            "file_path", 
            Audio(sampling_rate=config.ORNEKLEME_ORANI, decode=False)
        )
        
        print(f"   📊 Değerlendirme seti: {len(dataset)} örnek")
        return dataset

    def evaluate_model(self, dataset, max_samples=None):
        """
        Modeli değerlendirir ve WER/CER metriklerini hesaplar.
        
        Args:
            dataset: Değerlendirme veri seti
            max_samples: Maksimum değerlendirilecek örnek sayısı
        """
        if dataset is None:
            print("❌ Değerlendirme veri seti yok!")
            return
        
        from torch.utils.data import DataLoader

        # Whisper için özel collate fonksiyonu
        def collate_fn(batch):
            input_features = []
            labels = []
            
            for item in batch:
                try:
                    # Audio yükleme, processor input_features'ı oluşturacak
                    audio_input = item['file_path']['array'] # already loaded by datasets
                    
                    # Transcription tokenization
                    label = self.processor.tokenizer(item['transcript']).input_ids
                    
                    input_features.append({"input_features": audio_input})
                    labels.append({"input_ids": label})

                except Exception as e:
                    print(f"⚠️  Ses veya transkript işlenirken hata: {e}")
                    continue
            
            if not input_features:
                return None
            
            # Input features'ı batch halinde hazırla
            batch_input_features = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            # Labels'ı batch halinde hazırla ve padding tokenlarını -100 ile değiştir
            batch_labels = self.processor.tokenizer.pad(labels, return_tensors="pt")
            batch_labels["input_ids"] = batch_labels["input_ids"].masked_fill(
                batch_labels.attention_mask.ne(1), -100
            )

            return {
                "input_features": batch_input_features.input_features,
                "labels": batch_labels.input_ids,
                "attention_mask": batch_input_features.attention_mask
            }

        dataloader = DataLoader(
            dataset, 
            batch_size=config.FINETUNE_BATCH_SIZE, # config'den al
            collate_fn=collate_fn,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY,
            prefetch_factor=config.DATALOADER_PREFETCH_FACTOR if config.DATALOADER_NUM_WORKERS > 0 else None,
            persistent_workers=True if config.DATALOADER_NUM_WORKERS > 0 else False
        )

        predictions = []
        references = []

        print(f"\n🚀 {self.user_id} modeli değerlendiriliyor...")
        print(f"   Toplam kayıt: {len(dataset)}")
        
        self.model.eval()
        processed_count = 0
        
        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Değerlendirme"):
                    if batch is None:
                        continue
                    
                    input_features = batch["input_features"].to(self.device)
                    # Whisper modeli için generate metodunu kullanıyoruz
                    generated_ids = self.model.generate(
                        input_features=input_features,
                        # attention_mask=batch.get("attention_mask", None).to(self.device), # Whisper'da generate için attention_mask gerekli değil
                        language="tr", # Türkçe dilini belirt
                        task="transcribe", # Transkripsiyon görevi
                        return_timestamps=False # Zaman damgalarını döndürme
                    )
                    
                    # Tahminleri çözümlüyoruz
                    transcription = self.processor.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )

                    # Referans metinleri çözümlüyoruz (padding'i kaldırarak)
                    labels = batch["labels"].cpu().numpy()
                    labels[labels == -100] = self.processor.tokenizer.pad_token_id
                    reference_texts = self.processor.batch_decode(
                        labels, 
                        skip_special_tokens=True
                    )

                    predictions.extend(transcription)
                    references.extend(reference_texts)
                    processed_count += len(transcription)
                    
                    if max_samples and processed_count >= max_samples:
                        break

            if len(predictions) == 0:
                print("❌ Hiç tahmin yapılamadı!")
                return

            # Metrikleri hesapla
            wer = self.wer_metric.compute(
                predictions=predictions, 
                references=references
            )
            cer = self.cer_metric.compute(
                predictions=predictions, 
                references=references
            )

            print("\n" + "="*50)
            print(f"✅ Değerlendirme Tamamlandı!")
            print(f"   İşlenen örnek: {len(predictions)}")
            print(f"   Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
            print(f"   Character Error Rate (CER): {cer:.4f} ({cer*100:.2f}%)")
            print("="*50)
            
            # Örnek tahminler göster
            if len(predictions) > 0:
                print("\n📝 Örnek Tahminler:")
                for i in range(min(5, len(predictions))):
                    print(f"   {i+1}. Gerçek: '{references[i]}'")
                    print(f"      Tahmin: '{predictions[i]}'")
                    print()

            self.provide_suggestions(wer, cer)
            
        except Exception as e:
            print(f"\n❌ Değerlendirme sırasında hata: {e}")
            import traceback
            traceback.print_exc()

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
    parser = argparse.ArgumentParser(
        description="Kişiselleştirilmiş ASR modelini değerlendirir."
    )
    parser.add_argument(
        "user_id", 
        type=str, 
        help="Değerlendirilecek kullanıcının kimliği (örn: Furkan)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maksimum değerlendirilecek örnek sayısı (varsayılan: tümü)"
    )
    
    args = parser.parse_args()

    print("="*50)
    print(f"Model Değerlendirme: {args.user_id}")
    print("="*50)
    
    evaluator = ModelEvaluator(user_id=args.user_id)
    dataset = evaluator.prepare_dataset(max_samples=args.max_samples)
    if dataset:
        evaluator.evaluate_model(dataset, max_samples=args.max_samples)
    else:
        print("❌ Değerlendirme yapılamadı!")

if __name__ == "__main__":
    main()