import os
import argparse
import torch
import pandas as pd
from pathlib import Path
import dataclasses
from typing import List, Dict, Union
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm
import config
import librosa
import soundfile as sf
from datasets import Dataset, Audio, load_metric
from src.utils.utils import save_model
from sklearn.model_selection import train_test_split


@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

class PersonalizedTrainer:
    """Kullanıcıya özel model eğitici."""
    
    def __init__(self, user_id, base_model_path=None):
        self.user_id = user_id
        self.base_model_path = base_model_path or "openai/whisper-large-v2"
        self.user_data_path = Path(config.BASE_PATH) / self.user_id
        self.output_dir = Path("data/models/personalized_models") / self.user_id
        self.adapter_name = "user_adapter"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.wer_metric = load_metric("wer")

    def run(self):
        """Kişiselleştirme sürecini başlatır."""
        print(f"🎯 {self.user_id} için kişiselleştirme süreci başlıyor.")
        print("="*50)
        
        if not self.user_data_path.exists() or not (self.user_data_path / "metadata_words.csv").exists():
            print(f"❌ Hata: {self.user_data_path} için veri bulunamadı.")
            return

        self.load_model_and_processor()
        
        df = pd.read_csv(self.user_data_path / "metadata_words.csv")
        train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_dataset = self.prepare_dataset(train_df)
        eval_dataset = self.prepare_dataset(eval_df)
        
        self.train_model(train_dataset, eval_dataset)

    def load_model_and_processor(self):
        """Temel model ve işlemciyi yükler."""
        print(f"📥 Temel model yükleniyor: {self.base_model_path}")
        self.processor = WhisperProcessor.from_pretrained(self.base_model_path, language="tr", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(self.base_model_path)
        self.model.to(self.device)
        peft_config = LoraConfig(
            r=config.ADAPTER_REDUCTION_FACTOR,
            lora_alpha=config.ADAPTER_REDUCTION_FACTOR * 2,
            target_modules=["q_proj", "v_proj", "k_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        print(f"✅ Model yüklendi. Cihaz: {self.device}")

    def prepare_dataset(self, df):
        """Kullanıcıya özel veri setini hazırlar."""
        print(f"📊 Veri seti hazırlanıyor...")

        def audio_loader(path):
            filename = os.path.basename(path)
            filepath = self.user_data_path / "words" / filename
            try:
                speech, _ = librosa.load(filepath, sr=config.ORNEKLEME_ORANI)
                return speech
            except FileNotFoundError:
                print(f"⚠️  Uyarı: Ses dosyası bulunamadı, atlanıyor: {filepath}")
                return None
            except Exception as e:
                print(f"❌ Hata yüklenirken: {filepath} - {e}")
                return None

        df["audio"] = df["file_path"].apply(audio_loader)
        df = df.dropna(subset=["audio"])
        
        dataset = Dataset.from_pandas(df)
        print(f"📈 Veri seti boyutu: {len(dataset)} kayıt")
        return dataset

    def preprocess_function(self, examples):
        """Veri ön işleme fonksiyonu."""
        audio_arrays = [x for x in examples["audio"]]
        
        model_inputs = self.processor(audio_arrays, sampling_rate=config.ORNEKLEME_ORANI, return_tensors="pt", padding="max_length", truncation=True)
        
        labels = self.processor.tokenizer(text=examples["transcription"], padding=True, truncation=True).input_ids
        model_inputs["labels"] = labels

        return model_inputs

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def train_model(self, train_dataset, eval_dataset):
        """Modeli, transformers.Trainer kullanarak eğitir."""
        print("🚀 Kişiselleştirilmiş model eğitimi başlıyor...")

        processed_train_dataset = train_dataset.map(
            self.preprocess_function,
            remove_columns=train_dataset.column_names,
            batched=True,
            batch_size=config.FINETUNE_BATCH_SIZE
        )
        processed_eval_dataset = eval_dataset.map(
            self.preprocess_function,
            remove_columns=eval_dataset.column_names,
            batched=True,
            batch_size=config.FINETUNE_BATCH_SIZE
        )

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=config.FINETUNE_LEARNING_RATE,
            warmup_steps=50,
            num_train_epochs=config.NUM_FINETUNE_EPOCHS,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            run_name=f"personalize-{self.user_id}",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_train_dataset,
            eval_dataset=processed_eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        trainer.train()
        
        print("\n✅ Model ince ayarı tamamlandı!")
        
        unwrapped_model = self.model
        save_model(unwrapped_model, self.processor, str(self.output_dir))

        print(f"💾 Kişiselleştirilmiş model kaydedildi: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Kullanıcıya özel ASR modelini eğitir.")
    parser.add_argument("user_id", type=str, help="Verisi kullanılacak ve modeli kişiselleştirilecek kullanıcının kimliği.")
    parser.add_argument("--base_model", type=str, help="İnce ayar için kullanılacak temel modelin yolu.", default=None)
    
    args = parser.parse_args()
    
    trainer = PersonalizedTrainer(user_id=args.user_id, base_model_path=args.base_model)
    trainer.run()

if __name__ == "__main__":
    main()