# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Kişiselleştirilmiş Model Eğitimi

Bu script, belirli bir kullanıcıdan toplanan verileri kullanarak mevcut bir 
ASR modelini o kullanıcı için ince ayar (fine-tuning) yapar.
"""

import os
os.environ["TRANSFORMERS_DISABLE_PEFT"] = "1"

import argparse
import torch
import pandas as pd
from pathlib import Path
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, Audio
import config
from adapters import AdapterConfig
import adapters.composition
import adapters.models

# train_model.py dosyasından DataCollatorCTCWithPadding sınıfını alıyoruz
# Kod tekrarını önlemek için bu sınıf normalde paylaşılan bir modüle konulabilir.
class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

class PersonalizedTrainer:
    """Kullanıcıya özel model eğitici."""
    
    def __init__(self, user_id, base_model_path=None):
        self.user_id = user_id
        self.base_model_path = base_model_path or config.MODEL_NAME
        self.user_data_path = Path(config.BASE_PATH) / self.user_id
        self.output_dir = Path("data/models/personalized_models") / self.user_id
        self.adapter_name = "user_adapter"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None

    def run(self):
        """Kişiselleştirme sürecini başlatır."""
        print(f"🎯 {self.user_id} için kişiselleştirme süreci başlıyor.")
        print("="*50)
        
        if not self.user_data_path.exists() or not (self.user_data_path / "metadata_words.csv").exists():
            print(f"❌ Hata: {self.user_id} için veri bulunamadı.")
            print(f"Lütfen önce 'src/training/collect_user_data.py' scriptini çalıştırın.")
            return

        self.load_model_and_processor()
        dataset = self.prepare_dataset()
        self.train_model(dataset)

    def load_model_and_processor(self):
        """Temel model ve işlemciyi yükler."""
        print(f"📥 Temel model yükleniyor: {self.base_model_path}")
        self.processor = Wav2Vec2Processor.from_pretrained(self.base_model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.base_model_path)
        self.model.to(self.device)
        self.model.add_adapter(self.adapter_name, AdapterConfig.load("pfeiffer", reduction_factor=config.ADAPTER_REDUCTION_FACTOR))
        self.model.train_adapter(self.adapter_name)
        print(f"✅ Model yüklendi. Cihaz: {self.device}")

    def prepare_dataset(self):
        """Kullanıcıya özel veri setini hazırlar."""
        print(f"📊 Veri seti hazırlanıyor: {self.user_data_path}")
        metadata_path = self.user_data_path / "metadata_words.csv"
        df = pd.read_csv(metadata_path)
        
        dataset = Dataset.from_pandas(df)
        dataset = dataset.cast_column("file_path", Audio(sampling_rate=config.ORNEKLEME_ORANI))
        
        print(f"📈 Veri seti boyutu: {len(dataset)} kayıt")
        return dataset

    def preprocess_function(self, examples):
        """Veri ön işleme fonksiyonu."""
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = self.processor(
            audio_arrays, 
            sampling_rate=config.ORNEKLEME_ORANI, 
            return_tensors="pt", 
            padding=True
        )
        
        with self.processor.as_target_processor():
            labels = self.processor(examples["transcription"]).input_ids
            
        examples["input_values"] = inputs.input_values[0]
        examples["labels"] = labels
        return examples

    def train_model(self, dataset):
        """Modeli ince ayar (fine-tuning) ile eğitir."""
        print("🚀 Kişiselleştirilmiş model eğitimi başlıyor...")
        
        processed_dataset = dataset.map(
            self.preprocess_function,
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=2 # Küçük veri setleri için batch_size'ı düşür
        )
        
        # İnce ayar için eğitim parametreleri config.py dosyasından okunur
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
            num_train_epochs=config.NUM_FINETUNE_EPOCHS,
            learning_rate=config.FINETUNE_LEARNING_RATE,
            fp16=torch.cuda.is_available(),
            save_total_limit=1,
            logging_steps=config.FINETUNE_LOGGING_STEPS,
            eval_steps=config.FINETUNE_EVAL_STEPS,
            evaluation_strategy="no",  # Değerlendirme için ayrı bir set yok
            save_strategy="epoch"
        )
        
        data_collator = DataCollatorCTCWithPadding(
            processor=self.processor, 
            padding=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset,
            tokenizer=self.processor.feature_extractor,
            data_collator=data_collator,
        )
        
        trainer.train()
        
        print("✅ Model ince ayarı tamamlandı!")
        self.model.save_adapter(str(self.output_dir), self.adapter_name)
        
        print(f"💾 Kişiselleştirilmiş model kaydedildi: {self.output_dir}")
        print("\nKullanım için app.py veya config.py dosyasını bu yeni model yolunu kullanacak şekilde güncelleyebilirsiniz.")

def main():
    parser = argparse.ArgumentParser(description="Kullanıcıya özel ASR modelini eğitir.")
    parser.add_argument("user_id", type=str, help="Verisi kullanılacak ve modeli kişiselleştirilecek kullanıcının kimliği.")
    parser.add_argument("--base_model", type=str, help="İnce ayar için kullanılacak temel modelin yolu. Varsayılan: config.py'deki model.", default=None)
    
    args = parser.parse_args()
    
    trainer = PersonalizedTrainer(user_id=args.user_id, base_model_path=args.base_model)
    trainer.run()

if __name__ == "__main__":
    main()
