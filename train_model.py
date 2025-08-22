# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Model Eğitimi
Mevcut downloaded_data klasöründeki verileri kullanarak model eğitir.
"""

import os
import sys
import torch
import torchaudio
import soundfile as sf
import numpy as np
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, Audio, Features, Value
import config

# DataCollatorCTCWithPadding sınıfını kendimiz oluşturuyoruz
class DataCollatorCTCWithPadding:
    """
    CTC tabanlı ASR için data collator.
    Değişken uzunluktaki ses ve etiket dizilerini toplu halde işler.
    """
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        # input_values
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

class SpeechDisorderTrainer:
    """Konuşma bozukluğu için özel model eğitici."""
    
    def __init__(self):
        self.model_name = config.MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        
    def load_model_and_processor(self):
        """Model ve işlemciyi yükler."""
        print(f"📥 Model yükleniyor: {self.model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        print(f"✅ Model yüklendi. Cihaz: {self.device}")
        
    def prepare_dataset(self, data_path: str, split: str = "train"):
        """Veri setini hazırlar."""
        print(f"📊 Veri seti hazırlanıyor: {data_path}")
        
        # Parquet dosyasını oku
        try:
            import pandas as pd
            df = pd.read_parquet(data_path)
            print(f"📈 Veri seti boyutu: {len(df)} kayıt")
            print(f"📋 Sütunlar: {df.columns.tolist()}")
        except ImportError:
            print("⚠️  Pandas yüklü değil, veri seti bilgisi alınamadı.")
            return None
            
        # Veri setini Dataset formatına çevir
        dataset = Dataset.from_pandas(df)
        
        # Ses dosyalarını yükle
        dataset = dataset.cast_column("path", Audio())
        
        return dataset
        
    def preprocess_function(self, examples):
        """Veri ön işleme fonksiyonu."""
        audio = examples["audio"]
        
        # Ses verisini işle - farklı formatları kontrol et
        if isinstance(audio, dict) and "array" in audio:
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]
        elif isinstance(audio, dict) and "samples" in audio:
            audio_array = audio["samples"]
            sampling_rate = audio["sampling_rate"]
        else:
            # Ses dosyasını doğrudan oku
            try:
                audio_array, sampling_rate = sf.read(examples["path"])
            except:
                # Varsayılan değerler
                audio_array = np.zeros(16000)  # 1 saniye sessizlik
                sampling_rate = 16000
        
        # Ses verisini işle
        inputs = self.processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt", 
            padding=True
        )
        
        # Etiketleri işle
        with self.processor.as_target_processor():
            labels = self.processor(examples["sentence"]).input_ids
            
        examples["input_values"] = inputs.input_values[0]
        examples["labels"] = labels
        
        return examples
        
    def train_model(self, train_dataset, eval_dataset=None):
        """Modeli eğitir."""
        print("🚀 Model eğitimi başlıyor...")
        
        # Veri ön işleme
        train_dataset = train_dataset.map(
            self.preprocess_function,
            remove_columns=train_dataset.column_names,
            batch_size=8
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                self.preprocess_function,
                remove_columns=eval_dataset.column_names,
                batch_size=8
            )
        
        # Eğitim parametreleri
        training_args = TrainingArguments(
            output_dir="./trained_model",
            group_by_length=True,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="steps",
            num_train_epochs=3,
            fp16=False,  # CPU için FP16 kapalı
            save_steps=400,
            eval_steps=400,
            logging_steps=400,
            learning_rate=1e-4,
            weight_decay=0.005,
            warmup_steps=1000,
            save_total_limit=2,
        )
        
        # Data collator
        data_collator = DataCollatorCTCWithPadding(
            processor=self.processor, 
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor.feature_extractor,
            data_collator=data_collator,
        )
        
        # Eğitim
        trainer.train()
        
        # Modeli kaydet
        trainer.save_model()
        self.processor.save_pretrained("./trained_model")
        
        print("✅ Model eğitimi tamamlandı!")
        print("💾 Model kaydedildi: ./trained_model/")
        
    def main(self):
        """Ana eğitim fonksiyonu."""
        print("🎯 Konuşma Bozukluğu Model Eğitimi")
        print("=" * 50)
        
        # Model yükle
        self.load_model_and_processor()
        
        # Veri setlerini hazırla
        train_path = "downloaded_data/train/0000.parquet"
        eval_path = "downloaded_data/valudation/0000.parquet"
        
        if os.path.exists(train_path):
            train_dataset = self.prepare_dataset(train_path, "train")
        else:
            print(f"❌ Train veri seti bulunamadı: {train_path}")
            return
            
        eval_dataset = None
        if os.path.exists(eval_path):
            eval_dataset = self.prepare_dataset(eval_path, "validation")
        
        # Modeli eğit
        self.train_model(train_dataset, eval_dataset)
        
        print("\n🎉 Eğitim tamamlandı!")
        print("📝 Kullanım:")
        print("   config.py dosyasında MODEL_NAME'i './trained_model' olarak değiştirin")

if __name__ == "__main__":
    trainer = SpeechDisorderTrainer()
    trainer.main()
