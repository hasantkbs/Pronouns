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
from datasets import Dataset, Audio
from src.utils.utils import save_model


@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

class PersonalizedTrainer:
    """Kullanıcıya özel model eğitici."""
    
    def __init__(self, user_id, base_model_path=None):
        self.user_id = user_id
        self.base_model_path = base_model_path or "openai/whisper-large-v2" # Default Whisper large model
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
            print(f"❌ Hata: {self.user_data_path} için veri bulunamadı.")
            print(f"Lütfen önce 'src/training/collect_user_data.py' scriptini çalıştırın.")
            return

        self.load_model_and_processor()
        dataset = self.prepare_dataset()
        self.train_model(dataset)

    def load_model_and_processor(self):
        """Temel model ve işlemciyi yükler."""
        print(f"📥 Temel model yükleniyor: {self.base_model_path}")
        self.processor = WhisperProcessor.from_pretrained(self.base_model_path, language="tr", task="transcribe")

        # Performans Optimizasyonu: Mümkünse Flash Attention 2'yi etkinleştir
        try:
            self.model = WhisperForConditionalGeneration.from_pretrained(self.base_model_path, attn_implementation="flash_attention_2")
            print("⚡️ Model, Flash Attention 2 optimizasyonu ile yükleniyor.")
        except (ImportError, ValueError):
            print("⚠️ Flash Attention 2 kullanılamıyor. Standart dikkat mekanizması ile devam ediliyor.")
            self.model = WhisperForConditionalGeneration.from_pretrained(self.base_model_path)
        
        self.model.to(self.device)
        peft_config = LoraConfig(
            r=config.ADAPTER_REDUCTION_FACTOR,
            lora_alpha=config.ADAPTER_REDUCTION_FACTOR * 2, # A common heuristic
            target_modules=["q_proj", "v_proj", "k_proj"], # Common target modules for Whisper
            lora_dropout=0.1, # Example dropout
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        print(f"✅ Model PEFT/LoRA ile sarmalandı. Cihaz: {self.device}")

    def prepare_dataset(self):
        """Kullanıcıya özel veri setini hazırlar."""
        print(f"📊 Veri seti hazırlanıyor: {self.user_data_path}")
        metadata_path = self.user_data_path / "metadata_words.csv"
        df = pd.read_csv(metadata_path)

        def audio_loader(path):
            filename = os.path.basename(path)
            filepath = self.user_data_path / "words" / filename # Use self.user_data_path
            try:
                speech, sample_rate = librosa.load(filepath, sr=config.ORNEKLEME_ORANI)
                return speech
            except FileNotFoundError:
                print(f"⚠️  Uyarı: Ses dosyası bulunamadı, atlanıyor: {filepath}")
                return None
            except Exception as e:
                print(f"❌ Hata yüklenirken: {filepath} - {e}")
                return None

        df["audio"] = df["file_path"].apply(audio_loader)
        # Remove rows where audio loading failed
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

    def train_model(self, dataset):
        """Modeli, transformers.Trainer kullanmadan manuel bir PyTorch döngüsü ile eğitir."""
        print("🚀 Kişiselleştirilmiş model eğitimi başlıyor... (Manuel Döngü)")

        # 1. Veri Setini Hazırla
        # Not: num_proc > 1 kullanmak Windows'ta 'fork' metodu nedeniyle sorun yaratabilir. 
        # Sorun yaşarsanız bu değeri 1'e düşürün veya bu satırı kaldırın.
        try:
            num_cpus = os.cpu_count()
        except NotImplementedError:
            num_cpus = 1
        print(f"⚙️  Veri ön işleme {num_cpus} CPU çekirdeği ile paralelleştiriliyor...")
        
        processed_dataset = dataset.map(
            self.preprocess_function,
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=config.FINETUNE_BATCH_SIZE,
            num_proc=num_cpus
        )

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        dataloader = DataLoader(
            processed_dataset,
            batch_size=config.FINETUNE_BATCH_SIZE,
            collate_fn=data_collator
        )

        # 2. Optimizasyon ve Hızlandırıcı (Accelerator) Ayarları
        optimizer = AdamW(self.model.parameters(), lr=config.FINETUNE_LEARNING_RATE)
        
        accelerator = Accelerator(
            mixed_precision="fp16" if torch.cuda.is_available() else "no",
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS
        )
        
        self.model, optimizer, dataloader = accelerator.prepare(
            self.model, optimizer, dataloader
        )

        num_epochs = config.NUM_FINETUNE_EPOCHS
        num_training_steps = num_epochs * len(dataloader)
        progress_bar = tqdm(range(num_training_steps))

        # 3. Eğitim Döngüsü
        self.model.train()
        for epoch in range(num_epochs):
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    accelerator.backward(loss)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

        # 4. Modeli Kaydet
        print("\n✅ Model ince ayarı tamamlandı!")
        
        # Modeli unwrapping işlemi ve kaydetme
        unwrapped_model = accelerator.unwrap_model(self.model)
        save_model(unwrapped_model, self.processor, str(self.output_dir))

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