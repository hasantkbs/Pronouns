# train_adapter.py

import os
import argparse
import torch
import pandas as pd
from pathlib import Path
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm
from datasets import Dataset, Audio
from src.utils.utils import save_model_and_processor
import config
import librosa

class DataCollatorCTCWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def _standalone_preprocess_function(examples, processor):
    """Standalone data preprocessing function for multiprocessing."""
    audio_arrays = [librosa.load(path_dict['path'], sr=config.ORNEKLEME_ORANI)[0] for path_dict in examples["file_path"]]
    
    inputs = processor(audio_arrays, sampling_rate=config.ORNEKLEME_ORANI, return_tensors="pt", padding=True, truncation=True, max_length=96000)
    
    with processor.as_target_processor():
        labels = processor(examples["transcription"], return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids

    inputs["labels"] = labels
    return inputs

class PersonalizedTrainer:
    def __init__(self, user_id, base_model_path=None):
        self.user_id = user_id
        self.base_model_path = base_model_path or config.MODEL_NAME
        self.user_data_path = Path(config.BASE_PATH) / self.user_id
        self.output_dir = Path("data/models/personalized_models") / self.user_id
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None

    def run(self):
        print(f"🎯 {self.user_id} için kişiselleştirme süreci başlıyor.")
        print("="*50)
        
        if not self.user_data_path.exists() or not (self.user_data_path / "metadata_words.csv").exists():
            print(f"❌ Hata: {self.user_data_path} için veri bulunamadı.")
            return

        self.load_model_and_processor()
        dataset = self.prepare_dataset()
        self.train_model(dataset)

    def load_model_and_processor(self):
        print(f"📥 Temel model yükleniyor: {self.base_model_path}")
        self.processor = Wav2Vec2Processor.from_pretrained(self.base_model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.base_model_path)
        
        self.model.to(self.device)
        peft_config = LoraConfig(
            r=config.ADAPTER_REDUCTION_FACTOR,
            lora_alpha=config.ADAPTER_REDUCTION_FACTOR * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        print(f"✅ Model PEFT/LoRA ile sarmalandı. Cihaz: {self.device}")

    def prepare_dataset(self):
        print(f"📊 Veri seti hazırlanıyor: {self.user_data_path}")
        metadata_path = self.user_data_path / "metadata_words.csv"
        df = pd.read_csv(metadata_path)

        df["file_path"] = df["file_path"].apply(
            lambda x: str(self.user_data_path / "words" / os.path.basename(x))
        )
        
        original_size = len(df)
        df = df[df["file_path"].apply(os.path.exists)]
        if len(df) < original_size:
            print(f"⚠️  {original_size - len(df)} adet bulunamayan ses dosyası atlandı.")

        dataset = Dataset.from_pandas(df)
        dataset = dataset.cast_column("file_path", Audio(sampling_rate=config.ORNEKLEME_ORANI, decode=False))
        
        print(f"📈 Veri seti boyutu: {len(dataset)} kayıt")
        return dataset

    def train_model(self, dataset):
        print("🚀 Kişiselleştirilmiş model eğitimi başlıyor... (Manuel Döngü)")

        num_proc = 4 
        print(f"⚙️  Veri ön işleme {num_proc} CPU çekirdeği ile paralelleştiriliyor...")
        
        processed_dataset = dataset.map(
            _standalone_preprocess_function,
            fn_kwargs={"processor": self.processor},
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=config.FINETUNE_BATCH_SIZE,
            num_proc=num_proc
        )

        data_collator = DataCollatorCTCWithPadding(processor=self.processor)

        dataloader = DataLoader(
            processed_dataset,
            batch_size=config.FINETUNE_BATCH_SIZE,
            collate_fn=data_collator
        )

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

        self.model.train()
        for epoch in range(num_epochs):
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    accelerator.backward(loss)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

        print("\n✅ Model ince ayarı tamamlandı!")
        
        unwrapped_model = accelerator.unwrap_model(self.model)
        save_model_and_processor(unwrapped_model, self.processor, str(self.output_dir))

        print(f"💾 Kişiselleştirilmiş model kaydedildi: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Kullanıcıya özel ASR modelini eğitir.")
    parser.add_argument("user_id", type=str, help="Verisi kullanılacak ve modeli kişiselleştirilecek kullanıcının kimliği.")
    parser.add_argument("--base_model", type=str, help="İnce ayar için kullanılacak temel modelin yolu. Varsayılan: config.py'deki model.", default=None)
    
    args = parser.parse_args()
    
    trainer = PersonalizedTrainer(user_id=args.user_id, base_model_path=args.base_model)
    trainer.run()

if __name__ == "__main__":
    if torch.cuda.is_available():
        import multiprocess as mp
        mp.set_start_method("spawn", force=True)

    main()