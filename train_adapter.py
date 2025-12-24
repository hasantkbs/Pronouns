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
    """
    CTC tabanlı ASR için data collator.
    Wav2Vec2 için input_values ve labels ayrı ayrı pad edilir.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # input_values için padding
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        
        # labels için padding
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )
        
        # Padding token'ları -100'e çevir (CTC loss için)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), 
            -100
        )
        batch["labels"] = labels
        return batch

def _standalone_preprocess_function(examples, processor):
    """
    Standalone data preprocessing function for multiprocessing.
    Wav2Vec2 için özellik çıkarımı ve tokenization yapar.
    """
    import numpy as np
    
    # Ses dosyalarını yükle
    audio_arrays = []
    valid_transcripts = []
    
    # Transcript sütununu belirle
    transcript_key = "transcript" if "transcript" in examples else "transcription"
    transcripts = examples.get(transcript_key, [""] * len(examples["file_path"]))
    
    for i, path_dict in enumerate(examples["file_path"]):
        try:
            audio, sr = librosa.load(path_dict['path'], sr=config.ORNEKLEME_ORANI)
            if len(audio) > 100:  # En az 100 sample (çok kısa kayıtları filtrele)
                # Transcript kontrolü
                transcript = str(transcripts[i]).strip() if i < len(transcripts) else ""
                if transcript:
                    audio_arrays.append(audio)
                    valid_transcripts.append(transcript)
        except Exception as e:
            # Hata durumunda sessizce atla (loglama çok fazla olabilir)
            continue
    
    if len(audio_arrays) == 0:
        # Boş batch için dummy değerler döndür
        return {
            "input_values": np.array([0.0]),
            "labels": [processor.tokenizer.pad_token_id]
        }
    
    # Processor ile özellik çıkarımı
    inputs = processor(
        audio_arrays, 
        sampling_rate=config.ORNEKLEME_ORANI, 
        return_tensors="pt", 
        padding=True
    )
    
    # Transkriptleri tokenize et
    labels = processor.tokenizer(
        valid_transcripts, 
        return_tensors="pt", 
        padding=True
    ).input_ids
    
    # Sonuçları dict olarak döndür (collator için)
    result = {
        "input_values": inputs.input_values,
        "labels": labels
    }
    
    return result

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
        """Veri setini hazırlar ve yükler."""
        print(f"📊 Veri seti hazırlanıyor: {self.user_data_path}")
        
        # Önce train.csv ve eval.csv dosyalarını kontrol et
        train_csv = self.user_data_path / "train.csv"
        eval_csv = self.user_data_path / "eval.csv"
        
        if train_csv.exists():
            print(f"   ✅ train.csv bulundu, kullanılıyor.")
            df = pd.read_csv(train_csv, encoding='utf-8')
        else:
            # metadata_words.csv'den oluştur
            metadata_path = self.user_data_path / "metadata_words.csv"
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"❌ Hata: Ne train.csv ne de metadata_words.csv bulunamadı!\n"
                    f"   Lütfen önce 'python prepare_training_data.py {self.user_id}' çalıştırın."
                )
            
            print(f"   ⚠️  train.csv bulunamadı, metadata_words.csv kullanılıyor.")
            df = pd.read_csv(metadata_path, encoding='utf-8')
            df = df[['file_path', 'transcription']].copy()
            df.rename(columns={'transcription': 'transcript'}, inplace=True)

        # Dosya yollarını düzelt
        words_dir = self.user_data_path / "words"
        def fix_file_path(path):
            filename = os.path.basename(str(path))
            return str(words_dir / filename)
        
        df["file_path"] = df["file_path"].apply(fix_file_path)
        
        # Var olmayan dosyaları filtrele
        original_size = len(df)
        df = df[df["file_path"].apply(os.path.exists)]
        if len(df) < original_size:
            print(f"   ⚠️  {original_size - len(df)} adet bulunamayan ses dosyası atlandı.")
        
        if len(df) == 0:
            raise ValueError(f"❌ Hata: Hiç geçerli ses dosyası bulunamadı!")
        
        # Boş transkriptleri filtrele
        df = df[df['transcript'].notna() & (df['transcript'].str.strip() != '')]
        
        dataset = Dataset.from_pandas(df)
        dataset = dataset.cast_column("file_path", Audio(sampling_rate=config.ORNEKLEME_ORANI, decode=False))
        
        print(f"   📈 Veri seti boyutu: {len(dataset)} kayıt")
        return dataset

    def train_model(self, dataset):
        """Model eğitimini başlatır."""
        print("🚀 Kişiselleştirilmiş model eğitimi başlıyor...")
        print(f"   Epoch sayısı: {config.NUM_FINETUNE_EPOCHS}")
        print(f"   Batch size: {config.FINETUNE_BATCH_SIZE}")
        print(f"   Learning rate: {config.FINETUNE_LEARNING_RATE}")
        print(f"   Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")

        # Veri ön işleme
        num_proc = min(4, os.cpu_count() or 1)
        print(f"\n⚙️  Veri ön işleme {num_proc} CPU çekirdeği ile paralelleştiriliyor...")
        
        try:
            processed_dataset = dataset.map(
                _standalone_preprocess_function,
                fn_kwargs={"processor": self.processor},
                remove_columns=dataset.column_names,
                batched=True,
                batch_size=config.FINETUNE_BATCH_SIZE,
                num_proc=num_proc
            )
            
            # Boş örnekleri filtrele
            processed_dataset = processed_dataset.filter(
                lambda x: len(x.get("input_values", [])) > 0 and len(x.get("labels", [])) > 0
            )
            
            if len(processed_dataset) == 0:
                raise ValueError("❌ Hata: Ön işleme sonrası hiç geçerli örnek kalmadı!")
            
            print(f"   ✅ Ön işleme tamamlandı. {len(processed_dataset)} geçerli örnek.")
            
        except Exception as e:
            print(f"❌ Veri ön işleme hatası: {e}")
            import traceback
            traceback.print_exc()
            return

        # Data collator ve dataloader
        data_collator = DataCollatorCTCWithPadding(processor=self.processor)

        dataloader = DataLoader(
            processed_dataset,
            batch_size=config.FINETUNE_BATCH_SIZE,
            collate_fn=data_collator,
            shuffle=True
        )

        # Optimizer
        optimizer = AdamW(
            self.model.parameters(), 
            lr=config.FINETUNE_LEARNING_RATE,
            weight_decay=5e-3
        )
        
        # Accelerator (GPU desteği ve gradient accumulation için)
        accelerator = Accelerator(
            mixed_precision="fp16" if torch.cuda.is_available() else "no",
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS
        )
        
        self.model, optimizer, dataloader = accelerator.prepare(
            self.model, optimizer, dataloader
        )

        # Eğitim döngüsü
        num_epochs = config.NUM_FINETUNE_EPOCHS
        num_training_steps = num_epochs * len(dataloader)
        progress_bar = tqdm(range(num_training_steps), desc="Eğitim")

        self.model.train()
        total_loss = 0.0
        
        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for step, batch in enumerate(dataloader):
                    with accelerator.accumulate(self.model):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    total_loss += loss.item()
                    
                    progress_bar.update(1)
                    avg_loss = epoch_loss / num_batches
                    progress_bar.set_description(
                        f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}"
                    )
                
                print(f"\n   Epoch {epoch+1}/{num_epochs} tamamlandı. Ortalama Loss: {epoch_loss/num_batches:.4f}")

            print("\n✅ Model ince ayarı tamamlandı!")
            
            # Model kaydetme
            unwrapped_model = accelerator.unwrap_model(self.model)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_model_and_processor(unwrapped_model, self.processor, str(self.output_dir))

            print(f"💾 Kişiselleştirilmiş model kaydedildi: {self.output_dir}")
            print(f"   Toplam eğitim adımı: {num_training_steps}")
            print(f"   Ortalama loss: {total_loss / num_training_steps:.4f}")
            
        except Exception as e:
            print(f"\n❌ Eğitim sırasında hata oluştu: {e}")
            import traceback
            traceback.print_exc()
            raise

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