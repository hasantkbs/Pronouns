import os
import argparse
import torch
import pandas as pd
from pathlib import Path
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm
import config
import librosa
from datasets import Dataset

def prepare_dataset(user_id):
    """Kullanıcıya özel veri setini hazırlar."""
    user_data_path = Path(config.BASE_PATH) / user_id
    print(f"📊 Veri seti hazırlanıyor: {user_data_path}")
    metadata_path = user_data_path / "metadata_words.csv"
    df = pd.read_csv(metadata_path)

    def audio_loader(path):
        filename = os.path.basename(path)
        filepath = user_data_path / "words" / filename
        try:
            speech, sample_rate = librosa.load(filepath, sr=16000)
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

def main():
    parser = argparse.ArgumentParser(description="Kullanıcıya özel Whisper modelini eğitir.")
    parser.add_argument("user_id", type=str, help="Verisi kullanılacak ve modeli kişiselleştirilecek kullanıcının kimliği.")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v2", help="İnce ayar için kullanılacak temel modelin yolu.")
    args = parser.parse_args()

    # 1. Load Processor and Model
    processor = WhisperProcessor.from_pretrained(args.base_model, language="tr", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    
    # 2. Prepare Dataset
    dataset = prepare_dataset(args.user_id)

    def preprocess_function(examples):
        audio_arrays = [x for x in examples["audio"]]
        
        inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
        
        labels = processor.tokenizer(text=examples["transcription"], padding=True, truncation=True).input_ids
        
        inputs["labels"] = labels
        return inputs

    processed_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names, batched=True, batch_size=config.FINETUNE_BATCH_SIZE)

    # 3. Define Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, model=model)

    # 4. Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.FINETUNE_LEARNING_RATE)

    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS
    )

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, DataLoader(processed_dataset, batch_size=config.FINETUNE_BATCH_SIZE, collate_fn=data_collator)
    )

    num_epochs = config.NUM_FINETUNE_EPOCHS
    num_training_steps = num_epochs * len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

    # 5. Save the fine-tuned model
    output_dir = Path("data/models/personalized_models") / args.user_id
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"💾 Kişiselleştirilmiş model kaydedildi: {output_dir}")

if __name__ == "__main__":
    main()
