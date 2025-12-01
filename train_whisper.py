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
from src.utils.utils import save_model
import librosa
from datasets import Dataset, DatasetDict, Audio
import evaluate

def prepare_dataset(user_id):
    """Hazırlanmış train ve eval CSV dosyalarını yükler."""
    user_data_path = Path(config.BASE_PATH) / user_id
    print(f"📊 Veri setleri yükleniyor: {user_data_path}")
    
    train_path = user_data_path / "train.csv"
    eval_path = user_data_path / "eval.csv"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"'{train_path}' veya '{eval_path}' bulunamadı. Lütfen önce `prepare_training_data.py` betiğini çalıştırın.")

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    train_dataset = train_dataset.cast_column("file_path", Audio(decode=False))
    eval_dataset = eval_dataset.cast_column("file_path", Audio(decode=False))

    dataset = DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset
    })

    print(f"📈 Eğitim seti boyutu: {len(dataset['train'])} kayıt")
    print(f"📉 Değerlendirme seti boyutu: {len(dataset['eval'])} kayıt")
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Kullanıcıya özel Whisper modelini eğitir.")
    parser.add_argument("user_id", type=str, help="Verisi kullanılacak ve modeli kişiselleştirilecek kullanıcının kimliği.")
    parser.add_argument("--base_model", type=str, default="openai/whisper-small", help="İnce ayar için kullanılacak temel modelin yolu.")
    args = parser.parse_args()

    # 1. Load Processor and Model
    processor = WhisperProcessor.from_pretrained(args.base_model, language="tr", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # 2. Prepare Dataset
    dataset = prepare_dataset(args.user_id)

    def preprocess_function(examples):
        audio_arrays = [librosa.load(path['path'], sr=16000)[0] for path in examples["file_path"]]
        
        inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding="longest", truncation=True)
        
        # Manually pad the input features
        padded_input_features = torch.zeros(len(audio_arrays), 80, 3000)
        padded_input_features[:, :, :inputs.input_features.shape[2]] = inputs.input_features
        inputs['input_features'] = padded_input_features

        labels = processor.tokenizer(text=examples["transcript"], padding="longest", truncation=True).input_ids
        
        inputs["labels"] = labels
        return inputs

    processed_dataset = dataset.map(preprocess_function, remove_columns=dataset["train"].column_names, batched=True, batch_size=config.FINETUNE_BATCH_SIZE)

    # 3. Define Custom Data Collator
    def custom_collate_fn(features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

    # 4. Training
    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS
    )

    train_dataloader = DataLoader(processed_dataset["train"], shuffle=True, collate_fn=custom_collate_fn, batch_size=config.FINETUNE_BATCH_SIZE)
    eval_dataloader = DataLoader(processed_dataset["eval"], collate_fn=custom_collate_fn, batch_size=config.FINETUNE_BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=config.FINETUNE_LEARNING_RATE)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_epochs = config.NUM_FINETUNE_EPOCHS
    num_training_steps = num_epochs * len(train_dataloader)
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        model.eval()
        all_preds = []
        all_labels = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            with torch.no_grad():
                generated_tokens = (
                    accelerator.unwrap_model(model)
                    .generate(
                        input_features=batch["input_features"],
                        return_dict_in_generate=True,
                        max_length=225,
                    )
                    .sequences
                )
                labels = batch["labels"]
            
            generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=processor.tokenizer.pad_token_id)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=processor.tokenizer.pad_token_id)

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            labels[labels == -100] = processor.tokenizer.pad_token_id
            
            decoded_preds = processor.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
            
            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)

        wer = 100 * wer_metric.compute(predictions=all_preds, references=all_labels)
        cer = 100 * cer_metric.compute(predictions=all_preds, references=all_labels)
        print(f"Epoch {epoch+1} | WER: {wer:.2f} | CER: {cer:.2f}")


    # 5. Save the fine-tuned model
    output_dir = Path("data/models/personalized_models") / args.user_id
    unwrapped_model = accelerator.unwrap_model(model)
    save_model(unwrapped_model, processor, output_dir)

if __name__ == "__main__":
    main()
