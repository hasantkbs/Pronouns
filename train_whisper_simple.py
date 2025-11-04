import os
import argparse
import torch
import pandas as pd
from pathlib import Path
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, Audio
import evaluate
import config
import librosa

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
            return {"array": speech, "path": str(filepath)}
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
        audio = [x["array"] for x in examples["audio"]]
        
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
        
        labels = processor.tokenizer(text=examples["transcription"], padding=True, truncation=True).input_ids
        
        inputs["labels"] = labels
        return inputs

    processed_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names, batched=True, batch_size=config.FINETUNE_BATCH_SIZE)

    # 3. Define Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, model=model)

    # 4. Define Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{args.user_id}_whisper_finetuned",
        per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.FINETUNE_LEARNING_RATE,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=config.FINETUNE_BATCH_SIZE,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    # 5. Define Evaluation Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    # 6. Initialize and Train the Model
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset,
        eval_dataset=processed_dataset, # Using the same dataset for evaluation for now
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    # 7. Save the fine-tuned model
    output_dir = Path("data/models/personalized_models") / args.user_id
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
