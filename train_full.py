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
    DataCollatorForSpeechSeq2Seq,
)
from datasets import Dataset, DatasetDict, Audio
import evaluate
import config
import librosa

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
    
    # Otomatik decode'u engelle, sadece yolları yükle
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
    parser.add_argument("--base_model", type=str, default="openai/whisper-base", help="İnce ayar için kullanılacak temel modelin yolu.")
    args = parser.parse_args()

    # 1. Load Processor and Model
    processor = WhisperProcessor.from_pretrained(args.base_model, language="tr", task="transcribe")
    
    # Performans Optimizasyonu: Mümkünse Flash Attention 2'yi etkinleştir
    try:
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model, attn_implementation="flash_attention_2")
        print("⚡️ Model, Flash Attention 2 optimizasyonu ile yükleniyor.")
    except (ImportError, ValueError):
        print("⚠️ Flash Attention 2 kullanılamıyor. Standart dikkat mekanizması ile devam ediliyor.")
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
        
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # 2. Prepare Dataset
    dataset = prepare_dataset(args.user_id)

    def preprocess_function(examples):
        # Ses dosyalarını librosa ile manuel olarak yükle
        audio_arrays = [librosa.load(path['path'], sr=16000)[0] for path in examples["file_path"]]
        
        # Girişleri işle
        inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding="longest", truncation=True)
        
        # Hedef metinleri işle
        labels = processor.tokenizer(text=examples["transcript"], padding="longest", truncation=True).input_ids
        
        inputs["labels"] = labels
        return inputs

    # Not: num_proc > 1 kullanmak Windows'ta 'fork' metodu nedeniyle sorun yaratabilir. 
    # Sorun yaşarsanız bu değeri 1'e düşürün veya bu satırı kaldırın.
    try:
        num_cpus = os.cpu_count()
    except NotImplementedError:
        num_cpus = 1
    print(f"⚙️  Veri ön işleme {num_cpus} CPU çekirdeği ile paralelleştiriliyor...")
    
    processed_dataset = dataset.map(
        preprocess_function, 
        remove_columns=dataset["train"].column_names, 
        batched=True, 
        batch_size=config.FINETUNE_BATCH_SIZE, # Batch size'ı düşürerek bellek sorunlarını azalt
        num_proc=num_cpus
    )

    # 3. Define Data Collator
    data_collator = DataCollatorForSpeechSeq2Seq(processor=processor, model=model)

    # 4. Define Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{args.user_id}_whisper_finetuned",
        per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
        num_train_epochs=config.NUM_FINETUNE_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=True,
        gradient_checkpointing=True, # VRAM optimizasyonu için eklendi
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
    )

    # 5. Define Evaluation Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    # 6. Initialize and Train the Model
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['eval'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    # 7. Save the fine-tuned model
    output_dir = Path("data/models/personalized_models") / args.user_id
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"🎉 Model başarıyla '{output_dir}' dizinine kaydedildi.")


if __name__ == "__main__":
    main()
