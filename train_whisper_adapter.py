# train_whisper_adapter.py
import os
import sys
import argparse
import torch
import dataclasses
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, Audio
import config
import librosa
import evaluate
import platform
import time
from src.services.reporting_service import ReportingService

try:
    import audiomentations as A
    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False

def build_augment_pipeline(sampling_rate: int):
    if not AUDIOMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
        A.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.3),
        A.PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
    ], p=0.5)

# Logging configuration
def setup_logging(user_id):
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_whisper_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: any

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

class WhisperPersonalizedTrainer:
    def __init__(self, user_id, base_model_path=None):
        self.user_id = user_id
        self.base_model_path = base_model_path or "openai/whisper-medium"
        self.user_data_path = Path(config.BASE_PATH) / self.user_id
        self.output_dir = Path("data/models/personalized_models") / self.user_id
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        self.augmenter = build_augment_pipeline(16000) if config.USE_AUGMENTATION else None

    def load_model_and_processor(self):
        print(f"📥 Whisper model yükleniyor: {self.base_model_path}")
        self.processor = WhisperProcessor.from_pretrained(self.base_model_path, language="tr", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(self.base_model_path)
        
        if self.device == "cuda":
            # Optional: k-bit training if needed
            # self.model = prepare_model_for_kbit_training(self.model)
            pass

        peft_config = LoraConfig(
            r=config.ADAPTER_REDUCTION_FACTOR,
            lora_alpha=config.ADAPTER_REDUCTION_FACTOR * 2,
            target_modules=["q_proj", "v_proj", "k_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        
        if config.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            self.model.config.use_cache = False  # Whisper için eğitimde zorunlu
            
        self.model.to(self.device)
        print(f"✅ Whisper model LoRA ile yüklendi. Cihaz: {self.device}")

    def prepare_dataset(self, csv_path, is_train=False):
        if not os.path.exists(csv_path):
            return None
            
        df = pd.read_csv(csv_path)
        
        def preprocess(batch):
            try:
                audio, sr = librosa.load(batch["file_path"], sr=16000)
                # Apply augmentation if training
                if is_train and self.augmenter:
                    audio = self.augmenter(samples=audio, sample_rate=16000)
                
                batch["input_features"] = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
                batch["labels"] = self.processor.tokenizer(batch["transcript"]).input_ids
                return batch
            except Exception as e:
                return None

        ds = Dataset.from_pandas(df)
        ds = ds.map(preprocess, remove_columns=ds.column_names).filter(lambda x: x is not None)
        return ds

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * self.cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    def train(self):
        setup_logging(self.user_id)
        self.base_model_name = self.base_model_path
        self.load_model_and_processor()
        
        train_ds = self.prepare_dataset(self.user_data_path / "train.csv", is_train=True)
        eval_ds = self.prepare_dataset(self.user_data_path / "eval.csv", is_train=False)
        
        if not train_ds:
            print("❌ Eğitim verisi bulunamadı!")
            return

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
            per_device_eval_batch_size=config.FINETUNE_BATCH_SIZE, # Eval için de düşük batch size
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=config.FINETUNE_LEARNING_RATE,
            warmup_steps=config.WARMUP_STEPS,
            num_train_epochs=config.NUM_FINETUNE_EPOCHS,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            fp16=(config.MIXED_PRECISION == "fp16"),
            bf16=(config.MIXED_PRECISION == "bf16"),
            gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
            predict_with_generate=True,
            generation_max_length=225,
            report_to="none"
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        start_time = time.time()
        trainer.train()
        duration = time.time() - start_time

        # Save final model
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        
        # Report
        metrics = trainer.evaluate()
        reporting_service = ReportingService()
        training_data = {
            "base_model": self.base_model_name,
            "num_epochs": config.NUM_FINETUNE_EPOCHS,
            "final_wer": metrics.get("eval_wer"),
            "final_cer": metrics.get("eval_cer"),
            "duration_seconds": duration,
            "duration_formatted": f"{duration/60:.2f} min"
        }
        reporting_service.log_training_session(self.user_id, training_data)
        print(f"✅ Eğitim tamamlandı ve model {self.output_dir} konumuna kaydedildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id", type=str)
    parser.add_argument("--base_model", type=str, default="openai/whisper-medium")
    args = parser.parse_args()
    
    trainer = WhisperPersonalizedTrainer(args.user_id, args.base_model)
    trainer.train()
