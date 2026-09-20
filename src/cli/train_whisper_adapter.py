# train_whisper_adapter.py
# Whisper tabanlı kişiselleştirilmiş model eğitim scripti

import os
import sys
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import argparse
import dataclasses
import logging
import platform
import time
from datetime import datetime

import librosa
import pandas as pd
import torch
import evaluate
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import config
from src.services.reporting_service import ReportingService
from src.utils.utils import (
    build_augment_pipeline,
    normalize_turkish_text,
    resolve_audio_path,
    set_seed,
)


def setup_logging(user_id):
    """Whisper eğitimi için log dosyası oluşturur (train_adapter.py'nin CTC logging'iyle aynı desen)."""
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_whisper_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    print(f"📝 Log dosyası: {log_file}")
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
        self.base_model_path = base_model_path or config.WHISPER_MODEL_NAME
        self.user_data_path = Path(config.BASE_PATH) / self.user_id
        # Her zaman _whisper son ekiyle - canli Wav2Vec2 modeliyle asla ayni yolu paylasmaz.
        self.output_dir = Path("data/models/personalized_models") / f"{self.user_id}_whisper"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        self.augmenter = build_augment_pipeline(config.ORNEKLEME_ORANI) if config.USE_AUGMENTATION else None
        self.train_samples = 0
        self.eval_samples = 0

    def load_model_and_processor(self):
        print(f"📥 Whisper model yükleniyor: {self.base_model_path}")
        self.processor = WhisperProcessor.from_pretrained(
            self.base_model_path, language="tr", task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(self.base_model_path)

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
            self.model.config.use_cache = False  # Whisper icin egitimde zorunlu

        self.model.to(self.device)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Whisper model LoRA ile yüklendi. Cihaz: {self.device}")
        print(
            f"   Eğitilebilir parametreler: {trainable_params:,} / {total_params:,} "
            f"({100*trainable_params/total_params:.2f}%)"
        )

    def _load_and_filter_rows(self, split):
        """CTC trainer'daki (train_adapter.py:prepare_dataset) ile ayni filtre mantigi."""
        if split == 'train':
            csv_path = self.user_data_path / "train.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, encoding='utf-8')
            else:
                metadata_path = self.user_data_path / "metadata_words.csv"
                if not metadata_path.exists():
                    raise FileNotFoundError(
                        f"❌ Hata: Ne train.csv ne de metadata_words.csv bulunamadı: {self.user_data_path}"
                    )
                df = pd.read_csv(metadata_path, encoding='utf-8')
                df = df[['file_path', 'transcription']].copy()
                df.rename(columns={'transcription': 'transcript'}, inplace=True)
        else:
            csv_path = self.user_data_path / "eval.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, encoding='utf-8')
            else:
                metadata_path = self.user_data_path / "metadata_words.csv"
                if not metadata_path.exists():
                    return None
                df = pd.read_csv(metadata_path, encoding='utf-8')
                df = df[['file_path', 'transcription']].copy()
                df.rename(columns={'transcription': 'transcript'}, inplace=True)
                df = df.tail(int(len(df) * 0.2))

        df["file_path"] = df["file_path"].apply(lambda p: resolve_audio_path(p, self.user_data_path))
        df = df[df["file_path"].notna()]
        df = df[df["file_path"].apply(lambda x: os.path.exists(str(x)))]

        if "quality_score" in df.columns:
            df = df[df["quality_score"] >= config.QUALITY_THRESHOLD]

        df = df[df['transcript'].notna() & (df['transcript'].str.strip() != '')]

        if len(df) == 0:
            if split == 'eval':
                return None
            raise ValueError("❌ Hata: Ön işleme sonrası hiç geçerli örnek kalmadı!")

        return df

    def prepare_dataset(self, split='train'):
        print(f"📊 {split.upper()} veri seti hazırlanıyor: {self.user_data_path}")
        df = self._load_and_filter_rows(split)
        if df is None:
            return None

        is_train = split == 'train'
        augmenter = self.augmenter if is_train else None
        sample_rate = config.ORNEKLEME_ORANI

        def preprocess(batch):
            try:
                audio, _ = librosa.load(batch["file_path"], sr=sample_rate)
                if augmenter:
                    audio = augmenter(samples=audio, sample_rate=sample_rate)
                batch["input_features"] = self.processor.feature_extractor(
                    audio, sampling_rate=sample_rate
                ).input_features[0]
                batch["labels"] = self.processor.tokenizer(
                    normalize_turkish_text(batch["transcript"])
                ).input_ids
            except Exception as e:
                print(f"   ⚠️  Atlanan dosya ({batch.get('file_path')}): {e}")
                batch["input_features"] = None
                batch["labels"] = None
            return batch

        # Ayni guvenlik deseni train_adapter.py'de bu oturumda duzeltildi: num_proc=1
        # HuggingFace datasets'te hala bir multiprocessing pool acar ve "OSError: got
        # end of file during message" hatasina yol acar. num_proc>1 disinda None kullan.
        num_proc = min(config.DATA_PREPROCESSING_NUM_PROC, os.cpu_count() or 1)
        map_num_proc = num_proc if num_proc > 1 else None

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(preprocess, remove_columns=dataset.column_names, num_proc=map_num_proc)
        dataset = dataset.filter(lambda x: x["input_features"] is not None)

        print(f"   📈 {split.upper()} veri seti boyutu: {len(dataset)} kayıt")
        return dataset

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalize_turkish_text(p) for p in pred_str]
        label_str = [normalize_turkish_text(r) for r in label_str]

        valid_indices = [i for i, r in enumerate(label_str) if r.strip()]
        if not valid_indices:
            return {"wer": 1.0, "cer": 1.0}

        final_preds = [pred_str[i] for i in valid_indices]
        final_refs = [label_str[i] for i in valid_indices]

        wer = self.wer_metric.compute(predictions=final_preds, references=final_refs)
        cer = self.cer_metric.compute(predictions=final_preds, references=final_refs)
        return {"wer": wer, "cer": cer}

    def run(self):
        setup_logging(self.user_id)
        logger = logging.getLogger(__name__)

        training_start_time = time.time()
        start_datetime = datetime.now()

        print(f"🎯 {self.user_id} için Whisper kişiselleştirme süreci başlıyor.")
        print("=" * 50)
        logger.info(f"Whisper training started for user: {self.user_id}")

        sys_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python_version": sys.version.split()[0],
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": (
                round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
                if torch.cuda.is_available() else None
            ),
        }
        logger.info(f"Platform: {sys_info['platform']} {sys_info['platform_release']}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {sys_info['gpu_name']} ({sys_info['gpu_memory_gb']:.2f} GB)")

        if not self.user_data_path.exists():
            logger.error(f"❌ Hata: {self.user_data_path} için veri bulunamadı.")
            return

        self.load_model_and_processor()

        train_dataset = self.prepare_dataset(split='train')
        eval_dataset = self.prepare_dataset(split='eval')

        self.train_samples = len(train_dataset) if train_dataset else 0
        self.eval_samples = len(eval_dataset) if eval_dataset else 0

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        has_eval = eval_dataset is not None

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
            per_device_eval_batch_size=config.FINETUNE_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=config.FINETUNE_LEARNING_RATE,
            warmup_steps=config.WARMUP_STEPS,
            num_train_epochs=config.NUM_FINETUNE_EPOCHS,
            eval_strategy="epoch" if has_eval else "no",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=has_eval,
            metric_for_best_model="wer" if has_eval else None,
            greater_is_better=False,
            fp16=(config.MIXED_PRECISION == "fp16"),
            bf16=(config.MIXED_PRECISION == "bf16"),
            gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
            predict_with_generate=True,
            generation_max_length=225,
            report_to="none",
        )

        callbacks = []
        if has_eval:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE))

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if has_eval else None,
            processing_class=self.processor.feature_extractor,
            callbacks=callbacks,
        )

        trainer.train()
        duration = time.time() - training_start_time

        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        print(f"✅ Model kaydedildi: {self.output_dir}")

        final_metrics = trainer.evaluate() if has_eval else {}

        # load_best_model_at_end=True zaten en iyi checkpoint'i geri yukledigi icin
        # bu evaluate() cagrisinin sonucu = en iyi sonuc; final ve best ayni deger.
        final_wer = final_metrics.get("eval_wer")
        final_cer = final_metrics.get("eval_cer")

        train_losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        eval_losses = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]

        epochs_completed = round(trainer.state.epoch, 2) if trainer.state.epoch else 0
        # HF Trainer, early-stop tetiklendiginde kontrol bilgisini kolayca disari
        # vermiyor; NUM_FINETUNE_EPOCHS'a ulasmadan durmus olmasi early-stop'un
        # isaretidir (CTC trainer'daki acik early_stopped flag'inin esdegeri).
        early_stopped = has_eval and epochs_completed < config.NUM_FINETUNE_EPOCHS

        reporting_service = ReportingService()
        training_data = {
            "base_model": self.base_model_path,
            "num_epochs": config.NUM_FINETUNE_EPOCHS,
            "batch_size": config.FINETUNE_BATCH_SIZE,
            "learning_rate": config.FINETUNE_LEARNING_RATE,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
            "adapter_reduction_factor": config.ADAPTER_REDUCTION_FACTOR,
            "warmup_steps": config.WARMUP_STEPS,
            "weight_decay": config.WEIGHT_DECAY,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "use_augmentation": config.USE_AUGMENTATION,
            "mixed_precision": config.MIXED_PRECISION,
            "train_samples": self.train_samples,
            "eval_samples": self.eval_samples,
            "total_samples": self.train_samples + self.eval_samples,
            "total_steps": trainer.state.global_step,
            "epochs_completed": epochs_completed,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "best_val_loss": min(eval_losses) if eval_losses else None,
            "final_wer": final_wer,
            "final_cer": final_cer,
            "best_wer": final_wer,
            "best_cer": final_cer,
            "early_stopped": early_stopped,
            "device": self.device,
            "gpu_name": sys_info.get("gpu_name"),
            "gpu_memory_gb": sys_info.get("gpu_memory_gb"),
            "platform": sys_info.get("platform"),
            "python_version": sys_info.get("python_version"),
            "start_time": start_datetime.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": duration,
            "duration_formatted": f"{duration/60:.2f} dk",
        }
        report_file = reporting_service.log_training_session(self.user_id, training_data)
        print(f"📊 Training report saved: {report_file}")
        logger.info(f"Whisper training completed for user: {self.user_id}")


if __name__ == "__main__":
    if platform.system() == "Linux":
        import multiprocessing as mp
        try:
            mp.set_start_method(config.MULTIPROCESSING_START_METHOD, force=True)
        except RuntimeError:
            pass

    parser = argparse.ArgumentParser(description="Kullanıcıya özel Whisper ASR modelini LoRA ile eğitir.")
    parser.add_argument("user_id", type=str, help="Verisi kullanılacak ve modeli kişiselleştirilecek kullanıcının kimliği.")
    parser.add_argument(
        "--base_model", type=str, default=None,
        help="İnce ayar için kullanılacak temel Whisper modelinin yolu. Varsayılan: config.WHISPER_MODEL_NAME.",
    )
    args = parser.parse_args()

    set_seed(config.SEED)

    trainer = WhisperPersonalizedTrainer(user_id=args.user_id, base_model_path=args.base_model)
    trainer.run()
