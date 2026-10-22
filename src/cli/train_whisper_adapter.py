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
import json
import logging
import platform
import random
import time
from datetime import datetime

import librosa
import pandas as pd
import soundfile as sf
import torch
import evaluate
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import config
from src.services.reporting_service import ReportingService
from src.training import sentence_synth as ss
from src.utils.utils import (
    build_augment_pipeline,
    normalize_turkish_text,
    resolve_audio_path,
    set_seed,
)


PERSONALIZED_ROOT = Path("data/models/personalized_models")
TRAIN_SENTENCES = Path("datasets") / "sentence_sets" / "sentetik_egitim.txt"


def resolve_output_dir(user_id, output_name=None, init_from=None) -> Path:
    """
    Çıktı klasörü. Varsayılan canlı Whisper klasörü (self-learning bugünkü gibi).
    Devam eğitimi (--init-from) canlı klasöre ya da başladığı klasöre asla yazmaz.
    """
    live = PERSONALIZED_ROOT / f"{user_id}_whisper"
    if output_name is None:
        if init_from is not None:
            raise ValueError("--init-from ile --output-name verilmeli (canlı klasöre yazılmaz)")
        return live
    out = PERSONALIZED_ROOT / output_name
    if out.resolve() == live.resolve():
        raise ValueError(f"--output-name canlı klasör olamaz: {live}")
    if init_from is not None and out.resolve() == Path(init_from).resolve():
        raise ValueError("--output-name, --init-from ile aynı klasör olamaz")
    return out


def training_schedule(continuing: bool):
    """(en fazla tur, öğrenme hızı, erken durdurma sabrı)."""
    if continuing:
        return config.CONTINUE_EPOCHS, config.CONTINUE_LEARNING_RATE, config.CONTINUE_EARLY_STOPPING_PATIENCE
    return config.NUM_FINETUNE_EPOCHS, config.FINETUNE_LEARNING_RATE, config.EARLY_STOPPING_PATIENCE


def build_train_frame(words_df, sentences, takes_by_word, repeats=None):
    """
    Kelime satırları + her kullanılabilir cümleden `repeats` satır. Cümle
    satırının file_path'i boştur; sesi transform sırasında üretilir.
    Döner: (DataFrame[file_path, transcript, kind], atlanan cümleler)
    """
    repeats = config.SYNTH_SENTENCE_REPEATS if repeats is None else repeats
    words = words_df[["file_path", "transcript"]].copy()
    words["kind"] = "word"
    usable, skipped = ss.usable_sentences(sentences, takes_by_word)
    sentence_rows = pd.DataFrame(
        [{"file_path": "", "transcript": s, "kind": "sentence"} for s in usable for _ in range(repeats)],
        columns=["file_path", "transcript", "kind"],
    )
    return pd.concat([words, sentence_rows], ignore_index=True), skipped


def build_eval_frame(words_df, manifest_rows, user_dir):
    """eval.csv kelimeleri + sabit sentetik ölçüm sesleri (varsa)."""
    words = words_df[["file_path", "transcript"]].copy()
    words["kind"] = "word"
    synthetic = pd.DataFrame(
        [{"file_path": str(Path(user_dir) / r["file_path"]), "transcript": r["transcription"], "kind": "word"}
         for r in manifest_rows],
        columns=["file_path", "transcript", "kind"],
    )
    return pd.concat([words, synthetic], ignore_index=True)


def eval_manifest(user_dir):
    """Tur sonu ölçümüne girecek sentetik sesler; bugünkü eval.csv'den üretilmemişse []."""
    if not ss.EVAL_SENTENCES.exists():
        return []
    return ss.current_manifest(user_dir, ss.EVAL_SENTENCES)


def eval_metrics(pred_str, label_str, n_words, wer, cer):
    """
    Birleşik wer/cer (en iyi tur seçimi) + kelime (ilk n_words satır) ve
    sentetik cümle (geri kalan) ayrı ayrı. Ölçüm sırası korunur: build_eval_frame
    önce kelimeleri, sonra sentetik sesleri koyar; değerlendirme karıştırılmaz.
    """
    def score(preds, refs):
        idx = [i for i, r in enumerate(refs) if r.strip()]
        if not idx:
            return None
        p, r = [preds[i] for i in idx], [refs[i] for i in idx]
        return wer(predictions=p, references=r), cer(predictions=p, references=r)

    combined = score(pred_str, label_str)
    if combined is None:
        return {"wer": 1.0, "cer": 1.0}
    out = {"wer": combined[0], "cer": combined[1]}
    for prefix, preds, refs in (
        ("word_", pred_str[:n_words], label_str[:n_words]),
        ("sentence_", pred_str[n_words:], label_str[n_words:]),
    ):
        part = score(preds, refs)
        if part is not None:
            out[f"{prefix}wer"], out[f"{prefix}cer"] = part
    return out


def report_wers(final_metrics):
    """Rapordaki final WER kelime ölçümüdür (Ayarlar'daki model kartı bunu gösterir)."""
    return {
        "final_wer": final_metrics.get("eval_word_wer", final_metrics.get("eval_wer")),
        "final_cer": final_metrics.get("eval_word_cer", final_metrics.get("eval_cer")),
        "synthetic_sentence_wer": final_metrics.get("eval_sentence_wer"),
        "synthetic_sentence_cer": final_metrics.get("eval_sentence_cer"),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Kullanıcıya özel Whisper ASR modelini LoRA ile eğitir.")
    parser.add_argument("user_id", type=str, help="Verisi kullanılacak ve modeli kişiselleştirilecek kullanıcının kimliği.")
    parser.add_argument(
        "--base_model", type=str, default=None,
        help="İnce ayar için kullanılacak temel Whisper modelinin yolu. Varsayılan: config.WHISPER_MODEL_NAME.",
    )
    parser.add_argument("--init-from", type=str, default=None,
                        help="LoRA ağırlıklarının yükleneceği adaptör klasörü (devam eğitimi).")
    parser.add_argument("--output-name", type=str, default=None,
                        help="data/models/personalized_models altında çıktı klasörünün adı.")
    parser.add_argument("--no-sentences", action="store_true",
                        help="Sentetik cümleleri eğitime ekleme.")
    return parser.parse_args(argv)


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
    # Whisper'da tokenizer.bos_token_id (<|endoftext|>) != decoder_start_token_id
    # (<|startoftranscript|>); model bu token'i kendisi ekledigi icin etiketten silinmeli.
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class WhisperPersonalizedTrainer:
    def __init__(self, user_id, base_model_path=None, init_from=None, output_name=None, use_sentences=True):
        self.user_id = user_id
        self.base_model_path = base_model_path or config.WHISPER_MODEL_NAME
        self.user_data_path = Path(config.BASE_PATH) / self.user_id
        self.init_from = Path(init_from) if init_from else None
        self.output_name = output_name
        # Varsayilan _whisper son ekli canli klasor; canli Wav2Vec2 modeliyle asla ayni yolu paylasmaz.
        self.output_dir = resolve_output_dir(user_id, output_name, init_from)
        self.use_sentences = use_sentences
        self.num_epochs, self.learning_rate, self.patience = training_schedule(self.init_from is not None)
        self.train_takes = {}
        self.sentence_train_samples = 0
        self.synthetic_eval_samples = 0
        self.eval_word_count = 0

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
        # Eval sirasindaki generate() dil algilamaya kalkmasin; asr.py ile ayni ayar
        self.model.generation_config.language = "tr"
        self.model.generation_config.task = "transcribe"

        if self.init_from is not None:
            # Devam egitimi: canli adaptorun LoRA agirliklariyla baslar
            print(f"📥 LoRA adaptörü yükleniyor: {self.init_from}")
            self.model = PeftModel.from_pretrained(self.model, str(self.init_from), is_trainable=True)
        else:
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

        # Okunamayan (bozuk) ses dosyalarini egitimden once ele - sf.info sadece header okur.
        def is_readable(path):
            try:
                sf.info(path)
                return True
            except Exception as e:
                print(f"   ⚠️  Atlanan dosya ({path}): {e}")
                return False

        df = df[df["file_path"].apply(is_readable)]

        is_train = split == 'train'
        if is_train:
            df = self._with_sentences(df)
        else:
            manifest = eval_manifest(self.user_data_path)
            self.eval_word_count = len(df)
            df = build_eval_frame(df, manifest, self.user_data_path)
            self.synthetic_eval_samples = len(manifest)
            if manifest:
                print(f"   🗣️  Sentetik ölçüm cümlesi: {len(manifest)}")
        augmenter = self.augmenter if is_train else None
        sample_rate = config.ORNEKLEME_ORANI

        # Mel ozellikleri on-the-fly hesaplanir: tum veri setini onceden map() ile
        # hesaplamak ~1.5 MB/ornek RAM tutar ve 15 GB'lik sunucuda OOM'a yol acar.
        # Bonus: augmentation her epoch'ta yeniden uygulanir.
        train_takes = self.train_takes

        def transform(batch):
            input_features, labels = [], []
            # DataLoader her isciye ayri random tohumu verir; cumleler isciler arasi farkli uretilir
            rng = random.Random(random.random())
            for path, transcript, kind in zip(batch["file_path"], batch["transcript"], batch["kind"]):
                if kind == "sentence":
                    audio = ss.build_sentence(ss.sentence_words(transcript), train_takes, rng, sr=sample_rate)
                else:
                    audio, _ = librosa.load(path, sr=sample_rate)
                if augmenter:
                    audio = augmenter(samples=audio, sample_rate=sample_rate)
                input_features.append(
                    self.processor.feature_extractor(audio, sampling_rate=sample_rate).input_features[0]
                )
                labels.append(self.processor.tokenizer(normalize_turkish_text(transcript)).input_ids)
            return {"input_features": input_features, "labels": labels}

        dataset = Dataset.from_pandas(df[["file_path", "transcript", "kind"]], preserve_index=False)
        dataset.set_transform(transform)

        print(f"   📈 {split.upper()} veri seti boyutu: {len(dataset)} kayıt")
        return dataset

    def _with_sentences(self, words_df):
        """Eğitim kelimelerine sentetik cümleleri ekler (train.csv ve cümle dosyası varsa)."""
        train_csv = self.user_data_path / "train.csv"
        if not (self.use_sentences and TRAIN_SENTENCES.exists() and train_csv.exists()):
            words = words_df[["file_path", "transcript"]].copy()
            words["kind"] = "word"
            return words
        self.train_takes = ss.load_takes(train_csv, self.user_data_path)
        df, skipped = build_train_frame(words_df, ss.read_sentences(TRAIN_SENTENCES), self.train_takes)
        self.sentence_train_samples = int((df["kind"] == "sentence").sum())
        print(f"   🗣️  Sentetik eğitim cümlesi satırı: {self.sentence_train_samples} (atlanan cümle: {len(skipped)})")
        return df

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalize_turkish_text(p) for p in pred_str]
        label_str = [normalize_turkish_text(r) for r in label_str]

        return eval_metrics(
            pred_str, label_str, self.eval_word_count,
            wer=self.wer_metric.compute, cer=self.cer_metric.compute,
        )

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

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )
        has_eval = eval_dataset is not None

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
            per_device_eval_batch_size=config.FINETUNE_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=self.learning_rate,
            warmup_steps=config.WARMUP_STEPS,
            num_train_epochs=self.num_epochs,
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
            # set_transform kolonlari (file_path, transcript) Trainer tarafindan silinmemeli
            remove_unused_columns=False,
            dataloader_num_workers=4,
            save_total_limit=3,
            report_to="none",
        )

        callbacks = []
        if has_eval:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.patience))

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
        wers = report_wers(final_metrics)
        final_wer = wers["final_wer"]
        final_cer = wers["final_cer"]

        train_losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        eval_losses = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]

        epochs_completed = round(trainer.state.epoch, 2) if trainer.state.epoch else 0
        # HF Trainer, early-stop tetiklendiginde kontrol bilgisini kolayca disari
        # vermiyor; NUM_FINETUNE_EPOCHS'a ulasmadan durmus olmasi early-stop'un
        # isaretidir (CTC trainer'daki acik early_stopped flag'inin esdegeri).
        early_stopped = has_eval and epochs_completed < self.num_epochs

        training_data = {
            "base_model": self.base_model_path,
            "num_epochs": self.num_epochs,
            "batch_size": config.FINETUNE_BATCH_SIZE,
            "learning_rate": self.learning_rate,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
            "adapter_reduction_factor": config.ADAPTER_REDUCTION_FACTOR,
            "warmup_steps": config.WARMUP_STEPS,
            "weight_decay": config.WEIGHT_DECAY,
            "early_stopping_patience": self.patience,
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
            "init_from": str(self.init_from) if self.init_from else None,
            "output_name": self.output_dir.name,
            "sentence_train_samples": self.sentence_train_samples,
            "synthetic_eval_samples": self.synthetic_eval_samples,
            "combined_eval_wer": final_metrics.get("eval_wer"),
            "synthetic_sentence_wer": wers["synthetic_sentence_wer"],
            "synthetic_sentence_cer": wers["synthetic_sentence_cer"],
        }
        if self.output_name is None:
            report_file = ReportingService().log_training_session(self.user_id, training_data)
        else:
            # Ayri klasore egitilen model canli degil: training_<user>_*.json'a yazilirsa
            # TrainingService.get_last_wer onu canli modelin WER'i sanar.
            report_file = write_side_report(training_data, self.output_dir.name)
        print(f"📊 Training report saved: {report_file}")
        logger.info(f"Whisper training completed for user: {self.user_id}")


def write_side_report(training_data, output_name, reports_dir=None, now=None) -> Path:
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now()
    out = reports_dir / f"whisper_side_{output_name}_{now.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(training_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    if platform.system() == "Linux":
        import multiprocessing as mp
        try:
            mp.set_start_method(config.MULTIPROCESSING_START_METHOD, force=True)
        except RuntimeError:
            pass

    args = parse_args()

    set_seed(config.SEED)

    trainer = WhisperPersonalizedTrainer(
        user_id=args.user_id,
        base_model_path=args.base_model,
        init_from=args.init_from,
        output_name=args.output_name,
        use_sentences=not args.no_sentences,
    )
    trainer.run()
