# Whisper+LoRA Personalized Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, fully independent personalization path for FurkanV1's ASR — Whisper (encoder-decoder) fine-tuned with LoRA, starting from a Turkish-tuned checkpoint — without ever disturbing the live Wav2Vec2/CTC model, and make `settings["model"]` a real switch between the two.

**Architecture:** `src/cli/train_whisper_adapter.py` (moved and rehabilitated from the repo-root prototype) mirrors the robustness already built into `src/cli/train_adapter.py` this session (seeding, dataset filtering, safe `num_proc`, early stopping, rich reporting) but stays a fully separate module — no shared base class. Only two genuinely architecture-agnostic helpers (`set_seed`, `build_augment_pipeline`) move to `src/utils/utils.py` for both trainers to share. The Whisper trainer always writes to a `_whisper`-suffixed output directory, so it can run at any time without risk to the live model. `settings["model"]` (`"wav2vec2"` | `"whisper"`) becomes the single switch that both `TrainingService` (which trainer runs) and `api.py`'s `get_asr()` (which adapter serves `/translate`) read.

**Tech Stack:** PyTorch, HuggingFace `transformers` (5.8.0) `Seq2SeqTrainer`, `peft` (0.19.1) LoRA, `datasets` (4.8.5), `evaluate` (0.4.6), FastAPI, React/TypeScript (one file only, in the `react-rewrite` worktree).

**Spec:** `docs/superpowers/specs/2026-09-20-whisper-lora-training-design.md`

## Global Constraints

- `src/cli/train_adapter.py` (the CTC trainer, stabilized this session in commit `7160694c`) is **not modified** beyond removing its now-duplicated `set_seed`/`build_augment_pipeline` definitions in favor of importing them — no other behavior change.
- The Whisper trainer's output directory is **always** `data/models/personalized_models/{user_id}_whisper` — never the bare `{user_id}` path used by the live Wav2Vec2 model. This is a hard safety property, not a convention to remember.
- Default Whisper checkpoint: `"selimc/whisper-large-v3-turbo-turkish"` (new `config.WHISPER_MODEL_NAME`).
- LoRA is the only fine-tuning algorithm on either path; `config.ADAPTER_REDUCTION_FACTOR` (128) and its `lora_alpha = ADAPTER_REDUCTION_FACTOR * 2` convention are shared unchanged.
- `transformers` in this venv (`/home/mayasoft/app/venv/bin/python3`) is 5.8.0: use `eval_strategy` (not `evaluation_strategy`) and `processing_class` (not `tokenizer`) on `Seq2SeqTrainingArguments`/`Seq2SeqTrainer` — verified this session against the actual installed version.
- `ReportingService.log_training_session(user_id, training_data)` (`src/services/reporting_service.py:149`) expects a **flat** dict (`base_model`, `num_epochs`, `train_samples`, `final_wer`, etc. — not nested) — read that method before writing the Whisper trainer's report block; do not invent a nested shape.
- GPU is a `NVIDIA L40S-12Q` — a 12GB vGPU slice, not a full L40S. The 0.8B-parameter checkpoint's memory fit is unverified before Task 5.
- Train/eval data (`data/users/FurkanV1/train.csv` + `eval.csv`, 5562/1391 samples) has not changed since 2026-08-11 and is not touched by this plan.

---

## Task 1: Extract `set_seed` and `build_augment_pipeline` into `src/utils/utils.py`

**Files:**
- Modify: `src/utils/utils.py`
- Modify: `src/cli/train_adapter.py:1-145` (imports + remove local defs)

**Interfaces:**
- Produces: `set_seed(seed: int) -> None`, `build_augment_pipeline(sampling_rate: int) -> A.Compose | None` in `src.utils.utils`. Used by Task 3's Whisper trainer and by `train_adapter.py`.

- [ ] **Step 1: Add the two functions to `src/utils/utils.py`**

Add near the top of `src/utils/utils.py`, after the existing `import config` line (line 11) — add these two new imports there too (`random`, `torch`, and a local `audiomentations` availability check):

```python
import random
import torch

try:
    import audiomentations as A
    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False
```

Then append these two functions anywhere in the file (e.g. right after `normalize_path_for_cross_platform`, before `resolve_audio_path`):

```python
def set_seed(seed: int):
    """Reprodüsibilite için ağırlık başlangıcı, veri karıştırma ve augmentation RNG'lerini sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_augment_pipeline(sampling_rate: int):
    """
    Konuşma bozukluğu için optimize edilmiş augmentation pipeline.
    Parametreler config.py üzerinden kontrol edilir; hafif bozulma kullanılır
    böylece bozukluk kalıpları korunur.
    """
    if not AUDIOMENTATIONS_AVAILABLE:
        return None

    return A.Compose([
        A.AddGaussianNoise(
            min_amplitude=config.AUGMENT_NOISE_MIN,
            max_amplitude=config.AUGMENT_NOISE_MAX,
            p=0.3,
        ),
        A.TimeStretch(
            min_rate=config.AUGMENT_TIME_STRETCH_MIN,
            max_rate=config.AUGMENT_TIME_STRETCH_MAX,
            p=0.3,
            leave_length_unchanged=False,
        ),
        A.PitchShift(
            min_semitones=config.AUGMENT_PITCH_MIN,
            max_semitones=config.AUGMENT_PITCH_MAX,
            p=0.3,
        ),
        A.TimeMask(
            min_band_part=config.AUGMENT_TIME_MASK_MIN,
            max_band_part=config.AUGMENT_TIME_MASK_MAX,
            p=0.2,
        ),
    ], p=config.AUGMENT_PROBABILITY)
```

- [ ] **Step 2: Update `train_adapter.py` to import instead of define**

In `src/cli/train_adapter.py`:

Remove line 6 (`import random` — becomes unused, `random` is only used inside `set_seed`).

Change line 29:
```python
from src.utils.utils import save_model_and_processor
```
to:
```python
from src.utils.utils import save_model_and_processor, set_seed, build_augment_pipeline
```

Delete the local `set_seed` function (lines 44-49, the def and its 4-line body) and the local `build_augment_pipeline` function (lines 118-149, the def through its closing `], p=config.AUGMENT_PROBABILITY)`). Leave everything else in the file untouched — in particular keep the module-level `try: import audiomentations as A; AUDIOMENTATIONS_AVAILABLE = True / except ImportError: AUDIOMENTATIONS_AVAILABLE = False` block (lines 37-41) exactly as-is, since `train_adapter.py` also uses `AUDIOMENTATIONS_AVAILABLE` directly for print/report logic elsewhere (lines ~538, ~552, ~852) independent of the pipeline builder.

- [ ] **Step 3: Verify both files still import and behave identically**

```bash
cd /home/mayasoft/app/Pronouns
/home/mayasoft/app/venv/bin/python3 -c "
import ast
ast.parse(open('src/utils/utils.py').read())
ast.parse(open('src/cli/train_adapter.py').read())
print('syntax OK')
"
/home/mayasoft/app/venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from src.utils.utils import set_seed, build_augment_pipeline
set_seed(42)
pipeline = build_augment_pipeline(16000)
print('set_seed OK, build_augment_pipeline ->', type(pipeline).__name__ if pipeline else None)
"
/home/mayasoft/app/venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from src.cli.train_adapter import PersonalizedTrainer, set_seed, build_augment_pipeline
print('train_adapter imports OK')
"
```

Expected: all three commands print their OK lines with no traceback. The second command's pipeline type should print `Compose` (audiomentations is already installed and used successfully by every prior training run this session).

- [ ] **Step 4: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add src/utils/utils.py src/cli/train_adapter.py
git commit -m "refactor: extract set_seed/build_augment_pipeline to src/utils/utils.py

Both are architecture-agnostic and about to be needed by a second
(Whisper) trainer alongside the existing CTC one; dedup now rather
than fork a second copy."
```

---

## Task 2: Add `config.WHISPER_MODEL_NAME`, dedupe shadowed constants

**Files:**
- Modify: `config.py:6-9`, `config.py:55-59`

**Interfaces:**
- Produces: `config.WHISPER_MODEL_NAME: str`. Used by Task 3.

- [ ] **Step 1: Add the new constant**

In `config.py`, change:
```python
# --- ASR Model Ayarları ---
MODEL_NAME = "mpoyraz/wav2vec2-xls-r-300m-cv8-turkish"  # Varsayılan model
ORNEKLEME_ORANI = 16000  # Hz
```
to:
```python
# --- ASR Model Ayarları ---
MODEL_NAME = "mpoyraz/wav2vec2-xls-r-300m-cv8-turkish"  # Varsayılan model (Wav2Vec2/CTC)
WHISPER_MODEL_NAME = "selimc/whisper-large-v3-turbo-turkish"  # Whisper kişiselleştirme başlangıç noktası
ORNEKLEME_ORANI = 16000  # Hz
```

- [ ] **Step 2: Remove the shadowed duplicate constants**

`NUM_FINETUNE_EPOCHS`, `FINETUNE_BATCH_SIZE`, and `GRADIENT_ACCUMULATION_STEPS` are each defined twice in `config.py`; the first definition of each is dead code (Python keeps the second). Change:
```python
FINETUNE_OUTPUT_DIR = "./asr_model_finetuned"
FINETUNE_BATCH_SIZE = 4            # RTX A5000 için 16'dan 4'e düşürüldü (OOM engelleme)
GRADIENT_ACCUMULATION_STEPS = 8    # Gradyan biriktirme (etkin batch size = 4 * 8 = 32, korundu)
NUM_FINETUNE_EPOCHS = 50           # Daha fazla epoch ile daha derin öğrenme
FINETUNE_EVAL_STEPS = 100
```
to:
```python
FINETUNE_OUTPUT_DIR = "./asr_model_finetuned"
FINETUNE_EVAL_STEPS = 100
```
(This deletes exactly the three shadowed lines and leaves `FINETUNE_OUTPUT_DIR` and `FINETUNE_EVAL_STEPS` — which are not duplicated anywhere — in place.) The surviving second definitions further down (`EARLY_STOPPING_PATIENCE = 30`, `NUM_FINETUNE_EPOCHS = 100`, `FINETUNE_BATCH_SIZE = 8`, `GRADIENT_ACCUMULATION_STEPS = 4`, under the `# Eğitim stabilite ayarları` comment) are untouched.

- [ ] **Step 3: Verify effective values are unchanged and the new constant resolves**

```bash
cd /home/mayasoft/app/Pronouns
/home/mayasoft/app/venv/bin/python3 -c "
import config
assert config.NUM_FINETUNE_EPOCHS == 100, config.NUM_FINETUNE_EPOCHS
assert config.FINETUNE_BATCH_SIZE == 8, config.FINETUNE_BATCH_SIZE
assert config.GRADIENT_ACCUMULATION_STEPS == 4, config.GRADIENT_ACCUMULATION_STEPS
assert config.WHISPER_MODEL_NAME == 'selimc/whisper-large-v3-turbo-turkish'
print('config OK:', config.NUM_FINETUNE_EPOCHS, config.FINETUNE_BATCH_SIZE, config.GRADIENT_ACCUMULATION_STEPS, config.WHISPER_MODEL_NAME)
"
```

Expected: `config OK: 100 8 4 selimc/whisper-large-v3-turbo-turkish` — these are the exact same effective values as before this task (verified by reading the file: the second definitions, which Python already used, are the ones that survive).

- [ ] **Step 4: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add config.py
git commit -m "chore: add WHISPER_MODEL_NAME, remove shadowed duplicate constants

NUM_FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE, and GRADIENT_ACCUMULATION_STEPS
were each defined twice; Python silently used the second definition in
every case. Removing the dead first definitions is a no-op on behavior
(verified in the previous step) and stops the file lying about its own
effective values."
```

---

## Task 3: Move and rehabilitate the Whisper trainer into `src/cli/`

**Files:**
- Create: `src/cli/train_whisper_adapter.py`
- Delete: `train_whisper_adapter.py` (repo root)

**Interfaces:**
- Consumes: `set_seed`, `build_augment_pipeline`, `resolve_audio_path`, `normalize_turkish_text` from `src.utils.utils` (Task 1 for the first two; the latter two already exist there). `config.WHISPER_MODEL_NAME` (Task 2).
- Produces: `class WhisperPersonalizedTrainer` with `__init__(user_id, base_model_path=None)` and `.run()`, writing to `data/models/personalized_models/{user_id}_whisper`. CLI: `python -m src.cli.train_whisper_adapter <user_id> [--base_model ...]`. Used by Task 5 (smoke test), Task 6 (`TrainingService` wiring), Task 8 (full run).

- [ ] **Step 1: Delete the root prototype**

```bash
cd /home/mayasoft/app/Pronouns
git rm train_whisper_adapter.py
```

- [ ] **Step 2: Create the rehabilitated trainer**

`src/cli/train_whisper_adapter.py`:

```python
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
            audio, _ = librosa.load(batch["file_path"], sr=sample_rate)
            if augmenter:
                audio = augmenter(samples=audio, sample_rate=sample_rate)
            batch["input_features"] = self.processor.feature_extractor(
                audio, sampling_rate=sample_rate
            ).input_features[0]
            batch["labels"] = self.processor.tokenizer(
                normalize_turkish_text(batch["transcript"])
            ).input_ids
            return batch

        # Ayni guvenlik deseni train_adapter.py'de bu oturumda duzeltildi: num_proc=1
        # HuggingFace datasets'te hala bir multiprocessing pool acar ve "OSError: got
        # end of file during message" hatasina yol acar. num_proc>1 disinda None kullan.
        num_proc = min(config.DATA_PREPROCESSING_NUM_PROC, os.cpu_count() or 1)
        map_num_proc = num_proc if num_proc > 1 else None

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(preprocess, remove_columns=dataset.column_names, num_proc=map_num_proc)

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
```

- [ ] **Step 3: Verify it imports and the CLI parses**

```bash
cd /home/mayasoft/app/Pronouns
/home/mayasoft/app/venv/bin/python3 -c "
import ast
ast.parse(open('src/cli/train_whisper_adapter.py').read())
print('syntax OK')
"
/home/mayasoft/app/venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from src.cli.train_whisper_adapter import WhisperPersonalizedTrainer
t = WhisperPersonalizedTrainer(user_id='FurkanV1')
assert str(t.output_dir) == 'data/models/personalized_models/FurkanV1_whisper', t.output_dir
assert t.base_model_path == 'selimc/whisper-large-v3-turbo-turkish'
print('WhisperPersonalizedTrainer OK, output_dir =', t.output_dir)
"
/home/mayasoft/app/venv/bin/python3 -m src.cli.train_whisper_adapter --help
```

Expected: all three succeed; the output_dir assertion confirms the safety property (separate path) holds; `--help` prints the argparse usage without error.

- [ ] **Step 4: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add -A -- src/cli/train_whisper_adapter.py train_whisper_adapter.py
git commit -m "feat: rehabilitate Whisper+LoRA trainer, move to src/cli/

Moves the unfinished repo-root prototype into src/cli/, alongside its
CTC sibling. Brings it to the same robustness train_adapter.py gained
this session: default checkpoint is the Turkish-tuned
selimc/whisper-large-v3-turbo-turkish (was openai/whisper-medium),
output always goes to a _whisper-suffixed directory so it can never
collide with the live Wav2Vec2 model, dataset prep gains the same
file-existence/quality/empty-transcript filters and the same safe
num_proc handling (never passes num_proc=1 into datasets' multiprocessing
pool), early stopping is wired via EarlyStoppingCallback, and the
training report matches train_adapter.py's fuller shape."
```

---

## Task 4: Whisper prompt-based hotword biasing in `src/core/asr.py`

**Files:**
- Modify: `src/core/asr.py:238-243`

**Interfaces:**
- Consumes: `self.user_vocabulary` (already loaded in `__init__` via `_load_user_vocabulary`).
- Produces: no new public interface — behavior change only, inside `transcribe()`'s Whisper branch. Exercised end-to-end by Task 5.

- [ ] **Step 1: Add prompt biasing to the Whisper branch**

In `src/core/asr.py`, change:
```python
            if self.model_type == "whisper":
                input_features = self.processor(speech, sampling_rate=sr, return_tensors="pt").input_features.to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(input_features, language="tr", task="transcribe")
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                confidence = 0.85 
```
to:
```python
            if self.model_type == "whisper":
                input_features = self.processor(speech, sampling_rate=sr, return_tensors="pt").input_features.to(self.device)

                generate_kwargs = {"language": "tr", "task": "transcribe"}
                if self.user_vocabulary:
                    # Whisper'in resmi hotword/context-biasing yontemi: kullanicinin
                    # bildigi kelimeleri decoder'a "onceki baglam" gibi veriyoruz.
                    # 50 kelime sinirlamasi Whisper'in decoder context budgetini asmamak icin.
                    prompt_text = " ".join(self.user_vocabulary[:50])
                    generate_kwargs["prompt_ids"] = self.processor.get_prompt_ids(
                        prompt_text, return_tensors="pt"
                    ).to(self.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(input_features, **generate_kwargs)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                confidence = 0.85 
```

- [ ] **Step 2: Verify it compiles and the no-vocabulary path is unaffected**

```bash
cd /home/mayasoft/app/Pronouns
/home/mayasoft/app/venv/bin/python3 -c "
import ast
ast.parse(open('src/core/asr.py').read())
print('syntax OK')
"
```

Expected: `syntax OK`. Full behavioral verification (that `prompt_ids` actually improves transcription and that a real `generate()` call with it doesn't error) requires a loadable Whisper adapter, which doesn't exist yet — this is exercised for real in Task 5's smoke test and Task 8's validation, both of which call `transcribe()` through the Whisper branch.

- [ ] **Step 3: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add src/core/asr.py
git commit -m "feat: add Whisper prompt-based hotword biasing to asr.py

The Wav2Vec2/CTC branch already biases toward the user's known
vocabulary via pyctcdecode's hotwords parameter; Whisper had no
equivalent. Uses Whisper's documented context-conditioning hook
(processor.get_prompt_ids + generate(prompt_ids=...)) capped at the
user's first 50 vocabulary words. Only activates when a vocabulary
exists; falls through to the prior behavior otherwise."
```

---

## Task 5: Smoke test — confirm the Whisper pipeline runs end to end

**Files:** none (verification-only task; no source changes)

**Interfaces:**
- Consumes: `WhisperPersonalizedTrainer` (Task 3).
- Produces: a real (if under-trained) adapter at `data/models/personalized_models/FurkanV1_whisper`, and a report in `reports/`, both consumed by Task 8's later full-quality comparison (this task's own model is expected to be low-quality — 1 epoch — and is not the one that matters for WER; only completion without crashing matters here).

This is the first real GPU run of the new code and the first time this session's `num_proc` safety fix and the 0.8B checkpoint's memory footprint on the `L40S-12Q` 12GB slice are both exercised together. Run it small and fast before committing hours to Task 8.

- [ ] **Step 1: Run 1 epoch as a feasibility check**

```bash
cd /home/mayasoft/app/Pronouns
mkdir -p logs
/home/mayasoft/app/venv/bin/python3 -c "
import config
config.NUM_FINETUNE_EPOCHS = 1
config.EARLY_STOPPING_PATIENCE = 1
from src.cli.train_whisper_adapter import WhisperPersonalizedTrainer
from src.utils.utils import set_seed
set_seed(config.SEED)
trainer = WhisperPersonalizedTrainer(user_id='FurkanV1')
trainer.run()
" 2>&1 | tee logs/whisper_smoke_test.out
```

(Overriding `config.NUM_FINETUNE_EPOCHS`/`EARLY_STOPPING_PATIENCE` in-process, rather than editing `config.py`, keeps this a throwaway feasibility run with zero risk of accidentally committing a "1 epoch" config.)

- [ ] **Step 2: Confirm success**

```bash
tail -40 logs/whisper_smoke_test.out
ls -la data/models/personalized_models/FurkanV1_whisper/
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Expected: the tail shows `✅ Model kaydedildi: data/models/personalized_models/FurkanV1_whisper` and `📊 Training report saved: reports/...`, with no `OSError`, no `CUDA out of memory`, no Python traceback anywhere in the log. The `ls` shows adapter files (`adapter_model.safetensors` or `.bin`, `adapter_config.json`). If it OOMs: the first thing to try is lowering `config.FINETUNE_BATCH_SIZE` for the Whisper path specifically (this plan does not add a separate Whisper batch-size constant — if Task 5 OOMs, that is new information changing the plan, and the fix is to add one, e.g. `WHISPER_FINETUNE_BATCH_SIZE`, before proceeding to Task 8; do not silently reuse a smaller value without recording it).

- [ ] **Step 3: Confirm `asr.py` loads the resulting adapter and the hotword path runs**

```bash
cd /home/mayasoft/app/Pronouns
/home/mayasoft/app/venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from src.core.asr import ASRSystem
asr = ASRSystem(model_name='data/models/personalized_models/FurkanV1_whisper', user_id='FurkanV1')
assert asr.model_type == 'whisper', asr.model_type
print('Loaded OK, model_type =', asr.model_type, '| vocabulary size =', len(asr.user_vocabulary))
"
```

Expected: prints `Loaded OK, model_type = whisper | vocabulary size = <N>` with `<N> > 0` (confirms Task 4's `if self.user_vocabulary:` branch will actually engage on the next real `/translate` call, not silently skip due to an empty vocabulary).

No commit for this task — it produces only a throwaway model checkpoint and a log file under `logs/`/`reports/`, not a code change.

---

## Task 6: Wire `settings["model"]` to select architecture

**Files:**
- Modify: `src/services/training_service.py:83-96`
- Modify: `api.py:201-205`

**Interfaces:**
- Consumes: `SettingsService.load_settings(user_id)["model"]` (already exists, default `user_id`-as-string today — see Step 1's note on that default).
- Produces: `TrainingService.start_training()` runs the correct trainer; `api.py`'s `get_asr(user_id)` serves the correct adapter. Both read the same setting, so they always agree.

- [ ] **Step 1: Give `settings["model"]`'s default a real, meaningful value**

`SettingsService.load_settings`'s `default_settings` dict (`src/services/settings_service.py:24`) currently sets `"model": user_id` (e.g. `"FurkanV1"`) — a leftover placeholder, never checked against anything until now. Change it to the value this plan actually branches on:

In `src/services/settings_service.py`, change:
```python
        default_settings = {
            "model": user_id,
            "algorithm": "lora",
```
to:
```python
        default_settings = {
            "model": "wav2vec2",
            "algorithm": "lora",
```

This is the only change in that file — `save_settings`/the JSON storage format are untouched, so any user who has never saved settings gets `"wav2vec2"` (today's only working behavior) and anyone who explicitly saves `"model": "whisper"` gets the new path.

- [ ] **Step 2: Branch `TrainingService.start_training()` on the setting**

In `src/services/training_service.py`, change:
```python
        # PersonalizedTrainer'ı içe aktar (döngüsel bağımlılığı önlemek için)
        from src.cli.train_adapter import PersonalizedTrainer

        def run_training():
            try:
                # Ayarları yükle
                settings = self.settings_service.load_settings(self.user_id)
                
                # Config'i geçici olarak güncelle (opsiyonel, trainer config'den okuyor)
                # Not: train_adapter.py şu an config.py'den direkt okuyor. 
                # Gelecekte trainer'a settings objesi geçilebilir.
                
                trainer = PersonalizedTrainer(user_id=self.user_id)
                trainer.run()
```
to:
```python
        def run_training():
            try:
                settings = self.settings_service.load_settings(self.user_id)

                if settings.get("model") == "whisper":
                    from src.cli.train_whisper_adapter import WhisperPersonalizedTrainer
                    trainer = WhisperPersonalizedTrainer(user_id=self.user_id)
                else:
                    from src.cli.train_adapter import PersonalizedTrainer
                    trainer = PersonalizedTrainer(user_id=self.user_id)

                trainer.run()
```
(The trainer imports move inside `run_training()` — same "avoid circular import at module load time" reasoning the original comment gave, now applied to both trainers instead of one imported eagerly at `start_training()`'s top.)

- [ ] **Step 3: Make `get_asr()` resolve the architecture-specific path**

In `api.py`, change:
```python
def get_asr(user_id: str):
    if user_id not in asr_systems:
        model_path = ModelService.find_personalized_model(user_id)
        asr_systems[user_id] = ASRSystem(model_name=model_path, user_id=user_id)
    return asr_systems[user_id]
```
to:
```python
def get_asr(user_id: str):
    settings = settings_service.load_settings(user_id)
    model_key = f"{user_id}_whisper" if settings.get("model") == "whisper" else user_id
    if model_key not in asr_systems:
        model_path = ModelService.find_personalized_model(model_key)
        asr_systems[model_key] = ASRSystem(model_name=model_path, user_id=user_id)
    return asr_systems[model_key]
```

Note the two different uses of the ID here are deliberate: `find_personalized_model(model_key)` resolves the (possibly `_whisper`-suffixed) model directory, while `ASRSystem(..., user_id=user_id)` keeps the **unsuffixed** ID — that argument drives `_load_user_vocabulary()`, which reads `data/users/{user_id}/metadata_words.csv`, a path that must never be suffixed.

- [ ] **Step 4: Verify with the real API process**

```bash
cd /home/mayasoft/app/Pronouns
/home/mayasoft/app/venv/bin/python3 -c "
import ast
for f in ['src/services/settings_service.py', 'src/services/training_service.py', 'api.py']:
    ast.parse(open(f).read())
print('syntax OK')
"
```

Then, against the running production API (already live on this host per this session's earlier work), confirm the settings round-trip and the cutover:

```bash
API="https://furkanca.algorixstudio.com"
KEY="c2a6dd1afc678eb93f0c46c8c1528e2b1c07676fbde0951f"

curl -sS -H "X-API-Key: $KEY" "$API/settings?user_id=FurkanV1"
curl -sS -H "X-API-Key: $KEY" -X POST "$API/settings" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"FurkanV1","model":"whisper"}'
curl -sS -H "X-API-Key: $KEY" "$API/settings?user_id=FurkanV1"
```

Expected: the first call shows the current settings (likely no `"model"` key yet, or `"FurkanV1"` from the old placeholder default, depending on whether a `settings.json` already exists for this user — either way this step is read-only and safe). The second call returns `{"status": "saved", ...}` including `"model": "whisper"`. The third call confirms it persisted. **This changes what `/translate` serves for real users the next time `get_asr` is called for FurkanV1** — since Task 5's smoke-test adapter is low-quality (1 epoch), immediately set it back before moving on:

```bash
curl -sS -H "X-API-Key: $KEY" -X POST "$API/settings" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"FurkanV1","model":"wav2vec2"}'
```

This requires restarting `pronouns-api.service` first if you want to test against the code in this task before it's deployed — the currently *running* API process still has the old `get_asr`/`training_service.py` code in memory (Python doesn't hot-reload). Restarting a `Restart=always` systemd service serving a real patient-facing app is exactly the kind of action that needs a human's go-ahead first (see this plan's execution notes) — do not restart `pronouns-api.service` as part of this task's automated steps; stop after the syntax check and hand the settings-round-trip verification to the coordinator.

- [ ] **Step 5: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add src/services/settings_service.py src/services/training_service.py api.py
git commit -m "feat: wire settings[model] to real Wav2Vec2/Whisper selection

TrainingService.start_training() and api.py's get_asr() both now read
settings[\"model\"] (\"wav2vec2\" default | \"whisper\") to decide which
trainer runs and which personalized adapter serves /translate. This
was previously decorative — training_service.py had a comment
admitting it always trained CTC regardless of settings. Switching a
user to the Whisper path, once trained and validated, is now a single
POST /settings call; switching back is the same call with the
default. The live Wav2Vec2 adapter directory is never written to by
this change."
```

---

## Task 7: Trim the React `AyarlarPage`'s decorative options

**Files:**
- Modify: `.claude/worktrees/react-rewrite/web/src/pages/AyarlarPage.tsx`

**Interfaces:**
- Consumes: none new — this only changes which string values the existing `<select>` elements offer for the already-existing `settings.model`/`settings.algorithm` fields.
- Produces: nothing consumed elsewhere.

This task lands in the `react-rewrite` git worktree (a separate branch, `worktree-react-rewrite`, not `master`), since that is where `AyarlarPage.tsx` lives.

- [ ] **Step 1: Update the option arrays**

In `.claude/worktrees/react-rewrite/web/src/pages/AyarlarPage.tsx`, change:
```tsx
const MODEL_OPTIONS = ['Furkan', 'base', 'large'];
const ALGO_OPTIONS = ['lora', 'qlora', 'full'];
```
to:
```tsx
const MODEL_OPTIONS = ['wav2vec2', 'whisper'];
const ALGO_OPTIONS = ['lora'];
```

- [ ] **Step 2: Verify the build still compiles**

```bash
cd /home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite/web
npm run build
```

Expected: build succeeds with no TypeScript errors (this is a same-type array literal change — `string[]` in, `string[]` out — so no type errors are expected, but the project's `noUnusedLocals`/strict settings mean it's worth confirming).

- [ ] **Step 3: Commit (in the worktree)**

```bash
cd /home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite
git add web/src/pages/AyarlarPage.tsx
git commit -m "fix: Ayarlar model/algorithm dropdowns show only real options

MODEL_OPTIONS and ALGO_OPTIONS previously listed values the backend
never implemented (base/large/qlora/full). Now: wav2vec2 | whisper
for model (both real as of this session's backend work), lora only
for algorithm (the only fine-tuning method implemented for either
architecture)."
```

Do not deploy/redeploy the web app as part of this task — this worktree's deploy cadence belongs to the separate react-rewrite plan already in flight; leave this commit staged in its branch history for that plan's next deploy step.

---

## Task 8: Full training run and end-to-end validation

**Files:** none (execution + validation task; no further source changes expected unless Task 5's OOM contingency was triggered)

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: a real, fully-trained `FurkanV1_whisper` adapter and a decision (recorded in this plan's completion notes, not code) on whether it beats the Wav2Vec2 baseline enough to flip production over.

This task is long-running (historically 1-3 hours per full CTC run on this data; a 0.8B Whisper model with beam-search generation during eval is plausibly slower per epoch) and is not a good fit for a subagent that must finish in one sitting. **Launch it, then check back** — this mirrors how the CTC trainer's own verification run was handled earlier this session (background process, periodic log checks), rather than a subagent blocking synchronously for hours.

- [ ] **Step 1: Launch the real training run in the background**

```bash
cd /home/mayasoft/app/Pronouns
mkdir -p logs
nohup /home/mayasoft/app/venv/bin/python3 -m src.cli.train_whisper_adapter FurkanV1 \
  > logs/train_whisper_full_$(date +%Y%m%d_%H%M%S).out 2>&1 &
echo "PID: $!"
```

- [ ] **Step 2: Poll until it finishes**

```bash
ps -p <PID_FROM_STEP_1> -o pid,etime,cmd
tail -40 logs/train_whisper_full_*.out
```

Expected on completion: the same `✅ Model kaydedildi` / `📊 Training report saved` lines as Task 5, this time from a full run (early-stopped or reaching `NUM_FINETUNE_EPOCHS=100`).

- [ ] **Step 3: Compare WER against both baselines**

```bash
cd /home/mayasoft/app/Pronouns
/home/mayasoft/app/venv/bin/python3 -c "
import json, glob
latest = sorted(glob.glob('reports/training_FurkanV1_*.json'))[-1]
data = json.load(open(latest))['data']
print(latest)
print('WER:', data['training_metrics']['final_wer'], '| CER:', data['training_metrics']['final_cer'])
print('base_model:', data['training_parameters']['base_model'])
"
```

Compare the printed WER against the historical Wav2Vec2 best (64.99%, 2026-08-11) and the current seeded Wav2Vec2 baseline (82.50%, 2026-09-19, this session). There is no fixed numeric bar this plan requires clearing — report the number to the user and let them decide whether it's good enough to cut over, per this plan's non-goal of not baking in an untested quality gate.

- [ ] **Step 4: If cutting over — flip settings, verify live, verify rollback**

Only do this with explicit go-ahead (this affects a real patient-facing app):

```bash
API="https://furkanca.algorixstudio.com"
KEY="c2a6dd1afc678eb93f0c46c8c1528e2b1c07676fbde0951f"

# Cut over
curl -sS -H "X-API-Key: $KEY" -X POST "$API/settings" -H "Content-Type: application/json" \
  -d '{"user_id":"FurkanV1","model":"whisper"}'

# Confirm the live app transcribes end to end (manual: open the Furkanca page, speak,
# confirm a corrected transcription + audio comes back — same manual check the
# react-rewrite plan's Task 13 already used).

# Rollback check
curl -sS -H "X-API-Key: $KEY" -X POST "$API/settings" -H "Content-Type: application/json" \
  -d '{"user_id":"FurkanV1","model":"wav2vec2"}'
```

Expected: after the cutover call, the API logs (`journalctl -u pronouns-api.service -f` or the equivalent log tail) show `"Whisper mimarisi tespit edildi: ..."` on the next `/translate` call for FurkanV1 (this log line already exists in `asr.py`, unmodified by this plan). After the rollback call, the next `/translate` call loads the Wav2Vec2 adapter again with no restart required — confirming `get_asr`'s per-`model_key` caching (Task 6) works as designed.

No commit for this task by itself — Steps 1-3 produce data (a checkpoint + a report), not code, and Step 4's `curl` calls change runtime settings state, not source. If Task 5's OOM contingency required a code change (new `WHISPER_FINETUNE_BATCH_SIZE` or similar), that change should already be committed as part of a revised Task 5, not deferred here.

---

## Plan Self-Review Notes

- **Spec coverage:** §4.1 (shared utils) → Task 1. §4.2 (trainer rehab) → Task 3. §4.3 (hotword biasing) → Task 4. §4.4 (settings wiring) → Task 6, plus the React side → Task 7. §4.5 (config cleanup) → Task 2. §6 (testing/validation) → Tasks 5 and 8. §7 non-goals (Whisper confidence stub, QLoRA/full, Flutter UI, the two mobile-app items raised separately) are deliberately untouched by every task above.
- **Type/name consistency checked:** `WhisperPersonalizedTrainer(user_id, base_model_path=None)` constructor signature is identical between Task 3 (definition) and Tasks 5/6/8 (call sites). `config.WHISPER_MODEL_NAME` (Task 2) is the exact name read in Task 3. `get_asr`'s `model_key` (Task 6) matches the `{user_id}_whisper` suffix convention fixed in Task 3's `output_dir`.
