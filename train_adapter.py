# train_adapter.py
# Konuşma bozukluğu için optimize edilmiş model eğitim scripti

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
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
from src.services.reporting_service import ReportingService
import config
import librosa
import evaluate
import platform
import time
try:
    import audiomentations as A
    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False
    print("⚠️  audiomentations bulunamadı. Augmentation kullanılamayacak.")

# Linux sunucu için logging yapılandırması
def setup_logging(user_id):
    """Linux sunucu için logging yapılandırması."""
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Logging formatı
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Log seviyesi
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    print(f"📝 Log dosyası: {log_file}")
    return log_file

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

def _standalone_preprocess_function(examples, processor, augmenter=None):
    """
    Standalone data preprocessing function for multiprocessing.
    Wav2Vec2 için özellik çıkarımı ve tokenization yapar.
    Augmentation desteği eklenmiştir.
    """
    # Ses dosyalarını yükle
    audio_arrays = []
    valid_transcripts = []
    
    # Transcript sütununu belirle
    transcript_key = "transcript" if "transcript" in examples else "transcription"
    transcripts = examples.get(transcript_key, [""] * len(examples["file_path"]))
    
    for i, path_dict in enumerate(examples["file_path"]):
        try:
            audio, sr = librosa.load(path_dict['path'], sr=config.ORNEKLEME_ORANI)
            
            # Minimum uzunluk kontrolü (en az 0.1 saniye)
            if len(audio) < config.ORNEKLEME_ORANI * 0.1:
                continue
            
            # Augmentation uygula (eğer varsa)
            if augmenter is not None:
                try:
                    audio = augmenter(samples=audio, sample_rate=sr)
                except Exception as e:
                    # Augmentation hatası durumunda orijinal sesi kullan
                    pass
            
            # Transcript kontrolü
            transcript = str(transcripts[i]).strip() if i < len(transcripts) else ""
            if transcript:
                audio_arrays.append(audio)
                valid_transcripts.append(transcript)
        except Exception as e:
            # Hata durumunda sessizce atla
            continue
    
    if len(audio_arrays) == 0:
        # Boş batch için dummy değerler döndür
        return {
            "input_values": np.array([0.0]),
            "labels": [processor.tokenizer.pad_token_id]
        }
    
    # Processor ile özellik çıkarımı (her örnek için ayrı ayrı)
    input_values_list = []
    label_ids_list = []
    valid_indices = []
    
    for i, (audio, transcript) in enumerate(zip(audio_arrays, valid_transcripts)):
        try:
            # Her örnek için ayrı ayrı işle
            inputs = processor(
                audio, 
                sampling_rate=config.ORNEKLEME_ORANI, 
                return_tensors="pt", 
                padding=False  # Padding yapma, collator yapacak
            )
            
            # Input values'ı list'e çevir
            input_vals = inputs.input_values[0]
            if isinstance(input_vals, torch.Tensor):
                input_vals = input_vals.tolist()
            input_values_list.append(input_vals)
            
            # Transcript'i tokenize et
            # tokenizer() dict döndürür, input_ids'i al
            tokenized = processor.tokenizer(transcript)
            label_ids = tokenized.input_ids
            
            # Format kontrolü ve düzeltme
            if isinstance(label_ids, torch.Tensor):
                label_ids = label_ids.tolist()
            
            # Eğer nested list ise ([[1,2,3]] gibi) flatten et
            if label_ids and isinstance(label_ids[0], list):
                label_ids = label_ids[0]
            
            # Boş label kontrolü
            if not label_ids or len(label_ids) == 0:
                print(f"⚠️  Boş label: '{transcript}' -> atlanıyor")
                continue
            
            # Label'ların geçerli token ID'leri içerdiğini kontrol et
            if any(not isinstance(tid, int) or tid < 0 for tid in label_ids):
                print(f"⚠️  Geçersiz label IDs: {label_ids[:5]}... -> atlanıyor")
                continue
                
            label_ids_list.append(label_ids)
            valid_indices.append(i)
            
        except Exception as e:
            print(f"⚠️  Örnek {i} işlenirken hata: {e}")
            continue
    
    # Eğer hiç geçerli örnek yoksa
    if len(input_values_list) == 0:
        return {
            "input_values": [[0.0]],
            "labels": [[processor.tokenizer.pad_token_id]]
        }
    
    # Sonuçları dict olarak döndür (collator için)
    result = {
        "input_values": input_values_list,
        "labels": label_ids_list
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
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        self.best_val_loss = float('inf')
        self.best_val_wer = float('inf')   # WER-bazlı en iyi model seçimi için
        self.patience_counter = 0
        self.checkpoint_dir = self.output_dir / "checkpoints"

    def run(self):
        # Linux sunucu için logging başlat
        log_file = setup_logging(self.user_id)
        logger = logging.getLogger(__name__)
        
        # Reporting service
        reporting_service = ReportingService()
        
        # Training start time
        training_start_time = time.time()
        start_datetime = datetime.now()
        
        print(f"🎯 {self.user_id} için kişiselleştirme süreci başlıyor.")
        print("="*50)
        logger.info(f"Training started for user: {self.user_id}")
        
        # System information
        sys_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python_version": sys.version.split()[0],  # Just version number
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else None
        }
        
        logger.info(f"Platform: {sys_info['platform']} {sys_info['platform_release']}")
        logger.info(f"Python: {sys_info['python_version']}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {sys_info['cuda_version']}")
            logger.info(f"GPU: {sys_info['gpu_name']}")
            logger.info(f"GPU Memory: {sys_info['gpu_memory_gb']:.2f} GB")
        
        if not self.user_data_path.exists() or not (self.user_data_path / "metadata_words.csv").exists():
            error_msg = f"❌ Hata: {self.user_data_path} için veri bulunamadı."
            print(error_msg)
            logger.error(error_msg)
            return

        self.load_model_and_processor()
        train_dataset = self.prepare_dataset(split='train')
        eval_dataset = self.prepare_dataset(split='eval')
        
        # Store dataset info for reporting
        self.train_samples = len(train_dataset) if train_dataset else 0
        self.eval_samples = len(eval_dataset) if eval_dataset else 0
        
        # Train model (will store metrics in self)
        self.train_model(train_dataset, eval_dataset, reporting_service, sys_info, start_datetime, training_start_time)
        
        logger.info(f"Training completed for user: {self.user_id}")

    def load_model_and_processor(self):
        print(f"📥 Temel model yükleniyor: {self.base_model_path}")
        self.processor = Wav2Vec2Processor.from_pretrained(self.base_model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.base_model_path)
        
        # Gradient checkpointing (opsiyonel, VRAM tasarrufu için)
        if config.GRADIENT_CHECKPOINTING and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("   ✅ Gradient checkpointing aktif (VRAM tasarrufu)")
        
        self.model.to(self.device)
        
        # Konuşma bozukluğu için optimize edilmiş LoRA konfigürasyonu.
        # Attention projeksiyon katmanlarına ek olarak feed-forward yoğun katmanlar
        # da dahil edilerek modelin konuşma bozukluğu kalıplarını öğrenme kapasitesi artırıldı.
        peft_config = LoraConfig(
            r=config.ADAPTER_REDUCTION_FACTOR,            # rank=16; daha yüksek -> daha iyi uyum
            lora_alpha=config.ADAPTER_REDUCTION_FACTOR * 2,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "out_proj",   # Attention katmanları
                "intermediate_dense", "output_dense",         # Feed-forward katmanları
            ],
            lora_dropout=0.05,
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Trainable parametreleri göster
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model PEFT/LoRA ile sarmalandı. Cihaz: {self.device}")
        print(f"   Eğitilebilir parametreler: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    def prepare_dataset(self, split='train'):
        """
        Veri setini hazırlar ve yükler.
        
        Args:
            split: 'train' veya 'eval'
        """
        print(f"📊 {split.upper()} veri seti hazırlanıyor: {self.user_data_path}")
        
        # Split'e göre dosya seç
        if split == 'train':
            train_csv = self.user_data_path / "train.csv"
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
        else:  # eval
            eval_csv = self.user_data_path / "eval.csv"
            if eval_csv.exists():
                print(f"   ✅ eval.csv bulundu, kullanılıyor.")
                df = pd.read_csv(eval_csv, encoding='utf-8')
            else:
                # metadata_words.csv'den oluştur (validation için)
                metadata_path = self.user_data_path / "metadata_words.csv"
                if not metadata_path.exists():
                    print(f"   ⚠️  eval.csv bulunamadı, validation seti oluşturulamıyor.")
                    return None
                
                print(f"   ⚠️  eval.csv bulunamadı, metadata_words.csv'nin %20'si validation için kullanılıyor.")
                df = pd.read_csv(metadata_path, encoding='utf-8')
                df = df[['file_path', 'transcription']].copy()
                df.rename(columns={'transcription': 'transcript'}, inplace=True)
                # Son %20'yi validation için al
                df = df.tail(int(len(df) * 0.2))

        # Dosya yollarını platforma göre çöz
        # Windows'ta kaydedilen \, Linux/macOS'ta / olan yollar aynı şekilde işlenir.
        from src.utils.utils import resolve_audio_path
        user_base = self.user_data_path
        df["file_path"] = df["file_path"].apply(
            lambda p: resolve_audio_path(p, user_base)
        )
        
        # Var olmayan dosyaları filtrele
        original_size = len(df)
        df = df[df["file_path"].apply(os.path.exists)]
        if len(df) < original_size:
            print(f"   {original_size - len(df)} adet bulunamayan ses dosyası atlandı.")

        # Kalite skoru filtrelemesi (eğer sütun mevcutsa)
        if "quality_score" in df.columns:
            before = len(df)
            df = df[df["quality_score"] >= config.QUALITY_THRESHOLD]
            removed = before - len(df)
            if removed > 0:
                print(f"   {removed} adet düşük kaliteli kayıt filtrelendi (eşik: {config.QUALITY_THRESHOLD}).")
        
        if len(df) == 0:
            if split == 'eval':
                return None
            raise ValueError(f"❌ Hata: Hiç geçerli ses dosyası bulunamadı!")
        
        # Boş transkriptleri filtrele
        df = df[df['transcript'].notna() & (df['transcript'].str.strip() != '')]
        
        dataset = Dataset.from_pandas(df)
        dataset = dataset.cast_column("file_path", Audio(sampling_rate=config.ORNEKLEME_ORANI, decode=False))
        
        print(f"   📈 {split.upper()} veri seti boyutu: {len(dataset)} kayıt")
        return dataset

    def evaluate_model(self, eval_dataloader, accelerator):
        """Validation setinde modeli değerlendirir."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_references = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
                
                # WER/CER hesaplama için tahminler
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                predictions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                
                # Referans metinleri
                label_ids = batch["labels"]
                label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
                references = self.processor.batch_decode(label_ids, skip_special_tokens=True)
                
                all_predictions.extend(predictions)
                all_references.extend(references)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # WER ve CER hesapla
        wer = self.wer_metric.compute(predictions=all_predictions, references=all_references)
        cer = self.cer_metric.compute(predictions=all_predictions, references=all_references)
        
        self.model.train()
        return avg_loss, wer, cer

    def train_model(self, train_dataset, eval_dataset=None, reporting_service=None, 
                   sys_info=None, start_datetime=None, training_start_time=None):
        """Model eğitimini başlatır."""
        print("🚀 Kişiselleştirilmiş model eğitimi başlıyor...")
        print(f"   Epoch sayısı: {config.NUM_FINETUNE_EPOCHS}")
        print(f"   Batch size: {config.FINETUNE_BATCH_SIZE}")
        print(f"   Learning rate: {config.FINETUNE_LEARNING_RATE}")
        print(f"   Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"   Augmentation: {'Aktif' if config.USE_AUGMENTATION and AUDIOMENTATIONS_AVAILABLE else 'Pasif'}")
        print(f"   Validation: {'Aktif' if eval_dataset is not None else 'Pasif'}")
        
        # Initialize training metrics
        self.final_train_loss = None
        self.final_wer = None
        self.final_cer = None
        self.best_wer = float('inf')
        self.best_cer = float('inf')
        self.epochs_completed = 0
        self.early_stopped = False

        # Augmentation pipeline oluştur
        augmenter = None
        if config.USE_AUGMENTATION and AUDIOMENTATIONS_AVAILABLE:
            augmenter = build_augment_pipeline(config.ORNEKLEME_ORANI)

        # Veri ön işleme (sistem kaynaklarına göre optimize)
        num_proc = min(config.DATA_PREPROCESSING_NUM_PROC, os.cpu_count() or 1)
        print(f"\n⚙️  Veri ön işleme {num_proc} CPU çekirdeği ile paralelleştiriliyor...")
        
        try:
            # Training set preprocessing (with augmentation)
            processed_train_dataset = train_dataset.map(
                _standalone_preprocess_function,
                fn_kwargs={"processor": self.processor, "augmenter": augmenter},
                remove_columns=train_dataset.column_names,
                batched=True,
                batch_size=config.FINETUNE_BATCH_SIZE,
                num_proc=num_proc
            )
            
            # Boş örnekleri filtrele
            processed_train_dataset = processed_train_dataset.filter(
                lambda x: len(x.get("input_values", [])) > 0 and len(x.get("labels", [])) > 0
            )
            
            if len(processed_train_dataset) == 0:
                raise ValueError("❌ Hata: Ön işleme sonrası hiç geçerli örnek kalmadı!")
            
            print(f"   ✅ Training set ön işleme tamamlandı. {len(processed_train_dataset)} geçerli örnek.")
            
            # Validation set preprocessing (no augmentation)
            processed_eval_dataset = None
            eval_dataloader = None
            if eval_dataset is not None:
                processed_eval_dataset = eval_dataset.map(
                    _standalone_preprocess_function,
                    fn_kwargs={"processor": self.processor, "augmenter": None},
                    remove_columns=eval_dataset.column_names,
                    batched=True,
                    batch_size=config.FINETUNE_BATCH_SIZE,
                    num_proc=num_proc
                )
                
                processed_eval_dataset = processed_eval_dataset.filter(
                    lambda x: len(x.get("input_values", [])) > 0 and len(x.get("labels", [])) > 0
                )
                
                if len(processed_eval_dataset) > 0:
                    print(f"   ✅ Validation set ön işleme tamamlandı. {len(processed_eval_dataset)} geçerli örnek.")
                else:
                    print(f"   ⚠️  Validation seti boş, validation atlanacak.")
                    processed_eval_dataset = None
            
        except Exception as e:
            print(f"❌ Veri ön işleme hatası: {e}")
            import traceback
            traceback.print_exc()
            return

        # Data collator ve dataloader
        data_collator = DataCollatorCTCWithPadding(processor=self.processor)

        # RTX A5000 için optimize edilmiş DataLoader ayarları
        train_dataloader = DataLoader(
            processed_train_dataset,
            batch_size=config.FINETUNE_BATCH_SIZE,
            collate_fn=data_collator,
            shuffle=True,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY,
            prefetch_factor=config.DATALOADER_PREFETCH_FACTOR if config.DATALOADER_NUM_WORKERS > 0 else None,
            persistent_workers=True if config.DATALOADER_NUM_WORKERS > 0 else False
        )
        
        if processed_eval_dataset is not None:
            eval_dataloader = DataLoader(
                processed_eval_dataset,
                batch_size=config.FINETUNE_BATCH_SIZE,
                collate_fn=data_collator,
                shuffle=False,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY,
                prefetch_factor=config.DATALOADER_PREFETCH_FACTOR if config.DATALOADER_NUM_WORKERS > 0 else None,
                persistent_workers=True if config.DATALOADER_NUM_WORKERS > 0 else False
            )

        # Optimizer with warmup
        optimizer = AdamW(
            self.model.parameters(), 
            lr=config.FINETUNE_LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler (cosine veya linear; config.LR_SCHEDULER_TYPE ile seçilir)
        num_training_steps = config.NUM_FINETUNE_EPOCHS * len(train_dataloader)
        warmup_steps = config.WARMUP_STEPS
        scheduler_type = getattr(config, "LR_SCHEDULER_TYPE", "cosine")
        if scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        print(f"   LR Scheduler: {scheduler_type}")
        
        # Accelerator (RTX A5000 için optimize edilmiş ayarlar)
        mixed_precision_mode = config.MIXED_PRECISION if torch.cuda.is_available() else "no"
        if mixed_precision_mode == "fp16" and not torch.cuda.is_available():
            print("⚠️  FP16 seçildi ancak CUDA yok, mixed precision kapatılıyor.")
            mixed_precision_mode = "no"
        
        accelerator = Accelerator(
            mixed_precision=mixed_precision_mode,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS
        )
        
        if torch.cuda.is_available():
            print(f"✅ GPU kullanılıyor: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"   Mixed Precision: {mixed_precision_mode}")
            print(f"   Batch Size: {config.FINETUNE_BATCH_SIZE}")
            print(f"   Gradient Accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
            print(f"   Effective Batch Size: {config.FINETUNE_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
        else:
            print("⚠️  CUDA yok, CPU kullanılıyor.")
        
        self.model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            self.model, optimizer, train_dataloader, scheduler
        )
        
        if eval_dataloader is not None:
            eval_dataloader = accelerator.prepare(eval_dataloader)

        # Eğitim döngüsü
        num_epochs = config.NUM_FINETUNE_EPOCHS
        progress_bar = tqdm(range(num_training_steps), desc="Eğitim")
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            global_step = 0
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                self.model.train()
                for step, batch in enumerate(train_dataloader):
                    with accelerator.accumulate(self.model):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        
                        # Loss kontrolü (negatif veya invalid loss için)
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"⚠️  Epoch {epoch+1}, Step {step}: Geçersiz loss ({loss.item()}), batch atlanıyor.")
                            continue
                        
                        if loss.item() < 0:
                            print(f"⚠️  Epoch {epoch+1}, Step {step}: Negatif loss ({loss.item()}), batch atlanıyor.")
                            # Debug için batch bilgilerini yazdır (sadece ilk batch'te)
                            if step == 0 and epoch == 0:
                                print(f"   Debug - Batch keys: {list(batch.keys())}")
                                print(f"   Debug - Input shape: {batch['input_values'].shape}")
                                print(f"   Debug - Labels shape: {batch['labels'].shape}")
                                print(f"   Debug - Labels min/max: {batch['labels'].min().item()}/{batch['labels'].max().item()}")
                                print(f"   Debug - Labels sample (first 10): {batch['labels'][0][:10].tolist()}")
                                # İlk birkaç label'ı decode et
                                try:
                                    sample_labels = batch['labels'][0].clone()
                                    sample_labels[sample_labels == -100] = self.processor.tokenizer.pad_token_id
                                    decoded = self.processor.tokenizer.decode(sample_labels[:20], skip_special_tokens=False)
                                    print(f"   Debug - Decoded sample: {decoded}")
                                except Exception as e:
                                    print(f"   Debug - Decode hatası: {e}")
                            continue
                        
                        accelerator.backward(loss)
                        
                        # Gradient clipping (stabilite için)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(self.model.parameters(), max_norm=config.MAX_GRAD_NORM)
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    global_step += 1
                    
                    progress_bar.update(1)
                    avg_loss = epoch_loss / num_batches
                    current_lr = optimizer.param_groups[0]['lr']
                    progress_bar.set_description(
                        f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                    )
                    
                    # Validation (belirli adımlarda)
                    if eval_dataloader is not None and global_step % config.FINETUNE_EVAL_STEPS == 0:
                        val_loss, wer, cer = self.evaluate_model(eval_dataloader, accelerator)
                        print(f"\n   Validation (Step {global_step}):")
                        print(f"      Loss: {val_loss:.4f} | WER: {wer:.4f} ({wer*100:.2f}%) | CER: {cer:.4f} ({cer*100:.2f}%)")

                        # WER-bazlı en iyi model seçimi ve early stopping
                        if wer < self.best_val_wer:
                            self.best_val_wer = wer
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            unwrapped_model = accelerator.unwrap_model(self.model)
                            best_checkpoint = self.checkpoint_dir / "best_model"
                            save_model_and_processor(unwrapped_model, self.processor, str(best_checkpoint))
                            print(f"      En iyi model kaydedildi! (WER: {wer*100:.2f}%)")
                        else:
                            self.patience_counter += 1
                            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                                print(f"\n   Early stopping tetiklendi! (Patience: {config.EARLY_STOPPING_PATIENCE})")
                                print(f"      En iyi WER: {self.best_val_wer*100:.2f}%")
                                self.early_stopped = True
                                break
                
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                self.final_train_loss = avg_epoch_loss
                self.epochs_completed = epoch + 1
                print(f"\n   Epoch {epoch+1}/{num_epochs} tamamlandı. Ortalama Loss: {avg_epoch_loss:.4f}")
                
                # Epoch sonunda validation
                if eval_dataloader is not None:
                    val_loss, wer, cer = self.evaluate_model(eval_dataloader, accelerator)
                    print(f"   Epoch sonu Validation:")
                    print(f"      Loss: {val_loss:.4f} | WER: {wer:.4f} ({wer*100:.2f}%) | CER: {cer:.4f} ({cer*100:.2f}%)")

                    if wer < self.best_val_wer:
                        self.best_val_wer = wer
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        unwrapped_model = accelerator.unwrap_model(self.model)
                        best_checkpoint = self.checkpoint_dir / "best_model"
                        save_model_and_processor(unwrapped_model, self.processor, str(best_checkpoint))
                        print(f"      En iyi model kaydedildi! (WER: {wer*100:.2f}%)")
                    else:
                        self.patience_counter += 1

                    if wer < self.best_wer:
                        self.best_wer = wer
                    if cer < self.best_cer:
                        self.best_cer = cer
                
                # Early stopping kontrolü
                if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    break

            print("\n✅ Model ince ayarı tamamlandı!")
            
            # En iyi modeli yükle ve kaydet
            if (self.checkpoint_dir / "best_model").exists():
                print(f"   📥 En iyi model yükleniyor...")
                from peft import PeftModel
                base_model = Wav2Vec2ForCTC.from_pretrained(self.base_model_path)
                best_model = PeftModel.from_pretrained(base_model, str(self.checkpoint_dir / "best_model"))
                unwrapped_model = accelerator.unwrap_model(best_model)
            else:
                unwrapped_model = accelerator.unwrap_model(self.model)
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_model_and_processor(unwrapped_model, self.processor, str(self.output_dir))

            print(f"💾 Kişiselleştirilmiş model kaydedildi: {self.output_dir}")
            print(f"   Toplam eğitim adımı: {global_step}")
            if eval_dataloader is not None:
                final_val_loss, final_wer, final_cer = self.evaluate_model(eval_dataloader, accelerator)
                self.final_wer = final_wer
                self.final_cer = final_cer
                print(f"   Final Validation - Loss: {final_val_loss:.4f} | WER: {final_wer:.4f} ({final_wer*100:.2f}%) | CER: {final_cer:.4f} ({final_cer*100:.2f}%)")
            
            # Create training report
            if reporting_service:
                training_end_time = time.time()
                duration_seconds = training_end_time - training_start_time
                hours = int(duration_seconds // 3600)
                minutes = int((duration_seconds % 3600) // 60)
                seconds = int(duration_seconds % 60)
                duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
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
                    "use_augmentation": config.USE_AUGMENTATION and AUDIOMENTATIONS_AVAILABLE,
                    "mixed_precision": getattr(config, "MIXED_PRECISION", "no"),
                    "train_samples": self.train_samples,
                    "eval_samples": self.eval_samples,
                    "total_samples": self.train_samples + self.eval_samples,
                    "total_steps": global_step,
                    "epochs_completed": self.epochs_completed,
                    "final_train_loss": self.final_train_loss,
                    "best_val_loss": self.best_val_loss,
                    "final_wer": self.final_wer,
                    "final_cer": self.final_cer,
                    "best_wer": self.best_wer if self.best_wer != float('inf') else None,
                    "best_cer": self.best_cer if self.best_cer != float('inf') else None,
                    "early_stopped": self.early_stopped,
                    "device": self.device,
                    "gpu_name": sys_info.get("gpu_name") if sys_info else None,
                    "gpu_memory_gb": sys_info.get("gpu_memory_gb") if sys_info else None,
                    "platform": sys_info.get("platform") if sys_info else None,
                    "python_version": sys_info.get("python_version") if sys_info else None,
                    "start_time": start_datetime.isoformat() if start_datetime else None,
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": duration_seconds,
                    "duration_formatted": duration_formatted
                }
                
                report_file = reporting_service.log_training_session(self.user_id, training_data)
                print(f"\n📊 Training report saved: {report_file}")
            
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
    import platform
    import sys
    
    # Linux sunucu için multiprocessing optimizasyonu
    if platform.system() == "Linux":
        import multiprocessing as mp
        try:
            # Linux'ta fork daha hızlı ve verimli
            mp.set_start_method(config.MULTIPROCESSING_START_METHOD, force=True)
            print(f"✅ Linux sunucu: Multiprocessing start method = {config.MULTIPROCESSING_START_METHOD}")
        except RuntimeError:
            # Zaten ayarlanmışsa devam et
            pass
    else:
        # Windows/Mac için spawn
        if torch.cuda.is_available():
            try:
                import multiprocessing as mp
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass
    
    # CUDA device seçimi (Linux sunucuda birden fazla GPU varsa)
    if torch.cuda.is_available() and config.CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.CUDA_VISIBLE_DEVICES)
        print(f"✅ CUDA_VISIBLE_DEVICES = {config.CUDA_VISIBLE_DEVICES}")

    main()