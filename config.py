# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Yapılandırma Dosyası
"""

# --- ASR Model Ayarları ---
# Wav2Vec2 tabanlı Türkçe ASR modeli
MODEL_NAME = "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish"  # Varsayılan model
ORNEKLEME_ORANI = 16000  # Hz (Wav2Vec2 için standart örnekleme oranı)

# --- Dil Modeli Ayarları (Opsiyonel) ---
# Daha yüksek doğruluk için KenLM dil modeli kullanılabilir
KENLM_MODEL_PATH = "data/lm/lm.arpa"

# --- Ses Kayıt Ayarları ---
KAYIT_SURESI_SN = 5  # Varsayılan kayıt süresi (saniye)
SES_ESIK_DEGERI = 0.01  # Ses aktivitesi için hassasiyet eşiği

# --- Sistem Ayarları ---
GECICI_DOSYA_YOLU = "temp_recording.wav"  # Geçici ses dosyası

# --- Veri Yolları Ayarları ---
BASE_PATH = "data/users"  # Kullanıcı verilerinin depolanacağı ana dizin
USER_ID = "default_user"  # Varsayılan kullanıcı ID'si (kullanıcıya özel veriler için)

# --- Model İnce Ayar Ayarları ---
# RTX A5000 (24GB VRAM) + 48 CPU çekirdek için optimize edilmiş ayarlar
FINETUNE_OUTPUT_DIR = "./asr_model_finetuned"  # İnce ayarlı modelin kaydedileceği dizin
FINETUNE_BATCH_SIZE = 16  # RTX A5000 için optimize edilmiş batch boyutu (24GB VRAM ile 16-32 arası)
NUM_FINETUNE_EPOCHS = 20  # İnce ayar için epoch sayısı (konuşma bozukluğu için daha fazla epoch gerekebilir)
FINETUNE_EVAL_STEPS = 50  # Değerlendirme adımları (validation için)
FINETUNE_LOGGING_STEPS = 10  # Loglama sıklığı
FINETUNE_LEARNING_RATE = 5e-5  # İnce ayar için öğrenme oranı (konuşma bozukluğu için daha düşük LR daha stabil)
ADAPTER_REDUCTION_FACTOR = 16  # Adapter'ın öğrenme kapasitesi. Konuşma bozukluğu için daha fazla parametre (16-32 arası)
GRADIENT_ACCUMULATION_STEPS = 2  # Gradient accumulation (efektif batch size = 16 * 2 = 32, RTX A5000 için optimize)
WARMUP_STEPS = 100  # Warmup adımları (öğrenme oranını kademeli artırmak için)
WEIGHT_DECAY = 1e-3  # Weight decay (overfitting'i önlemek için)
EARLY_STOPPING_PATIENCE = 5  # Early stopping patience (validation loss iyileşmezse dur)
SAVE_TOTAL_LIMIT = 3  # Maksimum checkpoint sayısı
USE_AUGMENTATION = True  # Veri augmentation kullan (konuşma bozukluğu için önemli)

# --- Sistem Performans Ayarları (RTX A5000 + 48 CPU için) ---
DATALOADER_NUM_WORKERS = 8  # DataLoader worker sayısı (48 çekirdek için optimize, her worker ~6 çekirdek)
DATALOADER_PIN_MEMORY = True  # GPU'ya daha hızlı veri transferi için
DATALOADER_PREFETCH_FACTOR = 4  # Önceden yükleme faktörü (daha hızlı eğitim için)
DATA_PREPROCESSING_NUM_PROC = 16  # Veri ön işleme için paralel işlem sayısı (48 çekirdek için optimize)
MIXED_PRECISION = "fp16"  # RTX A5000 FP16 destekliyor, 2x hız artışı + VRAM tasarrufu
GRADIENT_CHECKPOINTING = False  # RTX A5000'de yeterli VRAM var, checkpointing gerekmez
MAX_GRAD_NORM = 1.0  # Gradient clipping norm değeri

# --- Linux Sunucu Ayarları ---
MULTIPROCESSING_START_METHOD = "fork"  # Linux'ta fork daha hızlı (spawn yerine)
CUDA_VISIBLE_DEVICES = None  # None = tüm GPU'lar, "0" = sadece GPU 0, "0,1" = GPU 0 ve 1
LOG_DIR = "logs"  # Log dosyalarının kaydedileceği dizin
LOG_LEVEL = "INFO"  # Log seviyesi: DEBUG, INFO, WARNING, ERROR
ENABLE_TENSORBOARD = True  # TensorBoard logging (opsiyonel)
TENSORBOARD_LOG_DIR = "runs"  # TensorBoard log dizini