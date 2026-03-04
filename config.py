# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Yapılandırma Dosyası
"""

# --- ASR Model Ayarları ---
MODEL_NAME = "mpoyraz/wav2vec2-xls-r-300m-cv8-turkish"  # Varsayılan model
ORNEKLEME_ORANI = 16000  # Hz

# --- Dil Modeli Ayarları (Opsiyonel) ---
KENLM_MODEL_PATH = "data/lm/lm.arpa"
LM_ALPHA = 0.5       # Dil modeli ağırlığı (CTC beam search için)
LM_BETA = 1.5        # Kelime sayısı penaltısı (CTC beam search için)
LM_BEAM_WIDTH = 100  # Beam search genişliği

# --- Ses Kayıt Ayarları (Konuşma Bozukluğu için Optimize) ---
KAYIT_SURESI_SN = 4
SES_ESIK_DEGERI = 0.008
VAD_SILENCE_LIMIT_SEC = 2.5
VAD_SPEECH_WAIT_SEC = 8

# --- Ses Kalitesi Kontrol Ayarları ---
MIN_RMS_LEVEL = 300
MAX_RMS_LEVEL = 25000
MIN_DURATION_SEC = 0.2
MAX_DURATION_SEC = 8
QUALITY_THRESHOLD = 40
AUTO_RERECORD_ENABLED = True

# --- Tutarlılık Kontrol Ayarları ---
CONSISTENCY_CHECK_ENABLED = True
CONSISTENCY_TOLERANCE = 0.5

# --- Kayıt Tekrar Ayarları ---
IDEAL_REPETITIONS = 10

# --- Sistem Ayarları ---
GECICI_DOSYA_YOLU = "temp_recording.wav"

# --- Veri Yolları Ayarları ---
BASE_PATH = "data/users"
USER_ID = "default_user"

# --- Logging ve Reporting Ayarları ---
LOG_DIR = "logs"
LOG_LEVEL = "INFO"
REPORTS_DIR = "reports"

# --- Model İnce Ayar Ayarları ---
FINETUNE_OUTPUT_DIR = "./asr_model_finetuned"
FINETUNE_BATCH_SIZE = 2
NUM_FINETUNE_EPOCHS = 30
FINETUNE_EVAL_STEPS = 200
FINETUNE_LOGGING_STEPS = 10
FINETUNE_LEARNING_RATE = 2e-5

# LoRA adapter ayarları
# r=16: Konuşma bozukluğu için daha yüksek rank; bozukluk kalıplarını daha iyi öğrenir
ADAPTER_REDUCTION_FACTOR = 16

# Eğitim stabilite ayarları
GRADIENT_ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
EARLY_STOPPING_PATIENCE = 5

# LR scheduler tipi: "linear" veya "cosine"
LR_SCHEDULER_TYPE = "cosine"

# --- Augmentation Ayarları ---
USE_AUGMENTATION = True
# Hafif augmentation; konuşma bozukluğu kalıplarını bozmamak için düşük yoğunluk
AUGMENT_NOISE_MIN = 0.0005
AUGMENT_NOISE_MAX = 0.003
AUGMENT_TIME_STRETCH_MIN = 0.9
AUGMENT_TIME_STRETCH_MAX = 1.1
AUGMENT_PITCH_MIN = -2
AUGMENT_PITCH_MAX = 2
AUGMENT_TIME_MASK_MIN = 0.02
AUGMENT_TIME_MASK_MAX = 0.08
AUGMENT_PROBABILITY = 0.5

# --- Donanım ve Performans Ayarları ---
MIXED_PRECISION = "fp16"          # "fp16", "bf16" veya "no"
GRADIENT_CHECKPOINTING = True     # VRAM tasarrufu için
DATALOADER_NUM_WORKERS = 4
DATALOADER_PIN_MEMORY = True
DATALOADER_PREFETCH_FACTOR = 2
DATA_PREPROCESSING_NUM_PROC = 4

# --- Çoklu İşlem Ayarları ---
MULTIPROCESSING_START_METHOD = "fork"  # Linux için "fork", Windows/Mac için "spawn"
CUDA_VISIBLE_DEVICES = None            # Örnek: "0" veya "0,1"

# --- Otonom Kayıt Ayarları (auto_collect.py) ---
# ENTER'a basmadan tam otomatik veri toplama için.

# Her kelime için maksimum bekleme süresi (saniye).
# Bu süre içinde geçerli kayıt alınamazsa kelime atlanır.
AUTO_WORD_TIMEOUT_SEC = 15

# Bir kelime için maksimum otomatik deneme sayısı.
AUTO_MAX_RETRIES = 3

# Kelimeler arası bekleme süresi (saniye); kullanıcıya hazırlanma süresi.
AUTO_INTER_WORD_PAUSE_SEC = 1.5

# VAD: kelime kaydı için kısa sessizlik limiti (cümle kaydından daha kısa).
AUTO_SILENCE_LIMIT_SEC = 1.0

# VAD: ilk ses gelene kadar bekleme süresi.
AUTO_SPEECH_WAIT_SEC = 4.0

# VAD: ses başlangıç eşiği (normalize, 0-1 arası).
# Yüksek değer = yalnızca güçlü ses tetikler (arka plan gürültüsünü engeller).
AUTO_SOUND_THRESHOLD = 0.012

# Minimum konuşma çerçevesi oranı (0-1).
# Kaydın bu oranından az ses içeriyorsa "boş kayıt" sayılır ve reddedilir.
AUTO_MIN_SPEECH_RATIO = 0.10

# ASR doğrulaması: True ise kayıt edilen ses, beklenen kelimeyle karşılaştırılır.
# Speech-impaired kullanıcılarda False yapın; ASR doğru tanıyamayabilir.
AUTO_ASR_VERIFY = False

# ASR doğrulaması için maksimum karakter hata oranı (0-1).
# 0 = tam eşleşme zorunlu, 1 = her şeyi kabul et.
# Konuşma bozukluğu için 0.5-0.7 arası önerilir.
AUTO_ASR_MAX_CER = 0.6

# Ekranda büyük gösterilecek hedef kelime için terminal genişliği.
AUTO_DISPLAY_WIDTH = 60

# Oturum sonu özet raporu kaydet.
AUTO_SAVE_REPORT = True

# --- Sentez (Synthesis) Ayarları ---
# Kaydedilmiş kelimelerden cümle sentezi için
SYNTHESIS_MIN_QUALITY = 50       # Sentez için kullanılacak minimum kalite skoru
SYNTHESIS_CROSSFADE_MS = 30      # Kelimeler arası crossfade süresi (ms)
SYNTHESIS_PAUSE_MS = 150         # Kelimeler arası duraklatma süresi (ms)
SYNTHESIS_BEST_K = 3             # Her kelime için en iyi K kayıt seçimi
