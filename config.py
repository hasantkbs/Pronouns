# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Yapılandırma Dosyası
"""

# --- ASR Model Ayarları ---
MODEL_NAME = "mpoyraz/wav2vec2-xls-r-300m-cv8-turkish"  # Varsayılan model (Wav2Vec2/CTC)
WHISPER_MODEL_NAME = "selimc/whisper-large-v3-turbo-turkish"  # Whisper kişiselleştirme başlangıç noktası
ORNEKLEME_ORANI = 16000  # Hz

# Eğitim reprodüsibilitesi: ağırlık başlangıcı, DataLoader karıştırma ve
# augmentation aynı seed ile tekrarlanabilir olsun diye sabitlendi — sabit
# seed olmadan aynı config+veri ile run'lar arası WER %65 ile %89 arasında
# oynayabiliyordu (2026-08-17).
SEED = 42

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
IDEAL_REPETITIONS = 5
SENTENCE_REPETITIONS = 2  # Cumle olcum kaydi: her cumle 2 kez okunur

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
FINETUNE_EVAL_STEPS = 100
FINETUNE_LOGGING_STEPS = 10
FINETUNE_LEARNING_RATE = 3e-5      # Biraz daha yüksek LR
WARMUP_STEPS = 100                 # Öğrenme oranı ısınma adımları
WEIGHT_DECAY = 0.01                # Ağırlık sönümü (overfitting engelleme)
MAX_GRAD_NORM = 1.0                # Gradyan kırpma (stabilite için)
# LoRA adapter ayarları
# reduction_factor = hidden_size / bottleneck_dim: DÜŞÜK değer DAHA FAZLA kapasite
# demektir. 32 denendi (4 kat daha fazla eğitilebilir parametre) ama sonuç
# WER %94.2 ile 128'in ürettiği en iyi modelden (WER %65.0, 2026-08-11) belirgin
# biçimde kötü çıktı — küçük veri setinde artan kapasite overfit/instabiliteye yol açtı.
# 128'e geri alındı.
ADAPTER_REDUCTION_FACTOR = 128

# Eğitim stabilite ayarları
EARLY_STOPPING_PATIENCE = 10       # 30 -> 10: Whisper epoch başına uzun sürüyor
NUM_FINETUNE_EPOCHS = 100          # Maksimum limit
FINETUNE_BATCH_SIZE = 8            # 4'ten 8'e çıkarıldı
GRADIENT_ACCUMULATION_STEPS = 4    # Toplam batch size 32 (8 * 4) korunuyor

# --- Anlamli Cumle Kurma Ayarlari ---
# Kelime tamamlama ve duzeltme esigi (0-1)
# 0.6: %60 benzerlik gosteren kelimeyi kullanicinin sozrugunden "en yakin" olanla degistirir.
FUZZY_MATCH_THRESHOLD = 0.6
LEXICON_PATH = "data/lexicon/tr_50k.txt"  # Genel Turkce siklik sozlugu (cumle duzeltici)
PIPER_VOICE_PATH = "data/models/tts/tr_TR-dfki-medium.onnx"  # scripts/download_piper_voice.sh indirir
WHISPER_PROMPT_BIASING = False  # Sozluk ipucu: ipucusuz egitilen modelde dogrulugu dusuruyor
USE_LANGUAGE_MODEL = True          # KenLM aktivasyonu

# --- Augmentation Ayarları ---
USE_AUGMENTATION = True
# Hafif augmentation; konuşma bozukluğu kalıplarını bozmamak için düşük yoğunluk
AUGMENT_NOISE_MIN = 0.0005
AUGMENT_NOISE_MAX = 0.003
AUGMENT_TIME_STRETCH_MIN = 0.7
AUGMENT_TIME_STRETCH_MAX = 1.3
AUGMENT_PITCH_MIN = -2
AUGMENT_PITCH_MAX = 2
AUGMENT_TIME_MASK_MIN = 0.02
AUGMENT_TIME_MASK_MAX = 0.08
AUGMENT_PROBABILITY = 0.7
# --- Donanım ve Performans Ayarları ---
MIXED_PRECISION = "fp16"          # "fp16", "bf16" veya "no"
GRADIENT_CHECKPOINTING = True     # VRAM tasarrufu için aktifleştirildi (Özellikle Whisper için kritik)
DATALOADER_NUM_WORKERS = 8        # 48 çekirdek için optimize edildi
DATALOADER_PIN_MEMORY = True
DATALOADER_PREFETCH_FACTOR = 4
DATA_PREPROCESSING_NUM_PROC = 1   # 2'den 1'e düşürüldü (Maksimum stabilite için)

# --- Çoklu İşlem Ayarları ---
MULTIPROCESSING_START_METHOD = "fork"   # Linux'ta daha hızlı
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

# --- API Güvenliği ---
# Mobil uygulama tüm isteklerde bu değeri `X-API-Key` header'ında gönderir.
# lib/main.dart içindeki _kApiKey ile birebir aynı olmalı.
API_SECRET_KEY = "c2a6dd1afc678eb93f0c46c8c1528e2b1c07676fbde0951f"

# --- Konuşmacı Doğrulama (Speaker Verification) Ayarları ---
# /translate'e gelen ses segmentinin gerçekten kayıtlı kullanıcıya mı ait
# olduğunu kontrol eder. False yapılırsa özellik tamamen devre dışı kalır.
SPEAKER_VERIFY_ENABLED = True
# Dokun-konus akisinda dugmeye Furkan basar; dogrulama karar vermez, sadece loglar.
# True yapilirsa eslesmeyen ses eskisi gibi reddedilir.
SPEAKER_VERIFY_BLOCKING = False
SPEAKER_VERIFY_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
# Kosinüs benzerliği eşiği (0-1). FurkanV1 (40 örnekli profil) ile Admin
# klasöründeki farklı konuşmacının kayıtları karşılaştırılarak kalibre
# edildi (2026-08-10): Admin skorları 0.242-0.480, Furkan'ın kendi sesi
# 0.557-0.800 arasında çıktı — aralarında boşluk var, eşik bu boşluğun
# ortasına yakın seçildi (yanlış reddi en aza indirmek için Furkan tarafına
# yakın).
SPEAKER_VERIFY_THRESHOLD = 0.50
# Referans profil oluşturulurken kullanılacak azami kayıt sayısı.
SPEAKER_ENROLL_MAX_SAMPLES = 40
