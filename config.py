# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Yapılandırma Dosyası
"""

# --- ASR Model Ayarları ---
MODEL_NAME = "mpoyraz/wav2vec2-xls-r-300m-cv8-turkish"  # Varsayılan model
ORNEKLEME_ORANI = 16000  # Hz

# --- Dil Modeli Ayarları (Opsiyonel) ---
# Daha yüksek doğruluk için KenLM dil modeli kullanılabilir
KENLM_MODEL_PATH = "data/lm/lm.arpa"

# --- Ses Kayıt Ayarları (Konuşma Bozukluğu için Optimize) ---
KAYIT_SURESI_SN = 4  # Varsayılan kayıt süresi (saniye) - kelime için yeterli
SES_ESIK_DEGERI = 0.008  # Ses aktivitesi için hassasiyet eşiği (daha düşük = daha sessiz sesleri algılar)
VAD_SILENCE_LIMIT_SEC = 2.5  # Konuşma bittikten sonra beklenen sessizlik süresi (konuşma bozukluğu için daha uzun)
VAD_SPEECH_WAIT_SEC = 8  # Konuşma beklenen maksimum süre (yavaş konuşma için)

# --- Ses Kalitesi Kontrol Ayarları (Konuşma Bozukluğu için Toleranslı) ---
MIN_RMS_LEVEL = 300  # Minimum RMS seviyesi (daha düşük sesleri kabul et)
MAX_RMS_LEVEL = 25000  # Maksimum RMS seviyesi (daha yüksek tolerans)
MIN_DURATION_SEC = 0.2  # Minimum kayıt süresi (çok kısa kelimeler için)
MAX_DURATION_SEC = 8  # Maksimum kayıt süresi (kelime için 4 sn + tolerans)
QUALITY_THRESHOLD = 40  # Minimum kalite skoru (konuşma bozukluğu için daha toleranslı)
AUTO_RERECORD_ENABLED = True  # Otomatik yeniden kayıt önerisi aktif mi?

# --- Tutarlılık Kontrol Ayarları ---
CONSISTENCY_CHECK_ENABLED = True  # Aynı kelime için süre tutarlılığı kontrolü
CONSISTENCY_TOLERANCE = 0.5  # Aynı kelime için süre farkı toleransı (saniye)

# --- Kayıt Tekrar Ayarları (Konuşma Bozukluğu için) ---
# Araştırmalara göre konuşma bozukluğu için 5-10 kayıt ideal
# 10 kayıt: İyi denge (yeterli veri + kullanıcı yorgunluğu düşük)
IDEAL_REPETITIONS = 10  # Her kelime için ideal kayıt sayısı

# --- Sistem Ayarları ---
GECICI_DOSYA_YOLU = "temp_recording.wav"  # Geçici ses dosyası

# --- Veri Yolları Ayarları ---
BASE_PATH = "data/users"  # Kullanıcı verilerinin depolanacağı ana dizin
USER_ID = "default_user"  # Varsayılan kullanıcı ID'si (kullanıcıya özel veriler için)

# --- Logging ve Reporting Ayarları ---
LOG_DIR = "logs"  # Log dosyalarının kaydedileceği dizin
LOG_LEVEL = "INFO"  # Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
REPORTS_DIR = "reports"  # Rapor dosyalarının kaydedileceği dizin

# --- Model İnce Ayar Ayarları ---
# Optimum performans için ayarlandı (küçük kişisel veri setleri için)
FINETUNE_OUTPUT_DIR = "./asr_model_finetuned"  # İnce ayarlı modelin kaydedileceği dizin
FINETUNE_BATCH_SIZE = 2  # İnce ayar için batch boyutu (küçük veri setleri için 2 veya 4 idealdir)
NUM_FINETUNE_EPOCHS = 30  # İnce ayar için epoch sayısı (küçük veri setleri için 10-15 arası önerilir)
FINETUNE_EVAL_STEPS = 500  # Değerlendirme adımları (şu an kullanılmıyor)
FINETUNE_LOGGING_STEPS = 10  # Loglama sıklığı (küçük veri setinde daha sık loglama faydalıdır)
FINETUNE_LEARNING_RATE = 2e-5  # İnce ayar için öğrenme oranı (daha stabil öğrenme için)
ADAPTER_REDUCTION_FACTOR = 8  # Adapter'ın öğrenme kapasitesi. Düşük değer = daha fazla parametre, potansiyel olarak daha yüksek doğruluk.
GRADIENT_ACCUMULATION_STEPS = 2