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

# --- Ses Kayıt Ayarları ---
KAYIT_SURESI_SN = 5  # Varsayılan kayıt süresi (saniye)
SES_ESIK_DEGERI = 0.01  # Ses aktivitesi için hassasiyet eşiği

# --- Sistem Ayarları ---
GECICI_DOSYA_YOLU = "temp_recording.wav"  # Geçici ses dosyası

# --- Veri Yolları Ayarları ---
BASE_PATH = "data/users"  # Kullanıcı verilerinin depolanacağı ana dizin
USER_ID = "default_user"  # Varsayılan kullanıcı ID'si (kullanıcıya özel veriler için)

# --- Model İnce Ayar Ayarları ---
# Optimum performans için ayarlandı (küçük kişisel veri setleri için)
FINETUNE_OUTPUT_DIR = "./asr_model_finetuned"  # İnce ayarlı modelin kaydedileceği dizin
FINETUNE_BATCH_SIZE = 4  # İnce ayar için batch boyutu (küçük veri setleri için 2 veya 4 idealdir)
NUM_FINETUNE_EPOCHS = 15  # İnce ayar için epoch sayısı (küçük veri setleri için 10-15 arası önerilir)
FINETUNE_EVAL_STEPS = 500  # Değerlendirme adımları (şu an kullanılmıyor)
FINETUNE_LOGGING_STEPS = 10  # Loglama sıklığı (küçük veri setinde daha sık loglama faydalıdır)
FINETUNE_LEARNING_RATE = 5e-5  # İnce ayar için öğrenme oranı (daha stabil öğrenme için)
ADAPTER_REDUCTION_FACTOR = 8  # Adapter'ın öğrenme kapasitesi. Düşük değer = daha fazla parametre, potansiyel olarak daha yüksek doğruluk.