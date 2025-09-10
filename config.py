# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Yapılandırma Dosyası
"""

# --- ASR Model Ayarları ---
MODEL_NAME = "facebook/wav2vec2-large-960h"  # Varsayılan model
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
FINETUNE_OUTPUT_DIR = "./asr_model_finetuned"  # İnce ayarlı modelin kaydedileceği dizin
FINETUNE_BATCH_SIZE = 8  # İnce ayar için batch boyutu
NUM_FINETUNE_EPOCHS = 3  # İnce ayar için epoch sayısı
FINETUNE_EVAL_STEPS = 500  # Değerlendirme adımları
FINETUNE_LOGGING_STEPS = 100  # Loglama adımları
FINETUNE_LEARNING_RATE = 1e-4  # İnce ayar için öğrenme oranı