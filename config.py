# -*- coding: utf-8 -*-

# --- Genel Ayarlar ---
USER_ID = "speech_commands_test"
BASE_PATH = "users"

# --- Embedding Model Ayarları ---
MODEL_NAME = "facebook/wav2vec2-large-960h"
ORNEKLEME_ORANI = 16000

# --- Sınıflandırıcı Eğitim Ayarları ---
EMBEDDING_DIM = 1024  # Wav2Vec2-Base modelinin çıktı boyutu
HIDDEN_DIM = 256
NUM_EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 0.001
TEST_SIZE = 0.35 # Test verisi oranı
MIN_SAMPLES_PER_CLASS = 3 # Bir sınıfın eğitime dahil edilmesi için gereken minimum örnek sayısı

# --- Gerçek Zamanlı Tanıma Ayarları ---
KAYIT_SURESI_SN = 2  # Saniye cinsinden kayıt süresi
SES_ESIK_DEGERI = 0.01 # Ses aktivitesi için hassasiyet eşiği