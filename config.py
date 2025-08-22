# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Yapılandırma Dosyası
"""

# --- ASR Model Ayarları ---
MODEL_NAME = "facebook/wav2vec2-large-960h"  # Varsayılan model
ORNEKLEME_ORANI = 16000  # Hz

# --- Dil Modeli Ayarları (Opsiyonel) ---
# Daha yüksek doğruluk için KenLM dil modeli kullanılabilir
KENLM_MODEL_PATH = "data/models/language_model/lm.binary"

# --- Ses Kayıt Ayarları ---
KAYIT_SURESI_SN = 5  # Varsayılan kayıt süresi (saniye)
SES_ESIK_DEGERI = 0.01  # Ses aktivitesi için hassasiyet eşiği

# --- Sistem Ayarları ---
GECICI_DOSYA_YOLU = "temp_recording.wav"  # Geçici ses dosyası