# -*- coding: utf-8 -*-
"""
Konuşma Bozukluğu Ses Tanıma Sistemi - Sabitler
Config'den ayrı, değişmeyen sabit değerler
"""

# --- Dosya Uzantıları ---
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a']
METADATA_EXTENSION = '.csv'
TEXT_EXTENSION = '.txt'

# --- Dosya Adlandırma Kalıpları ---
WORD_FILE_PATTERN = "{word}/rep{rep_num}.wav"
SENTENCE_FILE_PATTERN = "{user_id}_cümle_{file_number}_rep{rep_num}.wav"
LETTER_FILE_PATTERN = "{user_id}_harf_{file_number}_rep{rep_num}.wav"

# --- Metadata Sütunları ---
METADATA_COLUMNS = [
    "file_path",
    "transcription",
    "repetition",
    "quality_score",
    "rms",
    "snr_db",
    "duration"
]

# --- Kayıt Türleri ---
RECORD_TYPE_SENTENCE = "cümle"
RECORD_TYPE_WORD = "kelime"
RECORD_TYPE_LETTER = "harf"

RECORD_TYPES = [RECORD_TYPE_SENTENCE, RECORD_TYPE_WORD, RECORD_TYPE_LETTER]

# --- Varsayılan Tekrar Sayıları ---
# Not: IDEAL_REPETITIONS config.py'den import edilmeli
DEFAULT_REPETITIONS = {
    RECORD_TYPE_SENTENCE: 3,
    RECORD_TYPE_WORD: 10,  # IDEAL_REPETITIONS (config'den alınır)
    RECORD_TYPE_LETTER: 5
}

# --- Dosya Yolu Bileşenleri ---
USER_DATA_SUBDIRS = {
    RECORD_TYPE_SENTENCE: "audio",
    RECORD_TYPE_WORD: "words",
    RECORD_TYPE_LETTER: "letters"
}

METADATA_FILENAMES = {
    RECORD_TYPE_SENTENCE: "metadata.csv",
    RECORD_TYPE_WORD: "metadata_words.csv",
    RECORD_TYPE_LETTER: "metadata_letters.csv"
}

# --- Dataset Dizinleri ---
DATASET_DIRS = {
    RECORD_TYPE_SENTENCE: "datasets/sentence_sets",
    RECORD_TYPE_WORD: "datasets/words_set",
    RECORD_TYPE_LETTER: "datasets/letters_set"
}

# --- Model Dizinleri ---
MODEL_BASE_DIR = "data/models/personalized_models"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "best_model"

# --- Çıkış Komutları ---
EXIT_COMMANDS = ['çık', 'exit', 'quit', 'q']

# --- Platform Bilgileri ---
PLATFORM_MACOS = "Darwin"
PLATFORM_LINUX = "Linux"
PLATFORM_WINDOWS = "Windows"
