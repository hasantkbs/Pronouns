# config.py - Speech Disorder ASR Training Configuration (OPTIMIZED)

import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_NAME = "your_base_model_path"  # Base Wav2Vec2 model
ORNEKLEME_ORANI = 16000  # Sampling rate

# ============================================================================
# TRAINING HYPERPARAMETERS (OPTIMIZED FOR SPEECH DISORDERS)
# ============================================================================

# CRITICAL: Reduced LoRA rank to prevent overfitting
ADAPTER_REDUCTION_FACTOR = 8  # Changed from 16 to 8 (fewer trainable params)

# Learning rate - significantly reduced
FINETUNE_LEARNING_RATE = 5e-6  # Changed from likely 1e-4 to 5e-6

# Batch size
FINETUNE_BATCH_SIZE = 4  # Keep small for stability

# Epochs - reduced to prevent overfitting
NUM_FINETUNE_EPOCHS = 10  # Changed from 20 to 10

# Gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 32

# Weight decay - increased for regularization
WEIGHT_DECAY = 0.1  # Changed from 0.01 to 0.1

# Gradient clipping
MAX_GRAD_NORM = 0.5  # Changed from 1.0 to 0.5 (more aggressive)

# Warmup steps
WARMUP_STEPS = 200  # Increased for smoother training

# ============================================================================
# VALIDATION & EARLY STOPPING
# ============================================================================
FINETUNE_EVAL_STEPS = 50  # Validate more frequently
EARLY_STOPPING_PATIENCE = 5  # Stop earlier if no improvement

# ============================================================================
# DATA AUGMENTATION (MORE CONSERVATIVE)
# ============================================================================
USE_AUGMENTATION = True

# Augmentation probabilities (reduced)
AUG_NOISE_PROB = 0.2  # Changed from 0.3
AUG_TIME_STRETCH_PROB = 0.2  # Changed from 0.3
AUG_PITCH_SHIFT_PROB = 0.2  # Changed from 0.3
AUG_TIME_MASK_PROB = 0.1  # Changed from 0.2

# Augmentation intensity (reduced)
AUG_NOISE_MIN = 0.0002  # Changed from 0.0005
AUG_NOISE_MAX = 0.003  # Changed from 0.005
AUG_TIME_STRETCH_MIN = 0.95  # Changed from 0.9
AUG_TIME_STRETCH_MAX = 1.05  # Changed from 1.1
AUG_PITCH_SHIFT_MIN = -1  # Changed from -2
AUG_PITCH_SHIFT_MAX = 1  # Changed from 2

# Overall augmentation probability
AUGMENTATION_PROB = 0.4  # Changed from 0.6

# ============================================================================
# DROPOUT & REGULARIZATION
# ============================================================================
LORA_DROPOUT = 0.1  # Increased from 0.05 to 0.1
ATTENTION_DROPOUT = 0.1  # Add if model supports
HIDDEN_DROPOUT = 0.1  # Add if model supports

# ============================================================================
# MIXED PRECISION & OPTIMIZATION
# ============================================================================
MIXED_PRECISION = "fp16"  # fp16, bf16, or no
GRADIENT_CHECKPOINTING = True

# ============================================================================
# DATA LOADING (RTX A5000 OPTIMIZED)
# ============================================================================
DATA_PREPROCESSING_NUM_PROC = 8
DATALOADER_NUM_WORKERS = 4
DATALOADER_PIN_MEMORY = True
DATALOADER_PREFETCH_FACTOR = 2

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================
MULTIPROCESSING_START_METHOD = "fork"  # Linux only
CUDA_VISIBLE_DEVICES = "0"  # GPU selection

# ============================================================================
# PATHS
# ============================================================================
BASE_PATH = "data/users"
LOG_DIR = "logs"
LOG_LEVEL = "INFO"