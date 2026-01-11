# Training Guide - Optimized for Speech Disorders

## 🎯 Overview

This guide describes the optimized training pipeline for personalized ASR models designed for individuals with speech disorders.

## 📋 Pre-Training Preparation

### 1. Data Preparation

Prepare data for user "Furkan":

```bash
python prepare_training_data.py Furkan
```

This command will:
- Read the main `metadata_words.csv` file
- Split the data into 80% training and 20% evaluation sets
- Create `train.csv` and `eval.csv` in the user's data directory

### 2. Data Validation

Before training, verify data quality:
- Ensure audio files are located in `data/users/Furkan/words/`
- Verify that `metadata_words.csv` is correctly formatted
- Minimum recommended data: 100+ recordings

## 🚀 Model Training

### Basic Training

```bash
python train_adapter.py Furkan
```

### Advanced Features

The following features are automatically applied during training:

1. **Data Augmentation** (if enabled in config):
   - Light Gaussian noise addition
   - Time stretching (speech rate variation)
   - Pitch shifting (light, -2 to +2 semitones)
   - Time masking (small segments)
   - Applied with 60% probability (to avoid excessive distortion)

2. **Validation**:
   - Validation performed every 50 steps
   - WER (Word Error Rate) and CER (Character Error Rate) calculated
   - Best model automatically saved

3. **Early Stopping**:
   - Training stops if validation loss doesn't improve
   - Prevents overfitting
   - Patience: 5 epochs (configurable in config.py)

4. **Learning Rate Scheduling**:
   - Warmup: Learning rate gradually increases for first 100 steps
   - Linear decay afterwards

## 📊 Training Metrics

The following metrics are tracked during training:

- **Training Loss**: Displayed at the end of each epoch
- **Validation Loss**: Calculated every 50 steps
- **WER**: Word Error Rate (lower is better)
- **CER**: Character Error Rate (lower is better)

### Good Performance Indicators

- WER < 0.15 (less than 15% word error)
- CER < 0.05 (less than 5% character error)
- Validation loss close to training loss (no overfitting)

## ⚙️ Hyperparameter Settings

### Recommended Settings (config.py)

```python
# Optimized for speech disorders
NUM_FINETUNE_EPOCHS = 20          # More epochs
FINETUNE_BATCH_SIZE = 4           # Larger batch
FINETUNE_LEARNING_RATE = 5e-5    # Lower LR (for stability)
ADAPTER_REDUCTION_FACTOR = 16    # More parameters
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 16
WARMUP_STEPS = 100                # Learning rate warmup
EARLY_STOPPING_PATIENCE = 5      # Early stopping
USE_AUGMENTATION = True           # Augmentation enabled
```

### Tuning Tips

**For low accuracy:**
- Increase number of epochs (20 → 30)
- Decrease learning rate (5e-5 → 3e-5)
- Collect more training data
- Keep augmentation enabled

**For overfitting:**
- Decrease early stopping patience (5 → 3)
- Increase weight decay (1e-3 → 5e-3)
- Increase augmentation
- Collect more training data

**If training is too slow:**
- Increase batch size (4 → 8)
- Decrease gradient accumulation (4 → 2)
- Temporarily disable augmentation

## 🔍 Model Evaluation

After training, evaluate the model:

```bash
python evaluate_model.py Furkan
```

For only the first 100 samples:
```bash
python evaluate_model.py Furkan --max_samples 100
```

## 📁 Output Files

After training, the following files are created:

```
data/models/personalized_models/Furkan/
├── adapter_config.json          # LoRA adapter configuration
├── adapter_model.bin            # Adapter weights
└── checkpoints/
    └── best_model/              # Best model checkpoint
```

## 🐛 Troubleshooting

### Errors During Training

1. **CUDA out of memory**:
   - Reduce batch size (4 → 2)
   - Increase gradient accumulation (4 → 8)

2. **Validation loss increasing**:
   - Decrease learning rate
   - Early stopping may be working (normal)

3. **WER/CER too high**:
   - Collect more training data
   - Increase number of epochs
   - Check model checkpoints

### Data Issues

1. **Audio files not found**:
   - Run `prepare_training_data.py`
   - Check file paths

2. **Empty transcripts**:
   - Check `metadata_words.csv` file
   - Remove empty rows

## 💡 Tips

1. **First training**: Start with default settings
2. **Iterative improvement**: Evaluate after each training session
3. **Data quality**: Clean, clear recordings are important
4. **Regular checkpoints**: Best model is automatically saved
5. **Monitoring**: Track metrics during training

## 📈 Performance Improvement Strategy

1. **Start**: Train with default settings
2. **Evaluate**: Check WER/CER metrics
3. **Tune**: Optimize hyperparameters if needed
4. **Collect data**: Gather more data if accuracy is low
5. **Re-train**: Train again with improved settings

## 🎓 Resources

- Wav2Vec2: https://huggingface.co/docs/transformers/model_doc/wav2vec2
- LoRA: https://github.com/microsoft/LoRA
- PEFT: https://github.com/huggingface/peft
