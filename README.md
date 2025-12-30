# Personalized ASR System for Speech Disorders

## 🎯 About The Project

This project is a personalized Automatic Speech Recognition (ASR) system designed for individuals with speech disorders. It uses a Wav2Vec2-based model, which is fine-tuned with a user's voice recordings to achieve higher accuracy in real-time speech recognition.

The system employs an efficient fine-tuning technique called LoRA (Low-Rank Adaptation) to create a small, personalized adapter for the base model, rather than fully retraining it.

## 📋 System Requirements

- Python 3.9+
- CUDA-enabled GPU (Recommended, but CPU is also supported)
- FFmpeg (for audio processing)

### Optimized for High-Performance Servers

The system is optimized for:
- **GPU**: NVIDIA RTX A5000 (24GB VRAM) or similar
- **CPU**: 48+ cores (Intel Xeon E5-2670 v3 or similar)
- See `SERVER_OPTIMIZATION.md` for details

## 🚀 Quick Start

The following steps guide you through preparing data, training a model, and using it for a sample user named `Furkan`.

### 1. Prepare Data

The system requires user-specific audio recordings and their transcriptions. For this project, sample data for the user "Furkan" is already provided in the `data/users/Furkan/` directory.

To prepare the data for training, run the following command:

```bash
python prepare_training_data.py Furkan
```

This script will:
- Read the main `metadata_words.csv` file.
- Split the data into training (80%) and evaluation (20%) sets.
- Create `train.csv` and `eval.csv` in the user's data directory.

### 2. Train the Model

To train a personalized model for the user, run:

```bash
python train_adapter.py Furkan
```

This script performs the following actions:
- Loads the base Wav2Vec2 model.
- Fine-tunes it using a LoRA adapter with the user's data.
- Saves the trained adapter to `data/models/personalized_models/Furkan/`.

**Training Parameters** (configurable in `config.py`):
- `NUM_FINETUNE_EPOCHS`: 20 (Number of training epochs - optimized for speech disorders)
- `FINETUNE_BATCH_SIZE`: 4 (Batch size for training)
- `FINETUNE_LEARNING_RATE`: 5e-5 (Learning rate - lower for stability)
- `ADAPTER_REDUCTION_FACTOR`: 16 (LoRA adapter dimension - more parameters for better adaptation)
- `USE_AUGMENTATION`: True (Data augmentation for better generalization)
- `EARLY_STOPPING_PATIENCE`: 5 (Early stopping to prevent overfitting)
- `WARMUP_STEPS`: 100 (Learning rate warmup steps)

**New Features:**
- ✅ Data augmentation (optimized for speech disorders)
- ✅ Validation during training with WER/CER metrics
- ✅ Early stopping to prevent overfitting
- ✅ Learning rate scheduling with warmup
- ✅ Gradient clipping for training stability
- ✅ Automatic best model checkpointing

### 3. Evaluate the Model

To evaluate the performance of the fine-tuned model, use:

```bash
python evaluate_model.py Furkan
```

*Optional: To evaluate only the first 100 samples:*
```bash
python evaluate_model.py Furkan --max_samples 100
```

This command will:
- Calculate WER (Word Error Rate) and CER (Character Error Rate) metrics.
- Display sample predictions.
- Provide suggestions for improvement.

### 4. Real-time Usage

To use the trained model for real-time speech recognition, run the main application:

```bash
python app.py
```

The system will prompt you for a User ID. Type `Furkan` and press ENTER. The application will then:
- Automatically load the personalized model if it exists.
- Start listening to the microphone.
- Transcribe your speech to text and display it on the screen.

**To exit:** Say "çık" or "exit".

## 📁 Project Structure

```
Pronouns/
├── app.py                          # Main application entry point
├── config.py                       # Configuration file
├── prepare_training_data.py        # Data preparation script
├── train_adapter.py                # Model training script
├── evaluate_model.py               # Model evaluation script
├── src/
│   ├── core/
│   │   ├── asr.py                  # ASR System (Wav2Vec2)
│   │   ├── nlu.py                  # Natural Language Understanding
│   │   └── actions.py              # Action execution
│   └── utils/
│       └── utils.py                # Helper functions
├── data/
│   ├── users/
│   │   └── Furkan/
│   │       ├── metadata_words.csv  # Audio file metadata
│   │       ├── train.csv           # Training dataset
│   │       ├── eval.csv            # Evaluation dataset
│   │       └── words/              # Directory for audio files (.wav)
│   └── models/
│       └── personalized_models/
│           └── Furkan/             # Saved personalized model adapter
└── requirements.txt                # Python dependencies
```

## ⚙️ Configuration

You can adjust the following settings in `config.py`:

```python
# Model settings
MODEL_NAME = "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish"
ORNEKLEME_ORANI = 16000

# Training settings (optimized for speech disorders)
NUM_FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 4
FINETUNE_LEARNING_RATE = 5e-5
ADAPTER_REDUCTION_FACTOR = 16
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 100
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 5
USE_AUGMENTATION = True

# Audio recording settings
KAYIT_SURESI_SN = 5
SES_ESIK_DEGERI = 0.01
```

### Training Improvements for Speech Disorders

The training pipeline has been optimized specifically for speech disorder recognition:

1. **Data Augmentation**: Light augmentation (noise, time stretch, pitch shift) to improve generalization without distorting speech patterns
2. **Validation**: Real-time WER/CER metrics during training to monitor progress
3. **Early Stopping**: Prevents overfitting by stopping when validation loss stops improving
4. **Learning Rate Scheduling**: Warmup and linear decay for stable training
5. **Enhanced LoRA**: More adapter modules (q_proj, v_proj, k_proj, out_proj) for better adaptation
6. **Gradient Clipping**: Prevents gradient explosion for training stability

## 🐛 Troubleshooting

### Model fails to load
- Check your internet connection (the base model is downloaded on first use).
- Ensure the trained model exists at `data/models/personalized_models/Furkan/`.

### Errors during training
- Make sure the audio files are located in `data/users/Furkan/words/`.
- Verify that `metadata_words.csv` is correctly formatted.
- Run `prepare_training_data.py` before starting the training.

### Low accuracy
- Collect more training data.
- Increase the number of epochs (`NUM_FINETUNE_EPOCHS` in `config.py`).
- Adjust the learning rate (`FINETUNE_LEARNING_RATE`).

## 📊 Performance Metrics

Target metrics for a good model:
- **WER < 0.15** (Word Error Rate less than 15%)
- **CER < 0.05** (Character Error Rate less than 5%)
