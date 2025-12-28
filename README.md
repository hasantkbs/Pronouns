# Personalized ASR System for Speech Disorders

## 🎯 About The Project

This project is a personalized Automatic Speech Recognition (ASR) system designed for individuals with speech disorders. It uses a Wav2Vec2-based model, which is fine-tuned with a user's voice recordings to achieve higher accuracy in real-time speech recognition.

The system employs an efficient fine-tuning technique called LoRA (Low-Rank Adaptation) to create a small, personalized adapter for the base model, rather than fully retraining it.

## 📋 System Requirements

- Python 3.9+
- CUDA-enabled GPU (Recommended, but CPU is also supported)
- FFmpeg (for audio processing)

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
- `NUM_FINETUNE_EPOCHS`: 15 (Number of training epochs)
- `FINETUNE_BATCH_SIZE`: 2 (Batch size for training)
- `FINETUNE_LEARNING_RATE`: 1e-4 (Learning rate)
- `ADAPTER_REDUCTION_FACTOR`: 32 (LoRA adapter dimension)

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
SAMPLING_RATE = 16000

# Training settings
NUM_FINETUNE_EPOCHS = 15
FINETUNE_BATCH_SIZE = 2
FINETUNE_LEARNING_RATE = 1e-4
ADAPTER_REDUCTION_FACTOR = 32

# Audio recording settings
RECORD_SECONDS = 5
AUDIO_THRESHOLD = 0.01
```

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
