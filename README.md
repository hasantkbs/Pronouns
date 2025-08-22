# Speech Disorder Recognition System

This project is a **simple and focused system** specially designed for **individuals with speech disorders** to recognize their speech and convert it to text. The system uses a `Wav2Vec2`-based ASR (Automatic Speech Recognition) model for speech-to-text conversion.

## 🎯 Project Purpose

For individuals with speech disorders:
- **Convert speech to text**
- **Transform conversations into written form**
- **Facilitate communication**
- **Increase independence**

## 🏗️ System Architecture

The system has a simple and focused structure:

1. **Audio Recording:** Records audio using microphone
2. **Speech Recognition:** Converts speech to text using Wav2Vec2 model
3. **Text Output:** Displays the recognized text on screen

## 📁 Project Structure

```
konusma_anlama_sistemi/
├── src/                    # Source code
│   ├── core/              # Core components
│   │   └── asr.py         # Speech recognition system
│   └── utils/             # Utility tools
│       └── utils.py       # Audio recording functions
├── data/                  # Data files
│   └── models/           # Trained models
├── downloaded_data/       # Training data (48K+ audio files)
├── app.py                # Main application
├── config.py             # Configuration
├── train_model.py        # Model training
├── analyze_data.py       # Dataset analysis
└── requirements.txt      # Dependencies
```

## 🚀 Installation

### 1. Create Conda Environment

```bash
# Create new conda environment
conda create -n pronouns python=3.9
conda activate pronouns

# Install project dependencies
pip install -r requirements.txt
```

### 2. Language Model Installation (Optional)

For higher accuracy, you can install KenLM language model:

```bash
# Create language model directory
mkdir -p data/models/language_model

# Download KenLM model and place in data/models/language_model/ folder
# Update KENLM_MODEL_PATH in config.py
```

## 🎮 Usage

To start the system:

```bash
python app.py
```

### How it Works:

1. **Press ENTER** - Audio recording starts
2. **Speak** - You can speak for 5 seconds
3. **See the result** - Recognized text appears on screen
4. **To exit** - Say "çık" or "exit"

### Example Usage:

```
=========================================
   Speech Disorder Recognition System   
=========================================
This system recognizes speech from individuals
with speech disorders and converts it to text.
Say 'çık' or 'exit' to quit.

------------------------------------------
🎤 Press ENTER to speak and start recording...
🔴 Recording started - You can speak now...
🟢 Recording completed!
💾 Audio file saved: temp_recording.wav

🧠 Analyzing your speech...

📝 Recognized Text:
   'hello how are you'
```

## 🔧 Model Training

### Dataset Analysis

To analyze the existing dataset:

```bash
python analyze_data.py
```

**Available Dataset:**
- 🎵 **48,365 audio files** (Turkish)
- 📄 **6 metadata files** (Parquet format)
- 📊 **Train:** 32,802 files
- 📊 **Test:** 11,035 files
- 📊 **Validation:** Available

### Model Training

To train your own model:

```bash
# Model training (GPU recommended)
python train_model.py
```

**Training Parameters:**
- **Epochs:** 3
- **Batch Size:** 8
- **Learning Rate:** 1e-4
- **Model:** Wav2Vec2 Large 960h

### Using Trained Model

After training is complete:

1. Update model path in `config.py`:
   ```python
   MODEL_NAME = "./trained_model"
   ```

2. Restart the system:
   ```bash
   python app.py
   ```

## 🧪 Testing

To test the system:

```bash
# Dataset analysis
python analyze_data.py

# Audio recording test
python src/utils/utils.py

# ASR system test (if torch is installed)
python src/core/asr.py
```

## ⚙️ Configuration

You can change settings in `config.py`:

```python
# ASR Model Settings
MODEL_NAME = "facebook/wav2vec2-large-960h"  # Change model
SAMPLING_RATE = 16000  # Hz

# Audio Recording Settings
RECORDING_DURATION_SEC = 5  # Change recording duration
```

## 🎯 Features

### ✅ Current Features:
- **Simple and focused** interface
- **High accuracy** speech recognition
- **Turkish** language support
- **Real-time** audio processing
- **Easy to use** - just ENTER + speak
- **Rich dataset** - 48K+ Turkish audio files
- **Model training** - You can train your own model

### 🔄 Future Features:
- **Personalized** model training
- **Voice response** system
- **Web interface**
- **Mobile application**

## 📝 Development

### Adding New Model:

1. Change model name in `config.py`:
   ```python
   MODEL_NAME = "new/model/name"
   ```

2. The system will automatically use the new model.

### Adding New Feature:

1. Create new module in `src/core/`
2. Integrate in `app.py`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'New feature added'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Mozilla Common Voice project
- Hugging Face Transformers library
- Wav2Vec2 model developers

## ⚠️ Important Notes

### Dataset
- This repository **does not contain audio files**
- Training data must be provided separately
- Personal audio data is not included for privacy reasons

### Model Training
- GPU usage is recommended
- Sufficient disk space required for large datasets
- Training time depends on dataset size

---

**Note:** This project is developed to improve the quality of life for individuals with speech disorders.