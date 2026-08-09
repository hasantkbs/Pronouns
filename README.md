# Personalized ASR System for Speech Disorders

## About The Project

This project is a personalized Automatic Speech Recognition (ASR) and speech synthesis system designed for individuals with speech disorders. It learns a person's unique pronunciation patterns from their voice recordings and uses those same recordings to synthesize understandable speech — enabling meaningful communication with others.

The pipeline works in three stages:

1. **Record** — Capture the person's word recordings with automatic quality validation.
2. **Train** — Fine-tune a Wav2Vec2 model with LoRA adapters on the recordings.
3. **Run** — Recognize what the person says and respond using their own voice.

---

## System Requirements

- Python 3.9+
- Microphone
- CUDA-enabled GPU (recommended; CPU is supported but slower for training)
- Internet connection (base model is downloaded on first run, ~1.2 GB)

### Recommended Hardware

| Component | Recommended |
|---|---|
| GPU | NVIDIA RTX A5000 (24 GB VRAM) or equivalent |
| CPU | 48+ cores (Intel Xeon or similar) |
| RAM | 32 GB+ |

See `SERVER_OPTIMIZATION.md` and `LINUX_SERVER_SETUP.md` for server setup details.

---

## Quick Start

Replace `Furkan` with your actual user ID throughout these steps.

### Step 1 — Collect Voice Recordings

#### Autonomous mode (recommended)

Starts recording automatically when speech is detected. No button presses required.  
The user can start the script and leave — the system works unattended.

```bash
python auto_collect.py Furkan datasets/words_set/temel_kelimeler.txt
```

Every recording passes three validation layers before being saved:

1. **Quality filter** — rejects silent, too-short, or noisy recordings (RMS / duration / SNR).
2. **Speech ratio filter** — rejects recordings that contain mostly background noise.
3. **ASR verification** *(optional)* — rejects recordings where the spoken word does not match the target, using character error rate (CER) with tolerance for speech disorders.

Common options:

```bash
# Request 5 repetitions per word (default: 5)
python auto_collect.py Furkan datasets/words_set/temel_kelimeler.txt --reps 5

# Resume from where the session was interrupted (default behaviour)
python auto_collect.py Furkan datasets/words_set/temel_kelimeler.txt

# Start fresh, ignoring existing recordings
python auto_collect.py Furkan datasets/words_set/temel_kelimeler.txt --no-resume

# Lower sound detection threshold for quiet voices
python auto_collect.py Furkan datasets/words_set/temel_kelimeler.txt --threshold 0.008

# Enable ASR-based word verification (set AUTO_ASR_VERIFY=True in config.py first)
python auto_collect.py Furkan datasets/words_set/temel_kelimeler.txt --asr-verify
```

#### Manual mode

Prompts the user to press ENTER before each recording and offers playback/keep options.

```bash
python collect_data.py
```

#### Re-record specific words

To re-record words listed in `datasets/tekrar_kayit.txt`:

```bash
python collect_data.py --re-record
```

---

### Step 2 — Prepare Training Data

Split recordings into training (80%) and evaluation (20%) sets:

```bash
python prepare_training_data.py Furkan
```

This creates `train.csv` and `eval.csv` in `data/users/Furkan/`.

---

### Step 3 — Train the Model

```bash
python train_adapter.py Furkan
```

What happens during training:

- Loads the base Turkish Wav2Vec2 model (`mpoyraz/wav2vec2-xls-r-300m-cv8-turkish`).
- Applies a LoRA adapter targeting both attention and feed-forward layers (rank 16).
- Trains with cosine LR scheduling, gradient clipping, and mixed precision (fp16).
- Applies light data augmentation tuned for speech disorder patterns.
- Runs WER/CER validation after every epoch; saves the best model by WER.
- Stops early if WER stops improving (patience configurable in `config.py`).
- Saves the final adapter to `data/models/personalized_models/Furkan/`.

To use a different base model:

```bash
python train_adapter.py Furkan --base_model /path/to/other/model
```

---

### Step 4 — Evaluate the Model

```bash
python evaluate_model.py Furkan
```

Optional — limit evaluation to the first 100 samples:

```bash
python evaluate_model.py Furkan --max_samples 100
```

Target metrics for a well-trained model:

| Metric | Target |
|---|---|
| WER (Word Error Rate) | < 15% |
| CER (Character Error Rate) | < 5% |

---

### Step 5 — Run the Application

```bash
python app.py
```

Enter a user ID when prompted. The app will:

1. Load the personalized LoRA adapter if one exists, otherwise fall back to the base model.
2. Start the word synthesizer, indexing the user's recorded words for playback.
3. Listen via microphone (press ENTER to start each recording).
4. Transcribe speech using CTC beam search with KenLM language model (if available) or greedy decoding.
5. Display the transcription and a **confidence score** (0–100%).
6. Resolve the intent with the NLU system.
7. Execute the action and **play the response back in the user's own recorded voice**.

Say "çık" or "exit" to quit.

---

## Architecture Overview

```
auto_collect.py          Autonomous recording with 3-layer validation
collect_data.py          Interactive recording with manual review

prepare_training_data.py Split metadata into train/eval CSV

train_adapter.py         LoRA fine-tuning on Wav2Vec2
  └── PersonalizedTrainer
        ├── LoRA rank=16, targets: attention + feed-forward layers
        ├── Cosine LR scheduler with warmup
        ├── WER-based best model selection
        ├── Quality-filtered dataset loading
        └── Light augmentation (noise, time-stretch, pitch-shift)

app.py                   Main real-time application
  ├── ASRSystem           Wav2Vec2 + LoRA adapter
  │     ├── CTC beam search with KenLM (if available)
  │     └── Confidence scoring per utterance
  ├── NLU_System          Rule-based intent classifier (19 intents, Turkish)
  ├── WordSynthesizer     Assembles response from user's own recordings
  │     ├── Quality-indexed word database
  │     ├── Best-K recording selection
  │     └── Crossfade + natural pauses
  └── run_action()        Executes the resolved intent
```

---

## Project Structure

```
Pronouns/
├── app.py                          # Main application entry point
├── auto_collect.py                 # Autonomous recording script (new)
├── collect_data.py                 # Interactive recording script
├── config.py                       # All configuration parameters
├── prepare_training_data.py        # train/eval CSV preparation
├── train_adapter.py                # LoRA fine-tuning trainer
├── evaluate_model.py               # WER/CER evaluation
├── src/
│   ├── core/
│   │   ├── asr.py                  # ASRSystem: Wav2Vec2 + LM beam search
│   │   ├── nlu.py                  # NLU: intent + entity extraction
│   │   ├── actions.py              # Action handlers
│   │   └── synthesizer.py          # WordSynthesizer: voice response (new)
│   ├── services/
│   │   ├── recording_service.py
│   │   ├── model_service.py
│   │   └── reporting_service.py
│   ├── data/
│   │   └── repository.py
│   ├── training/
│   │   ├── train_asr.py
│   │   ├── train_lm.py
│   │   ├── custom_collator.py
│   │   └── augment_from_words.py
│   ├── utils/
│   │   ├── utils.py                # record_audio_auto, validate_recording (new)
│   │   └── reporting.py
│   └── constants.py
├── data/
│   ├── users/{user_id}/
│   │   ├── words/{word}/rep{n}.wav
│   │   ├── metadata_words.csv
│   │   ├── train.csv
│   │   └── eval.csv
│   ├── models/personalized_models/{user_id}/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── checkpoints/best_model/
│   └── lm/lm.arpa                  # Optional KenLM language model
├── datasets/
│   ├── words_set/
│   ├── sentence_sets/
│   └── letters_set/
├── reports/                        # JSON session reports
├── logs/                           # Training and app logs
├── KULLANICI_KILAVUZU.md           # Turkish user guide
└── requirements.txt
```

---

## Configuration Reference

All settings live in `config.py`. Key parameters:

### Model

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `mpoyraz/wav2vec2-xls-r-300m-cv8-turkish` | Base Wav2Vec2 model |
| `ORNEKLEME_ORANI` | `16000` | Sample rate (Hz) |
| `KENLM_MODEL_PATH` | `data/lm/lm.arpa` | Path to KenLM language model |
| `LM_ALPHA` | `0.5` | LM weight in CTC beam search |
| `LM_BEAM_WIDTH` | `100` | Beam search width |

### Training

| Parameter | Default | Description |
|---|---|---|
| `ADAPTER_REDUCTION_FACTOR` | `16` | LoRA rank (higher = more capacity) |
| `NUM_FINETUNE_EPOCHS` | `30` | Maximum training epochs |
| `FINETUNE_LEARNING_RATE` | `2e-5` | Peak learning rate |
| `LR_SCHEDULER_TYPE` | `"cosine"` | `"cosine"` or `"linear"` |
| `EARLY_STOPPING_PATIENCE` | `5` | Stop after N epochs without WER improvement |
| `USE_AUGMENTATION` | `True` | Enable data augmentation |
| `MIXED_PRECISION` | `"fp16"` | `"fp16"`, `"bf16"`, or `"no"` |

### Autonomous Recording

| Parameter | Default | Description |
|---|---|---|
| `AUTO_WORD_TIMEOUT_SEC` | `15` | Skip word after this many seconds with no valid recording |
| `AUTO_MAX_RETRIES` | `3` | Automatic retry attempts per repetition |
| `AUTO_SILENCE_LIMIT_SEC` | `1.0` | Silence duration that ends a recording |
| `AUTO_SPEECH_WAIT_SEC` | `4.0` | Max wait for speech onset |
| `AUTO_SOUND_THRESHOLD` | `0.012` | VAD threshold (lower = more sensitive) |
| `AUTO_MIN_SPEECH_RATIO` | `0.10` | Minimum speech frame ratio (rejects empty recordings) |
| `AUTO_ASR_VERIFY` | `False` | Enable ASR word verification |
| `AUTO_ASR_MAX_CER` | `0.6` | Max CER for ASR verification (tolerant for speech disorders) |

### Synthesis

| Parameter | Default | Description |
|---|---|---|
| `SYNTHESIS_MIN_QUALITY` | `50` | Minimum quality score for synthesis candidates |
| `SYNTHESIS_BEST_K` | `3` | Top-K recordings considered per word |
| `SYNTHESIS_CROSSFADE_MS` | `30` | Crossfade between words (ms) |
| `SYNTHESIS_PAUSE_MS` | `150` | Silence between words (ms) |

---

## Troubleshooting

### No speech detected during auto_collect

Lower the detection threshold:
```bash
python auto_collect.py Furkan wordlist.txt --threshold 0.008
```
Or set `AUTO_SOUND_THRESHOLD = 0.008` in `config.py`.

### Recording rejected as "too quiet"

Bring the microphone closer (20–30 cm). Also lower `MIN_RMS_LEVEL` in `config.py`:
```python
MIN_RMS_LEVEL = 200
```

### Base model fails to load

Check your internet connection. The model (~1.2 GB) is downloaded from Hugging Face on first run.

### Training is very slow

Without a CUDA GPU, training runs on CPU and may take 10–20+ hours.  
Transfer to a GPU server — see `LINUX_SERVER_SETUP.md`.

### Low recognition accuracy

- Collect more recordings (5 repetitions per word minimum).
- Re-run `train_adapter.py`.
- Ensure recording environment is quiet.
- Increase `NUM_FINETUNE_EPOCHS` or lower `FINETUNE_LEARNING_RATE` in `config.py`.

### Synthesizer says "missing words"

The synthesizer can only play back words that have been recorded.  
Add the missing words to your word list and run `auto_collect.py` again.

---

## Documentation

| File | Contents |
|---|---|
| `KULLANICI_KILAVUZU.md` | Turkish step-by-step user guide |
| `ARCHITECTURE.md` | System architecture and design patterns |
| `TRAINING_GUIDE.md` | Hyperparameter tuning and training best practices |
| `SERVER_OPTIMIZATION.md` | GPU server optimization for RTX A5000 |
| `LINUX_SERVER_SETUP.md` | Linux server setup and deployment |
