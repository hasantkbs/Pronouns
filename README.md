# Speech Recognition and Personalization System

This project is an **Automatic Speech Recognition (ASR) system** designed to assist individuals with speech disorders by accurately converting their speech to text. It leverages state-of-the-art `Whisper` models from the Hugging Face `transformers` library and supports **user-specific personalization** for enhanced accuracy.

## ✨ Features

*   **Accurate Speech-to-Text:** Utilizes powerful `Whisper` models for high-fidelity speech recognition.
*   **User-Specific Personalization:** Efficiently fine-tune models for individual users using Parameter-Efficient Fine-Tuning (PEFT), specifically LoRA.
*   **Flexible Data Collection:** Includes scripts for collecting custom voice data (words, sentences) from users.
*   **Dynamic Model Loading:** The main application (`app.py`) automatically detects and loads a user's personalized model if available.
*   **Turkish Language Support:** Optimized for Turkish speech recognition.
*   **Performance Optimizations for Training:**
    *   **Parallel Data Preprocessing:** Utilizes multi-core CPUs for faster dataset preparation.
    *   **Gradient Checkpointing:** Reduces GPU memory consumption during full model fine-tuning.
    *   **Optional Flash Attention 2:** Significantly accelerates training on compatible CUDA-enabled GPUs.

## 🚀 Getting Started

### 1. Prerequisites

*   **Python 3.9+**
*   `pip` (Python package installer)
*   **FFmpeg (System-wide):** Essential for various audio processing tasks. Install it using your system's package manager:
    *   **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    *   **macOS (with Homebrew):** `brew install ffmpeg`
    *   **Windows (with Chocolatey):** `choco install ffmpeg`
    *   Or download from [FFmpeg Official Website](https://ffmpeg.org/download.html).

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/Pronouns.git
    cd Pronouns
    ```
    *(Replace `https://github.com/your_username/Pronouns.git` with your actual repository URL)*

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install project dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Optional: Install Flash Attention 2 (for compatible CUDA GPUs):**
    If you are training on a server with a compatible NVIDIA GPU (e.g., A100, H100) and have the necessary CUDA setup, installing `flash-attn` can significantly speed up training.
    ```bash
    # Ensure you are in your activated virtual environment
    pip install flash-attn --no-build-isolation
    ```
    *(Note: `flash-attn` may be difficult to install on non-Linux or non-CUDA systems. The code will gracefully fall back if it's not available.)*

### 3. Project Structure Overview

*   `app.py`: The main application to run real-time speech recognition.
*   `config.py`: Centralized configuration settings for models, data paths, and training.
*   `scripts/data_management/`: Utility scripts for managing and fixing data.
*   `train_full.py`: Script for full fine-tuning of a Whisper model (e.g., a base model on a large dataset).
*   `train_adapter.py`: Script for efficient, user-specific fine-tuning using LoRA (PEFT).
*   `src/core/`: Contains core ASR and Natural Language Understanding (NLU) logic.
*   `data/`: Directory for storing user-specific data, personalized models, etc.

## 👨‍💻 Usage

### 1. Collect Personal Data

Use `collect_data.py` to record speech data for a specific user.

```bash
python collect_data.py
```
*   You will be prompted to enter a `User ID`.
*   The script will display sentences/words for you to read. Press `ENTER` to start recording for each prompt.
*   Recorded audio and metadata will be saved under `data/users/YOUR_USER_ID/`.

#### Re-recording Specific Items

To re-record words/letters the model struggles with, create a `tekrar_kayit.txt` file in the `datasets/` directory (one item per line) and run:
```bash
python collect_data.py --re-record
```

#### Audio Quality Analysis

The `collect_data.py` script includes real-time audio quality analysis using Whisper embeddings. If the similarity between repetitions of a word is low, it suggests re-recording for better model accuracy.

### 2. Personalize Your Model (Efficient Fine-tuning)

Use `train_adapter.py` to fine-tune a Whisper model specifically for a user's voice using LoRA (PEFT). This is highly recommended for user personalization due to its efficiency.

```bash
python train_adapter.py YOUR_USER_ID --base_model openai/whisper-small
```
*(Replace `YOUR_USER_ID` with the actual ID. You can also specify a different `--base_model`.)*
The personalized model will be saved to `data/models/personalized_models/YOUR_USER_ID/`.

### 3. Run the Application

Start the main application to use the ASR system.

```bash
python app.py
```
*   Enter your `User ID` when prompted.
*   The system will automatically detect and load your personalized model if it exists. Otherwise, it will use the default model specified in `config.py`.

### 4. Full Model Fine-tuning (Advanced/Optional)

For training a base Whisper model from scratch on a large dataset or for a full fine-tune (not LoRA), use `train_full.py`. This usually requires significant computational resources.

```bash
python train_full.py YOUR_USER_ID --base_model openai/whisper-base
```
*(This script includes Gradient Checkpointing for VRAM optimization.)*

## 📈 Performance Considerations

*   **Parallel Preprocessing:** Data loading and preprocessing steps are parallelized across available CPU cores for speed.
*   **Gradient Checkpointing:** Enabled in `train_full.py` to optimize GPU memory usage.
*   **Flash Attention 2:** Automatically attempted in training scripts if `flash-attn` is installed and compatible hardware is present, providing substantial speedups on supported GPUs.

## 🇹🇷 Language Model (Optional)

For further accuracy improvements, especially for Turkish, you can integrate a KenLM language model.
1.  Download a pre-trained Turkish KenLM model.
2.  Update the `KENLM_MODEL_PATH` variable in `config.py` to point to your downloaded model.

## 🤝 Contributing

1.  Fork the repository
2.  Create feature branch (`git checkout -b feature/new-feature`)
3.  Commit changes (`git commit -am 'New feature added'`)
4.  Push to branch (`git push origin feature/new-feature`)
5.  Create Pull Request

## 📄 License

This project is licensed under the MIT License.
