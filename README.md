# Speech Disorder Recognition System

This project is a **simple and focused system** specially designed for **individuals with speech disorders** to recognize their speech and convert it to text. The system uses a `Wav2Vec2`-based ASR (Automatic Speech Recognition) model and supports **personalization** to adapt to a specific user's voice.

## 🚀 Workflow: From Zero to Personalized Recognition

This guide explains how to collect personal voice data, train a personalized model, and use it in the main application.

### Step 1: Collect Personal Data

This script records your voice for a series of sentences and prepares the data for training.

1.  **Navigate to the project directory** in your terminal.
2.  **Run the data collection script.**

    ```bash
    python collect_data.py
    ```
3.  **Enter a User ID** when prompted (e.g., `user_001`, `hasan`, etc.). This ID will be used to save your data.
4.  **Read the sentences** that appear on the screen. Press ENTER before each sentence to start recording.

#### Using a Custom Sentence List

By default, the script uses a predefined list of sentences. You can provide your own list in a `.txt` file (one sentence per line).

```bash
# Create a text file with your sentences
# my_sentences.txt
#   "First custom sentence."
#   "Another sentence to record."

# Run the script with the --file argument
python src/training/collect_user_data.py --file my_sentences.txt
```

### Re-recording Data (`--re-record`)

If you want to re-record a specific list of words or letters, you can use the `--re-record` flag. This is useful for targeting words that the model struggles with.

1.  **Create a file** named `tekrar_kayit.txt` in the `datasets/` directory.
2.  **Add the words or letters** you want to re-record to this file, one item per line.
3.  **Run the script** with the `--re-record` flag:

    ```bash
    python collect_data.py --re-record
    ```

The script will then guide you through the process of re-recording each item in the `tekrar_kayit.txt` file.

Your recorded audio and a `metadata.csv` file will be saved under `data/users/YOUR_USER_ID/`.

### Step 2: Personalize the Model

This script fine-tunes the base ASR model using your collected data.

1.  **Run the personalization script** with the same User ID you used in Step 1.

    ```bash
    python personalize_model.py YOUR_USER_ID
    ```
    *(Replace `YOUR_USER_ID` with the actual ID, e.g., `python personalize_model.py hasan`)*

2.  The script will load the base model, fine-tune it with your data, and save the new, personalized model to `data/models/personalized_models/YOUR_USER_ID/`.

### Step 3: Run the Application

Now you can use the main application, which will automatically load your personalized model.

1.  **Start the application.**
    ```bash
    python app.py
    ```
2.  **Enter your User ID** when prompted.
3.  The system will detect your personalized model and load it. If no personalized model is found for the ID, it will fall back to the default model.
4.  Press **ENTER** to speak, and the system will transcribe your speech using the appropriate model.

---

## 🎯 Project Purpose

For individuals with speech disorders:
- **Convert speech to text** with high accuracy through personalization.
- **Facilitate communication** and **increase independence**.

## ⚙️ Features

- **Personalized Model Training:** Fine-tune the model for a specific user's voice for significantly improved accuracy.
- **Flexible Data Collection:** Use a default list of sentences or provide your own via a text file.
- **Dynamic Model Loading:** The app automatically detects and loads a user's personalized model if it exists.
- **High-accuracy** base model (`Wav2Vec2`).
- **Real-time** audio processing.
- **Turkish** language support.

### Audio Quality Analysis

To ensure the quality and consistency of the collected data, the `collect_data.py` script includes a real-time audio quality analysis feature. This feature is especially useful for identifying inconsistencies in the pronunciation of a word across multiple repetitions.

**How it works:**

1.  **Embedding Extraction:** After you a record a word or a letter for the specified number of repetitions (e.g., 5 times), the script uses a pre-trained `Whisper` model to extract an "embedding" (a numerical representation) from each audio recording.
2.  **Similarity Analysis:** The script then calculates the `cosine similarity` between all pairs of embeddings. This provides a measure of how similar the recordings are to each other.
3.  **Feedback:** If the average similarity score is below `0.80`, the script will display a warning message, suggesting that you may want to re-record the word later using the `--re-record` option.

This feature helps to ensure that the training data is consistent, which can lead to a more accurate and robust personalized model.

## 🔧 General Model Training (Optional)

If you want to train the base model from scratch on a large dataset (like Mozilla Common Voice), you can use the `train_model.py` script.

```bash
# This requires a large, pre-processed dataset in the `downloaded_data` folder.
# (GPU recommended)
python train_model.py
```
After training, you can set this new model as the default in `config.py`.

## Development Roadmap Notes

### Model Update

The project's base ASR model has been updated to `mpoyraz/wav2vec2-xls-r-300m-cv8-turkish` to provide higher accuracy. This model is trained on a larger dataset for Turkish and is expected to perform better.

### On Data Collection Strategy

The idea of "creating synthetic sentences by recording individual words and combining them" was evaluated to speed up the process of collecting a large dataset (e.g., 5000 words).

The analysis of this approach led to the following decision:

*   **Core Problem:** This method destroys the most important elements of natural speech: **intonation (prosody)** and **sound transitions (coarticulation)**. This can cause the model to learn an unrealistic, robotic speaking style and reduce real-world performance.

*   **Decision:** This method should **not** be used as the main training strategy.

*   **Proposed Hybrid Approach:**
    1.  **Main Dataset:** The priority for the foundation of the training is to record **complete sentences read naturally and fluently**. This ensures the model learns correct intonation and rhythm.
    2.  **Data Augmentation:** Critical words that are rare or non-existent in the main dataset can be recorded individually. Synthetic sentences created from these words can be used for "data augmentation," making up a small portion of the total training data (e.g., 10-20%).

This hybrid model allows the model to learn natural speech while also helping to expand its vocabulary.

---

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

For higher accuracy, you can install the KenLM language model. Download the model and update the `KENLM_MODEL_PATH` in `config.py`.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'New feature added'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License.