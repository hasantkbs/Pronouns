## Changelog - 2025-10-09

### Features
-   **Implemented Whisper Model Integration:** Switched the ASR system from Wav2Vec2 to OpenAI's Whisper model for improved personalization capabilities. This involved updating `src/core/asr.py`, `personalize_model.py`, and `evaluate_model.py` to correctly load, fine-tune, and evaluate Whisper models.

### Enhancements
-   **Improved Data Loading Robustness:** Enhanced `personalize_model.py` to gracefully handle missing audio files during dataset preparation, preventing crashes due to `FileNotFoundError`.
-   **Normalized Metadata Paths:** Modified `update_metadata.py` to ensure `metadata_words.csv` uses Linux-style relative paths, resolving `FileNotFoundError` on Linux servers caused by Windows-formatted paths.
-   **Adjusted Fine-tuning Hyperparameters:** Increased `NUM_FINETUNE_EPOCHS` to 20, `FINETUNE_LEARNING_RATE` to `1e-4`, and `ADAPTER_REDUCTION_FACTOR` to 16 in `config.py` to optimize Whisper model learning.

### Bug Fixes
-   **Resolved `ModuleNotFoundError: 'peft'`:** Ensured correct `peft` version (`0.10.0`) is installed to prevent import errors.
-   **Fixed `ImportError: cannot import name 'Cache' from 'transformers'`:** Addressed version incompatibility between `peft` and `transformers`.
-   **Corrected `personalize_model.py` Argument Parsing:** Fixed `unrecognized arguments` error by changing `user_id` to a positional argument.
-   **Mitigated `NotImplementedError` on MPS Devices:** Ensured `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable is used for PyTorch operations on Apple Silicon.
-   **Addressed `preprocessor_config.json` Missing Error:** Modified `personalize_model.py` to save the Whisper processor along with the model.
-   **Fixed `KeyError: 'input_values'` in Data Collator:** Updated `src/training/custom_collator.py` to correctly use `input_features` for Whisper models.
-   **Resolved `TypeError: __init__() got an unexpected keyword argument 'language'`:** Corrected `WhisperForConditionalGeneration.from_pretrained()` calls in `personalize_model.py` and `evaluate_model.py` by removing direct `language` and `task` arguments, as these are handled by the processor.
-   **Addressed `size mismatch` during PEFT loading:** Identified that this error occurs when loading a personalized model trained on a different base model size. The fix involves deleting the old incompatible personalized model and re-training.

### Known Issues
-   **Incomplete `metadata_words.csv`:** The `metadata_words.csv` file for user "Furkan" is still incomplete, leading to 303 audio files being skipped during training. A complete and accurate `metadata_words.csv` is required for optimal training on all available data.
-   **Persistent `size mismatch` error:** The `size mismatch` error during evaluation indicates that an old, incompatible personalized model is still being loaded. This requires deleting the old personalized model directory (`data/models/personalized_models/Furkan`) and re-training.
