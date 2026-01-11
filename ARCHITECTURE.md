# Architecture Documentation

## 📐 General Architecture

This project is organized according to the **Layered Architecture** principle.

## 🏗️ Directory Structure

```
Pronouns/
├── src/                          # Main source code
│   ├── cli/                      # Command Line Interfaces (CLI)
│   │   ├── app.py                # Main application entry point
│   │   ├── collect_data.py      # Data collection CLI
│   │   ├── train.py              # Model training CLI
│   │   └── evaluate.py           # Model evaluation CLI
│   │
│   ├── core/                     # Core Business Logic
│   │   ├── asr.py                # ASR System (Wav2Vec2)
│   │   ├── nlu.py                # Natural Language Understanding
│   │   └── actions.py            # Action execution
│   │
│   ├── services/                 # Business Services Layer
│   │   ├── recording_service.py  # Recording operations service
│   │   ├── training_service.py   # Training operations service
│   │   └── model_service.py      # Model management service
│   │
│   ├── data/                     # Data Access Layer
│   │   └── repository.py         # Repository pattern (data access)
│   │
│   ├── training/                 # Training Modules
│   │   ├── train_asr.py          # ASR training module
│   │   ├── train_lm.py           # Language Model training module
│   │   ├── custom_collator.py   # Custom data collator
│   │   └── augment_from_words.py # Data augmentation
│   │
│   ├── utils/                    # Utility Functions
│   │   ├── utils.py              # General helper functions
│   │   └── reporting.py          # Reporting functions
│   │
│   └── constants.py              # Constants (separate from config)
│
├── config.py                     # Configuration file
├── collect_data.py              # Data collection script (legacy, to be moved to src/cli)
├── train_adapter.py             # Model training script (legacy)
├── evaluate_model.py            # Model evaluation script (legacy)
├── app.py                       # Main application (legacy, to be moved to src/cli)
│
├── data/                        # Data directory
│   ├── users/                   # User data
│   │   └── {user_id}/
│   │       ├── words/           # Word recordings
│   │       │   └── {word}/
│   │       │       └── rep{num}.wav
│   │       ├── letters/        # Letter recordings
│   │       ├── audio/          # Sentence recordings
│   │       └── metadata_*.csv  # Metadata files
│   │
│   └── models/                  # Model directory
│       └── personalized_models/
│           └── {user_id}/
│               └── checkpoints/
│                   └── best_model/
│
└── datasets/                    # Dataset files
    ├── words_set/
    ├── sentence_sets/
    └── letters_set/
```

## 🔄 Layers and Responsibilities

### 1. CLI Layer (Command Line Interface)
**Location:** `src/cli/` (planned) or root directory (current)

**Responsibilities:**
- User interaction
- Command line arguments
- Main application flow
- Error handling and user feedback

**Examples:**
- `app.py` - Main application
- `collect_data.py` - Data collection
- `train_adapter.py` - Model training

### 2. Services Layer (Business Logic)
**Location:** `src/services/`

**Responsibilities:**
- Business logic
- Inter-service coordination
- Data validation
- Business rule enforcement

**Services:**
- `RecordingService` - Recording operations
- `ModelService` - Model management
- `TrainingService` - Training operations (planned)

### 3. Data Access Layer (Repository Pattern)
**Location:** `src/data/`

**Responsibilities:**
- Data access operations
- Metadata management
- File system operations
- Data consistency

**Classes:**
- `UserDataRepository` - User data access

### 4. Core Layer (Domain Logic)
**Location:** `src/core/`

**Responsibilities:**
- Core domain logic
- ASR, NLU systems
- Action execution

**Classes:**
- `ASRSystem` - Automatic speech recognition
- `NLU_System` - Natural language understanding
- `run_action` - Action execution

### 5. Utils Layer (Utilities)
**Location:** `src/utils/`

**Responsibilities:**
- Helper functions
- Common operations
- Reporting

## 🔗 Dependency Flow

```
CLI Layer
    ↓
Services Layer
    ↓
Data Access Layer ←→ Core Layer
    ↓                    ↓
Utils Layer         External Libraries
```

**Rules:**
- Upper layers can depend on lower layers
- Lower layers should not depend on upper layers
- Inter-layer communication should be through interfaces

## 📦 Newly Added Components

### Constants (`src/constants.py`)
- Constants separate from config, immutable values
- File extensions, patterns, default values

### Repository Pattern (`src/data/repository.py`)
- Centralizes data access operations
- Metadata management
- User data access

### Services (`src/services/`)
- Business logic layer
- Recording, model, training services
- Business rule enforcement

## 🚀 Usage Examples

### Recording Service Usage

```python
from src.services.recording_service import RecordingService
from src.constants import RECORD_TYPE_WORD

service = RecordingService(user_id="Furkan")
stats = service.get_recording_stats(record_type=RECORD_TYPE_WORD)
items_to_record, stats = service.get_items_to_record(items_list, RECORD_TYPE_WORD)
```

### Repository Usage

```python
from src.data.repository import UserDataRepository
from src.constants import RECORD_TYPE_WORD

repo = UserDataRepository()
metadata = repo.load_metadata("Furkan", RECORD_TYPE_WORD)
recorded_items = repo.get_recorded_items("Furkan", RECORD_TYPE_WORD)
```

### Model Service Usage

```python
from src.services.model_service import ModelService

model_path = ModelService.find_personalized_model("Furkan")
if model_path:
    print(f"Model found: {model_path}")
```

## 🔄 Future Improvements

1. **Move CLIs**: Move scripts from root to `src/cli/` folder
2. **Training Service**: Move training operations to service layer
3. **Error Handling**: Centralized error handling
4. **Logging**: Centralized logging system
5. **Testing**: Add unit tests
6. **API Layer**: Add REST API (optional)

## 📝 Notes

- Current scripts (collect_data.py, app.py, train_adapter.py) are still in root directory
- New architecture is being implemented gradually
- Backward compatibility is maintained
