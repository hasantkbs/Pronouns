# Reporting and Logging System

## 📊 Overview

The project now includes a centralized reporting and logging system that tracks all operations (recording, training, evaluation) and saves detailed reports in JSON format.

## 📁 Directory Structure

```
Pronouns/
├── reports/                    # JSON reports directory
│   ├── recording_{user_id}_{timestamp}.json
│   ├── training_{user_id}_{timestamp}.json
│   └── evaluation_{user_id}_{timestamp}.json
│
└── logs/                       # Log files directory
    ├── operations_{date}.log  # Daily operation logs
    └── training_{user_id}_{timestamp}.log  # Training-specific logs
```

## 🔧 Configuration

Logging and reporting settings are configured in `config.py`:

```python
# Logging and Reporting Settings
LOG_DIR = "logs"              # Log files directory
LOG_LEVEL = "INFO"            # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
REPORTS_DIR = "reports"       # Reports directory
```

## 📝 Report Types

### 1. Recording Reports

**Location:** `reports/recording_{user_id}_{timestamp}.json`

**Contains:**
- Recording statistics (total items, recorded items, success rate)
- Quality metrics (average quality score)
- Settings used (ideal repetitions, record duration, quality threshold)
- Timestamp and operation details

**Example:**
```json
{
  "report_id": "recording_Furkan_20260107_143022",
  "operation_type": "recording",
  "user_id": "Furkan",
  "timestamp": "2026-01-07T14:30:22",
  "data": {
    "record_type": "kelime",
    "statistics": {
      "total_items": 100,
      "recorded_items": 85,
      "successful_recordings": 850,
      "average_quality_score": 75.5
    }
  }
}
```

### 2. Training Reports

**Location:** `reports/training_{user_id}_{timestamp}.json`

**Contains:**
- Training parameters (epochs, batch size, learning rate, etc.)
- Dataset information (train/eval samples)
- Training metrics (WER, CER, loss values)
- System information (GPU, platform, Python version)
- Training duration

**Example:**
```json
{
  "report_id": "training_Furkan_20260107_150000",
  "operation_type": "training",
  "user_id": "Furkan",
  "timestamp": "2026-01-07T15:00:00",
  "data": {
    "training_parameters": {
      "num_epochs": 20,
      "batch_size": 4,
      "learning_rate": 5e-5,
      "final_wer": 0.12,
      "final_cer": 0.04
    },
    "training_duration": {
      "duration_seconds": 7200,
      "duration_formatted": "02:00:00"
    }
  }
}
```

### 3. Evaluation Reports

**Location:** `reports/evaluation_{user_id}_{timestamp}.json`

**Contains:**
- Evaluation metrics (WER, CER)
- Sample predictions
- Model information
- Number of evaluated samples

## 🚀 Usage

### Automatic Reporting

Reports are automatically created when operations complete:

1. **Recording**: After `collect_data.py` finishes
2. **Training**: After `train_adapter.py` completes
3. **Evaluation**: After `evaluate_model.py` finishes

### Manual Reporting

You can also create custom reports:

```python
from src.services.reporting_service import ReportingService

reporting_service = ReportingService()

# Create a custom report
reporting_service.log_general_operation(
    operation_type="custom_operation",
    user_id="Furkan",
    description="Custom operation description",
    data={"custom_key": "custom_value"}
)
```

## 📋 Log Files

### Operations Log (`logs/operations_{date}.log`)

Contains all operations with timestamps:
```
2026-01-07 14:30:22 - asr_system - INFO - Operation: recording | User: Furkan | Report: recording_Furkan_20260107_143022.json
2026-01-07 15:00:00 - asr_system - INFO - Training Session Completed | User: Furkan | Epochs: 20/20 | Final WER: 0.1200 (12.00%)
```

### Training Log (`logs/training_{user_id}_{timestamp}.log`)

Detailed training logs with:
- System information
- Training progress
- Validation metrics
- Error messages

## 🔍 Viewing Reports

### List All Reports

```bash
ls -lt reports/
```

### View a Specific Report

```bash
cat reports/training_Furkan_20260107_150000.json | jq
```

### Filter Reports by User

```bash
ls reports/*Furkan*.json
```

### Filter Reports by Type

```bash
ls reports/training_*.json
ls reports/recording_*.json
ls reports/evaluation_*.json
```

## 📊 Report Analysis

### Training Performance Tracking

Compare training sessions:

```bash
# Extract WER values from all training reports
grep -h '"final_wer"' reports/training_*.json | jq '.data.training_metrics.final_wer'
```

### Recording Statistics

```bash
# View recording success rates
grep -h '"successful_recordings"' reports/recording_*.json | jq '.data.statistics.successful_recordings'
```

## 🎯 Benefits

1. **Traceability**: Every operation is logged with timestamp
2. **Performance Tracking**: Compare metrics across sessions
3. **Debugging**: Detailed logs help identify issues
4. **Audit Trail**: Complete history of all operations
5. **Analysis**: JSON format enables easy data analysis

## 📝 Notes

- Reports are saved in JSON format for easy parsing
- Logs are saved in text format for human readability
- Both reports and logs include timestamps
- Reports directory is automatically created if it doesn't exist
- Logs directory is automatically created if it doesn't exist
