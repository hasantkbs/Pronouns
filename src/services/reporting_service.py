# -*- coding: utf-8 -*-
"""
Reporting Service - Centralized reporting and logging system
Handles reports for all operations (recording, training, evaluation)
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import config


class ReportingService:
    """Centralized reporting service for all operations"""
    
    def __init__(self, reports_dir: str = "reports", logs_dir: str = "logs"):
        """
        Initialize reporting service.
        
        Args:
            reports_dir: Directory for JSON reports
            logs_dir: Directory for log files
        """
        self.reports_dir = Path(reports_dir)
        self.logs_dir = Path(logs_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup application logger"""
        logger = logging.getLogger("asr_system")
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler
        log_file = self.logs_dir / f"operations_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def create_report(self, operation_type: str, user_id: str, 
                     operation_data: Dict[str, Any]) -> Path:
        """
        Create a report for an operation.
        
        Args:
            operation_type: Type of operation (recording, training, evaluation)
            user_id: User ID
            operation_data: Operation data dictionary
        
        Returns:
            Path to created report file
        """
        timestamp = datetime.now()
        report_id = f"{operation_type}_{user_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create report structure
        report = {
            "report_id": report_id,
            "operation_type": operation_type,
            "user_id": user_id,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime('%Y-%m-%d'),
            "time": timestamp.strftime('%H:%M:%S'),
            "data": operation_data
        }
        
        # Save JSON report
        report_file = self.reports_dir / f"{report_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Log to logger
        self.logger.info(
            f"Operation: {operation_type} | User: {user_id} | "
            f"Report: {report_file.name}"
        )
        
        return report_file
    
    def log_recording_session(self, user_id: str, record_type: str,
                             stats: Dict[str, Any]) -> Path:
        """
        Log a recording session.
        
        Args:
            user_id: User ID
            record_type: Type of recording (word, sentence, letter)
            stats: Recording statistics
        
        Returns:
            Path to created report file
        """
        operation_data = {
            "record_type": record_type,
            "statistics": {
                "total_items": stats.get("total_items", 0),
                "recorded_items": stats.get("recorded_items", 0),
                "skipped_items": stats.get("skipped_items", 0),
                "total_recordings": stats.get("total_recordings", 0),
                "successful_recordings": stats.get("successful_recordings", 0),
                "failed_recordings": stats.get("failed_recordings", 0),
                "average_quality_score": stats.get("avg_quality_score", 0.0),
                "items_completed": stats.get("items_completed", 0)
            },
            "settings": {
                "ideal_repetitions": config.IDEAL_REPETITIONS,
                "record_duration": config.KAYIT_SURESI_SN,
                "quality_threshold": config.QUALITY_THRESHOLD
            }
        }
        
        report_file = self.create_report("recording", user_id, operation_data)
        
        # Detailed log
        self.logger.info(
            f"Recording Session Completed | User: {user_id} | "
            f"Type: {record_type} | "
            f"Recorded: {stats.get('recorded_items', 0)}/{stats.get('total_items', 0)} | "
            f"Success Rate: {(stats.get('successful_recordings', 0) / max(stats.get('total_recordings', 1), 1) * 100):.1f}%"
        )
        
        return report_file
    
    def log_training_session(self, user_id: str, training_data: Dict[str, Any]) -> Path:
        """
        Log a training session with detailed metrics.
        
        Args:
            user_id: User ID
            training_data: Training data including metrics, parameters, etc.
        
        Returns:
            Path to created report file
        """
        operation_data = {
            "training_parameters": {
                "base_model": training_data.get("base_model", config.MODEL_NAME),
                "num_epochs": training_data.get("num_epochs", config.NUM_FINETUNE_EPOCHS),
                "batch_size": training_data.get("batch_size", config.FINETUNE_BATCH_SIZE),
                "learning_rate": training_data.get("learning_rate", config.FINETUNE_LEARNING_RATE),
                "gradient_accumulation_steps": training_data.get("gradient_accumulation_steps", config.GRADIENT_ACCUMULATION_STEPS),
                "adapter_reduction_factor": training_data.get("adapter_reduction_factor", config.ADAPTER_REDUCTION_FACTOR),
                "warmup_steps": training_data.get("warmup_steps", config.WARMUP_STEPS),
                "weight_decay": training_data.get("weight_decay", config.WEIGHT_DECAY),
                "early_stopping_patience": training_data.get("early_stopping_patience", config.EARLY_STOPPING_PATIENCE),
                "use_augmentation": training_data.get("use_augmentation", config.USE_AUGMENTATION),
                "mixed_precision": training_data.get("mixed_precision", getattr(config, "MIXED_PRECISION", "no"))
            },
            "dataset_info": {
                "train_samples": training_data.get("train_samples", 0),
                "eval_samples": training_data.get("eval_samples", 0),
                "total_samples": training_data.get("total_samples", 0)
            },
            "training_metrics": {
                "total_steps": training_data.get("total_steps", 0),
                "total_epochs_completed": training_data.get("epochs_completed", 0),
                "final_train_loss": training_data.get("final_train_loss"),
                "best_val_loss": training_data.get("best_val_loss"),
                "final_wer": training_data.get("final_wer"),
                "final_cer": training_data.get("final_cer"),
                "best_wer": training_data.get("best_wer"),
                "best_cer": training_data.get("best_cer"),
                "early_stopped": training_data.get("early_stopped", False)
            },
            "system_info": {
                "device": training_data.get("device", "cpu"),
                "gpu_name": training_data.get("gpu_name"),
                "gpu_memory_gb": training_data.get("gpu_memory_gb"),
                "platform": training_data.get("platform"),
                "python_version": training_data.get("python_version")
            },
            "training_duration": {
                "start_time": training_data.get("start_time"),
                "end_time": training_data.get("end_time"),
                "duration_seconds": training_data.get("duration_seconds"),
                "duration_formatted": training_data.get("duration_formatted")
            }
        }
        
        report_file = self.create_report("training", user_id, operation_data)
        
        # Detailed log
        final_wer = training_data.get("final_wer", 0)
        final_cer = training_data.get("final_cer", 0)
        self.logger.info(
            f"Training Session Completed | User: {user_id} | "
            f"Epochs: {training_data.get('epochs_completed', 0)}/{training_data.get('num_epochs', 0)} | "
            f"Final WER: {final_wer:.4f} ({final_wer*100:.2f}%) | "
            f"Final CER: {final_cer:.4f} ({final_cer*100:.2f}%) | "
            f"Best Val Loss: {training_data.get('best_val_loss', 'N/A')}"
        )
        
        return report_file
    
    def log_evaluation_session(self, user_id: str, evaluation_data: Dict[str, Any]) -> Path:
        """
        Log an evaluation session.
        
        Args:
            user_id: User ID
            evaluation_data: Evaluation metrics and results
        
        Returns:
            Path to created report file
        """
        operation_data = {
            "evaluation_metrics": {
                "wer": evaluation_data.get("wer", 0),
                "cer": evaluation_data.get("cer", 0),
                "total_samples": evaluation_data.get("total_samples", 0),
                "evaluated_samples": evaluation_data.get("evaluated_samples", 0)
            },
            "model_info": {
                "model_path": evaluation_data.get("model_path"),
                "base_model": evaluation_data.get("base_model")
            },
            "sample_predictions": evaluation_data.get("sample_predictions", [])
        }
        
        report_file = self.create_report("evaluation", user_id, operation_data)
        
        # Detailed log
        wer = evaluation_data.get("wer", 0)
        cer = evaluation_data.get("cer", 0)
        self.logger.info(
            f"Evaluation Completed | User: {user_id} | "
            f"WER: {wer:.4f} ({wer*100:.2f}%) | "
            f"CER: {cer:.4f} ({cer*100:.2f}%) | "
            f"Samples: {evaluation_data.get('evaluated_samples', 0)}"
        )
        
        return report_file
    
    def log_general_operation(self, operation_type: str, user_id: str,
                             description: str, data: Optional[Dict[str, Any]] = None) -> Path:
        """
        Log a general operation.
        
        Args:
            operation_type: Type of operation
            user_id: User ID
            description: Operation description
            data: Additional data
        
        Returns:
            Path to created report file
        """
        operation_data = {
            "description": description,
            "additional_data": data or {}
        }
        
        report_file = self.create_report(operation_type, user_id, operation_data)
        
        self.logger.info(
            f"Operation: {operation_type} | User: {user_id} | "
            f"Description: {description}"
        )
        
        return report_file
