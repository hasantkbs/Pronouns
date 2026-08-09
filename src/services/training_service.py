# -*- coding: utf-8 -*-
import os
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import config
from src.services.reporting_service import ReportingService
from src.services.settings_service import SettingsService

class TrainingService:
    """Eğitim süreçlerini yöneten servis."""
    
    # In-memory durum takibi
    _statuses: Dict[str, Dict[str, Any]] = {}

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.reporting_service = ReportingService()
        self.settings_service = SettingsService()
        
        if user_id not in self._statuses:
            self._statuses[user_id] = {
                "status": "idle",
                "progress": 0,
                "start_time": None,
                "end_time": None,
                "error": None,
                "last_wer": self.get_last_wer()
            }

    def get_last_wer(self) -> Optional[float]:
        """Kullanıcının en son tamamlanan eğitiminden WER değerini döndürür."""
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return None
        
        # Kullanıcının eğitim raporlarını bul (en yeniden en eskiye)
        reports = sorted(
            reports_dir.glob(f"training_{self.user_id}_*.json"),
            key=os.path.getmtime,
            reverse=True
        )
        
        for report_path in reports:
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                    # reporting_service.log_training_session yapısına göre:
                    # report['data']['training_metrics']['final_wer']
                    metrics = report.get("data", {}).get("training_metrics", {})
                    wer = metrics.get("final_wer") or metrics.get("best_wer")
                    if wer is not None:
                        return float(wer)
            except Exception:
                continue
        return None

    def get_status(self) -> Dict[str, Any]:
        """Eğitim durumunu döndürür."""
        status = self._statuses.get(self.user_id, {"status": "idle"}).copy()
        # Her sorguda son WER'i güncelle (eğer eğitim bittiyse)
        if status["status"] == "idle":
            status["last_wer"] = self.get_last_wer()
        return status

    def start_training(self, background_tasks=None) -> bool:
        """Eğitimi başlatır."""
        current_status = self._statuses.get(self.user_id, {})
        if current_status.get("status") == "training":
            return False

        self._statuses[self.user_id].update({
            "status": "training",
            "progress": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "error": None
        })

        # PersonalizedTrainer'ı içe aktar (döngüsel bağımlılığı önlemek için)
        from src.cli.train_adapter import PersonalizedTrainer

        def run_training():
            try:
                # Ayarları yükle
                settings = self.settings_service.load_settings(self.user_id)
                
                # Config'i geçici olarak güncelle (opsiyonel, trainer config'den okuyor)
                # Not: train_adapter.py şu an config.py'den direkt okuyor. 
                # Gelecekte trainer'a settings objesi geçilebilir.
                
                trainer = PersonalizedTrainer(user_id=self.user_id)
                trainer.run()
                
                self._statuses[self.user_id].update({
                    "status": "idle",
                    "progress": 100,
                    "end_time": datetime.now().isoformat(),
                    "last_wer": self.get_last_wer()
                })
            except Exception as e:
                self._statuses[self.user_id].update({
                    "status": "error",
                    "error": str(e),
                    "end_time": datetime.now().isoformat()
                })

        if background_tasks:
            background_tasks.add_task(run_training)
        else:
            thread = threading.Thread(target=run_training, daemon=True)
            thread.start()
            
        return True
