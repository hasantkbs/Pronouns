# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from typing import Dict, Any
import config

class SettingsService:
    """Kullanıcı ayarlarını kalıcı olarak saklayan servis."""
    
    @staticmethod
    def get_settings_path(user_id: str) -> Path:
        """Kullanıcı ayarları dosyasının yolunu döndürür."""
        path = Path(config.BASE_PATH) / user_id
        path.mkdir(parents=True, exist_ok=True)
        return path / "settings.json"

    @staticmethod
    def load_settings(user_id: str) -> Dict[str, Any]:
        """Kullanıcının ayarlarını yükler. Dosya yoksa varsayılanları döner."""
        path = SettingsService.get_settings_path(user_id)
        
        default_settings = {
            "model": user_id,
            "algorithm": "lora",
            "self_learning": True,
            "learning_rate": float(config.FINETUNE_LEARNING_RATE),
            "epochs": int(config.NUM_FINETUNE_EPOCHS),
            "batch_size": int(config.FINETUNE_BATCH_SIZE),
        }

        if not path.exists():
            return default_settings

        try:
            with open(path, "r", encoding="utf-8") as f:
                saved_settings = json.load(f)
                # Varsayılanları güncelle
                default_settings.update(saved_settings)
                return default_settings
        except Exception:
            return default_settings

    @staticmethod
    def save_settings(user_id: str, settings: Dict[str, Any]) -> bool:
        """Kullanıcı ayarlarını dosyaya kaydeder."""
        path = SettingsService.get_settings_path(user_id)
        try:
            # Mevcut ayarları yükle ve güncelle (tamamen ezmemek için)
            current = SettingsService.load_settings(user_id)
            current.update(settings)
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=4, ensure_ascii=False)
            return True
        except Exception:
            return False
