# -*- coding: utf-8 -*-
"""
Model Service - Model yönetimi işlemlerini yönetir
"""

from pathlib import Path
from typing import Optional
import config
from src.constants import MODEL_BASE_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR


class ModelService:
    """Model yönetimi için servis"""
    
    @staticmethod
    def find_personalized_model(user_id: str) -> Optional[str]:
        """
        Kullanıcı için kişiselleştirilmiş model yolunu bulur.
        
        Args:
            user_id: Kullanıcı ID'si
        
        Returns:
            str: Model yolu veya None
        """
        # Mevcut ve eski dizinleri kontrol et
        personalized_roots = [
            Path("data/models/personalized_models"),
            Path("models/personalized_models"),
        ]
        
        for root in personalized_roots:
            base_path = root / user_id
            if base_path.exists() and any(base_path.iterdir()):
                # Önce ana dizindeki adapter dosyalarını kontrol et
                if (base_path / "adapter_model.bin").exists() or (base_path / "adapter_model.safetensors").exists():
                    return str(base_path)
                
                # Sonra best model checkpoint'i kontrol et
                best_ckpt_path = base_path / CHECKPOINT_DIR / BEST_MODEL_DIR
                if best_ckpt_path.exists():
                    if (best_ckpt_path / "adapter_model.bin").exists() or (best_ckpt_path / "adapter_model.safetensors").exists():
                        return str(best_ckpt_path)
                    
                # Herhangi bir dosya varsa dizini döndür
                return str(base_path)
        
        return None
    
    @staticmethod
    def get_model_path(user_id: str) -> Path:
        """Model kayıt dizinini döndürür"""
        return Path(MODEL_BASE_DIR) / user_id
    
    @staticmethod
    def model_exists(user_id: str) -> bool:
        """Kullanıcı için model var mı kontrol eder"""
        return ModelService.find_personalized_model(user_id) is not None
