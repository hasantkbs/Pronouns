# -*- coding: utf-8 -*-
"""
Recording Service - Kayıt işlemlerini yönetir
Business logic katmanı
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import config
from src.data.repository import UserDataRepository
from src.utils.utils import record_audio, calculate_audio_quality, play_audio, check_consistency, normalize_path_for_cross_platform
from src.constants import (
    RECORD_TYPE_WORD, RECORD_TYPE_SENTENCE, RECORD_TYPE_LETTER,
    DEFAULT_REPETITIONS
)


class RecordingService:
    """Ses kayıt işlemlerini yöneten servis"""
    
    def __init__(self, user_id: str):
        """
        Recording service'i başlatır.
        
        Args:
            user_id: Kullanıcı ID'si
        """
        self.user_id = user_id
        self.repository = UserDataRepository()
    
    def get_recording_stats(self, record_type: str) -> Dict:
        """
        Kayıt istatistiklerini döndürür.
        
        Returns:
            Dict: İstatistik bilgileri
        """
        recorded_items = self.repository.get_recorded_items(self.user_id, record_type)
        recorded_details = self.repository.get_recorded_details(self.user_id, record_type)
        
        # Yeterli kaydı olan itemler
        ideal_reps = config.IDEAL_REPETITIONS if record_type == RECORD_TYPE_WORD else DEFAULT_REPETITIONS.get(record_type, 5)
        fully_recorded = {
            item for item, count in recorded_details.items() 
            if count >= ideal_reps
        }
        
        return {
            "total_recorded": len(recorded_items),
            "fully_recorded": len(fully_recorded),
            "partially_recorded": len(recorded_items) - len(fully_recorded),
            "recorded_details": recorded_details,
            "fully_recorded_items": fully_recorded
        }
    
    def get_items_to_record(self, items_list: List[str], record_type: str, 
                            re_record: bool = False) -> Tuple[List[str], Dict]:
        """
        Kaydedilecek item'ları belirler.
        
        Args:
            items_list: Tüm item listesi
            record_type: Kayıt türü
            re_record: Yeniden kayıt modu
        
        Returns:
            Tuple: (kaydedilecek_itemler, istatistikler)
        """
        stats = self.get_recording_stats(record_type)
        fully_recorded = stats["fully_recorded_items"]
        
        if re_record:
            items_to_record = items_list
        else:
            items_to_record = [item for item in items_list if item not in fully_recorded]
        
        return items_to_record, stats
    
    def get_remaining_repetitions(self, transcription: str, record_type: str) -> int:
        """
        Belirli bir item için eksik tekrar sayısını döndürür.
        
        Args:
            transcription: Item (kelime/cümle/harf)
            record_type: Kayıt türü
        
        Returns:
            int: Eksik tekrar sayısı
        """
        current_count = self.repository.get_recorded_count(
            self.user_id, record_type, transcription
        )
        
        ideal_reps = config.IDEAL_REPETITIONS if record_type == RECORD_TYPE_WORD else DEFAULT_REPETITIONS.get(record_type, 5)
        return max(0, ideal_reps - current_count)
    
    def record_item(self, transcription: str, record_type: str, rep_num: int,
                   record_duration: int) -> Optional[Dict]:
        """
        Tek bir item kaydı yapar.
        
        Args:
            transcription: Item (kelime/cümle/harf)
            record_type: Kayıt türü
            rep_num: Tekrar numarası
            record_duration: Kayıt süresi (saniye)
        
        Returns:
            Dict: Kayıt bilgileri veya None (başarısız)
        """
        save_path = self.repository.get_save_path(self.user_id, record_type)
        
        # Dosya yolu oluştur
        if record_type == RECORD_TYPE_WORD:
            word_dir = save_path / transcription
            word_dir.mkdir(parents=True, exist_ok=True)
            file_path = word_dir / f"rep{rep_num}.wav"
        elif record_type == RECORD_TYPE_SENTENCE:
            # Cümle için dosya numarası gerekli (burada basitleştirilmiş)
            file_path = save_path / f"{self.user_id}_{record_type}_{rep_num}.wav"
        else:  # LETTER
            file_path = save_path / f"{self.user_id}_{record_type}_{rep_num}.wav"
        
        # Kayıt yap
        recorded_file = record_audio(
            file_path=str(file_path), 
            record_seconds=record_duration
        )
        
        if not recorded_file:
            return None
        
        # Kalite kontrolü
        quality_info = calculate_audio_quality(recorded_file)
        
        # Relative path (cross-platform)
        relative_path = self._normalize_path(str(file_path.absolute()), save_path.parent)
        
        return {
            "file_path": relative_path,
            "transcription": transcription,
            "repetition": rep_num,
            "quality_score": quality_info['quality_score'],
            "rms": quality_info['rms'],
            "snr_db": quality_info['snr_db'],
            "duration": quality_info['duration'],
            "is_valid": quality_info['is_valid']
        }
    
    def save_recording_metadata(self, record_type: str, metadata: List[Dict], 
                                append: bool = True) -> bool:
        """Kayıt metadata'sını kaydeder"""
        return self.repository.save_metadata(
            self.user_id, record_type, metadata, append
        )
    
    @staticmethod
    def _normalize_path(file_path: str, base_path: Path) -> str:
        """Cross-platform path normalizasyonu"""
        file_path_obj = Path(file_path)
        
        if file_path_obj.is_absolute():
            try:
                relative_path = file_path_obj.relative_to(base_path)
                return str(relative_path)
            except ValueError:
                return str(file_path_obj)
        
        return str(file_path_obj)
