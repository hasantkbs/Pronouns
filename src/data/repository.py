# -*- coding: utf-8 -*-
"""
Data Access Layer - Repository Pattern
Veri erişim işlemlerini merkezileştirir
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Set
import config
from src.constants import (
    METADATA_FILENAMES, USER_DATA_SUBDIRS, METADATA_COLUMNS,
    RECORD_TYPE_WORD, RECORD_TYPE_SENTENCE, RECORD_TYPE_LETTER
)


class UserDataRepository:
    """Kullanıcı verilerine erişim için repository sınıfı"""
    
    def __init__(self, base_path: str = None):
        """
        Repository'yi başlatır.
        
        Args:
            base_path: Veri dizini yolu (None ise config'den alınır)
        """
        self.base_path = Path(base_path or config.BASE_PATH)
    
    def get_user_path(self, user_id: str) -> Path:
        """Kullanıcı dizin yolunu döndürür"""
        return self.base_path / user_id
    
    def get_metadata_path(self, user_id: str, record_type: str) -> Path:
        """Metadata dosya yolunu döndürür"""
        user_path = self.get_user_path(user_id)
        filename = METADATA_FILENAMES.get(record_type, "metadata.csv")
        return user_path / filename
    
    def get_save_path(self, user_id: str, record_type: str) -> Path:
        """Kayıt dizin yolunu döndürür"""
        user_path = self.get_user_path(user_id)
        subdir = USER_DATA_SUBDIRS.get(record_type, "audio")
        return user_path / subdir
    
    def load_metadata(self, user_id: str, record_type: str) -> Optional[pd.DataFrame]:
        """
        Metadata dosyasını yükler.
        
        Returns:
            DataFrame veya None (dosya yoksa)
        """
        metadata_path = self.get_metadata_path(user_id, record_type)
        
        if not metadata_path.exists():
            return None
        
        try:
            df = pd.read_csv(metadata_path)
            return df
        except (pd.errors.EmptyDataError, Exception) as e:
            print(f"⚠️  Metadata yüklenirken hata: {e}")
            return None
    
    def save_metadata(self, user_id: str, record_type: str, metadata: List[Dict], 
                     append: bool = True) -> bool:
        """
        Metadata'yı kaydeder.
        
        Args:
            user_id: Kullanıcı ID'si
            record_type: Kayıt türü
            metadata: Metadata listesi
            append: Mevcut dosyaya ekle (True) veya üzerine yaz (False)
        
        Returns:
            bool: Başarılı ise True
        """
        if not metadata:
            return False
        
        metadata_path = self.get_metadata_path(user_id, record_type)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        new_df = pd.DataFrame(metadata)
        
        if append and metadata_path.exists():
            try:
                existing_df = pd.read_csv(metadata_path)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Yinelenen satırları temizle
                updated_df.drop_duplicates(
                    subset=['file_path', 'transcription', 'repetition'], 
                    inplace=True
                )
            except pd.errors.EmptyDataError:
                updated_df = new_df
        else:
            updated_df = new_df
        
        try:
            updated_df.to_csv(metadata_path, index=False, encoding='utf-8')
            return True
        except Exception as e:
            print(f"❌ Metadata kaydedilirken hata: {e}")
            return False
    
    def get_recorded_items(self, user_id: str, record_type: str) -> Set[str]:
        """
        Kayıtlı item'ları (kelime/cümle/harf) döndürür.
        
        Returns:
            Set of recorded transcriptions
        """
        df = self.load_metadata(user_id, record_type)
        if df is None or 'transcription' not in df.columns:
            return set()
        return set(df['transcription'].dropna().unique())
    
    def get_recorded_count(self, user_id: str, record_type: str, 
                          transcription: str) -> int:
        """
        Belirli bir item için kayıt sayısını döndürür.
        
        Args:
            user_id: Kullanıcı ID'si
            record_type: Kayıt türü
            transcription: Item (kelime/cümle/harf)
        
        Returns:
            int: Kayıt sayısı
        """
        df = self.load_metadata(user_id, record_type)
        if df is None or 'transcription' not in df.columns:
            return 0
        
        item_records = df[df['transcription'] == transcription]
        return len(item_records)
    
    def get_recorded_details(self, user_id: str, record_type: str) -> Dict[str, int]:
        """
        Her item için kayıt sayılarını döndürür.
        
        Returns:
            Dict: {transcription: count}
        """
        df = self.load_metadata(user_id, record_type)
        if df is None or 'transcription' not in df.columns:
            return {}
        
        details = {}
        for transcription in df['transcription'].dropna().unique():
            item_records = df[df['transcription'] == transcription]
            details[transcription] = len(item_records)
        
        return details
    
    def user_exists(self, user_id: str) -> bool:
        """Kullanıcı dizini var mı kontrol eder"""
        return self.get_user_path(user_id).exists()
    
    def create_user_directory(self, user_id: str) -> Path:
        """Kullanıcı dizinini oluşturur"""
        user_path = self.get_user_path(user_id)
        user_path.mkdir(parents=True, exist_ok=True)
        return user_path
