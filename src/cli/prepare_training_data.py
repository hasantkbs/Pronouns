#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Eğitim verisi hazırlama scripti.
Kullanıcının metadata_words.csv dosyasından train.csv ve eval.csv dosyalarını oluşturur.
"""

import os
import sys
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import config

def prepare_training_data(user_id, test_size=0.2, random_state=42):
    """
    Kullanıcı verilerini eğitim ve değerlendirme setlerine ayırır.
    
    Args:
        user_id: Kullanıcı kimliği (örn: "Furkan")
        test_size: Değerlendirme seti oranı (varsayılan: 0.2 = %20)
        random_state: Rastgelelik tohumu (tekrarlanabilirlik için)
    
    Returns:
        tuple: (train_df, eval_df) veya (None, None) hata durumunda
    """
    base_path = Path(config.BASE_PATH)
    user_path = base_path / user_id
    
    # Metadata dosyasını kontrol et
    metadata_path = user_path / "metadata_words.csv"
    if not metadata_path.exists():
        print(f"❌ Hata: {metadata_path} bulunamadı!")
        return None, None
    
    print(f"📊 Veri hazırlama başlatılıyor: {user_id}")
    print(f"   Metadata dosyası: {metadata_path}")
    
    try:
        # Metadata'yı yükle
        df = pd.read_csv(metadata_path, encoding='utf-8')
        print(f"   Toplam kayıt sayısı: {len(df)}")
        
        # Gerekli sütunları kontrol et
        # 'transcription' veya 'transcript' sütunlarından biri olmalı
        transcript_col = 'transcription' if 'transcription' in df.columns else 'transcript'
        if 'file_path' not in df.columns or transcript_col not in df.columns:
            print(f"❌ Hata: Gerekli sütunlar bulunamadı. Mevcut: {df.columns.tolist()}")
            return None, None
        
        # Dosya yollarını platforma göre çöz
        from src.utils.utils import resolve_audio_path
        
        # NaN değerleri temizle
        df = df.dropna(subset=['file_path'])
        
        df['file_path'] = df['file_path'].apply(
            lambda p: resolve_audio_path(str(p), user_path) if pd.notna(p) else None
        )
        
        # Var olmayan dosyaları filtrele
        original_size = len(df)
        df = df[df['file_path'].notna()]
        df = df[df['file_path'].apply(lambda x: os.path.exists(str(x)))]
        removed_count = original_size - len(df)
        
        if removed_count > 0:
            print(f"⚠️  {removed_count} adet bulunamayan veya geçersiz ses dosyası atlandı.")
        
        if len(df) == 0:
            print(f"❌ Hata: Hiç geçerli ses dosyası bulunamadı!")
            return None, None
        
        # Sütunları seç ve yeniden adlandır
        df = df[['file_path', transcript_col]].copy()
        df.rename(columns={transcript_col: 'transcript'}, inplace=True)
        
        # Boş transkriptleri filtrele
        df = df[df['transcript'].notna() & (df['transcript'].str.strip() != '')]
        
        if len(df) == 0:
            print(f"❌ Hata: Hiç geçerli transkript bulunamadı!")
            return None, None
        
        # Eğitim ve değerlendirme setlerine ayır
        train_df, eval_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        
        # CSV dosyalarını kaydet
        train_csv_path = user_path / "train.csv"
        eval_csv_path = user_path / "eval.csv"
        
        train_df.to_csv(train_csv_path, index=False, encoding='utf-8')
        eval_df.to_csv(eval_csv_path, index=False, encoding='utf-8')
        
        print(f"\n✅ Veri hazırlama tamamlandı!")
        print(f"   📁 Eğitim seti: {train_csv_path} ({len(train_df)} kayıt)")
        print(f"   📁 Değerlendirme seti: {eval_csv_path} ({len(eval_df)} kayıt)")
        print(f"   📊 Eğitim/Değerlendirme oranı: {len(train_df)}/{len(eval_df)}")
        
        return train_df, eval_df
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(
        description="Kullanıcı verilerini eğitim ve değerlendirme setlerine ayırır."
    )
    parser.add_argument(
        "user_id",
        type=str,
        help="Kullanıcı kimliği (örn: Furkan)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Değerlendirme seti oranı (varsayılan: 0.2)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Rastgelelik tohumu (varsayılan: 42)"
    )
    
    args = parser.parse_args()
    
    prepare_training_data(
        user_id=args.user_id,
        test_size=args.test_size,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()
