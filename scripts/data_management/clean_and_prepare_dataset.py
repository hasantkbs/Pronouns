# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
import os
import shutil

def clean_and_merge_data(source_users, target_user, max_reps=10):
    all_data = []
    
    for user in source_users:
        metadata_path = Path(f"data/users/{user}/metadata_words.csv")
        if not metadata_path.exists():
            print(f"⚠️  {user} için metadata bulunamadı.")
            continue
            
        print(f"🔍 {user} verileri işleniyor...")
        df = pd.read_csv(metadata_path)
        
        # 1. Sentetik verileri ele (Eğer varsa - dosya yolunda 'augment' veya 'synth' geçenler)
        original_count = len(df)
        df = df[~df['file_path'].str.contains('augment|synth', case=False, na=False)]
        removed_synth = original_count - len(df)
        if removed_synth > 0:
            print(f"   - {removed_synth} adet sentetik/üretilmiş veri elendi.")
            
        def check_file(path, user):
            p = Path(path)
            if p.exists(): return str(p.absolute())
            
            # Farklı varyasyonları dene
            variants = [
                Path("data/users") / user / p.name,
                Path("data/users") / user / "words" / p.name,
                Path("data/users") / user / "words" / p.parent.name / p.name
            ]
            for v in variants:
                if v.exists(): return str(v.absolute())
            return None

        df['file_path'] = df.apply(lambda x: check_file(x['file_path'], user), axis=1)
        df = df[df['file_path'].notna()]
        
        all_data.append(df)
        
    if not all_data:
        print("❌ Hiç veri bulunamadı!")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 3. Kelime bazında en iyi 10 kaydı seç (Quality Score'a göre)
    print(f"📊 Toplam gerçek kayıt sayısı: {len(combined_df)}")
    
    # Transcription sütununu normalize et
    combined_df['transcription'] = combined_df['transcription'].astype(str).str.lower().str.strip()
    
    # Quality score yoksa 100 ver (varsayılan)
    if 'quality_score' not in combined_df.columns:
        combined_df['quality_score'] = 100
        
    # Kelime bazında grupla ve en iyi N tanesini al
    final_df = combined_df.sort_values('quality_score', ascending=False).groupby('transcription').head(max_reps)
    
    print(f"✅ Filtreleme sonrası (Kelime başı max {max_reps} kayıt): {len(final_df)} satır.")
    
    # Hedef klasörü oluştur
    target_path = Path(f"data/users/{target_user}")
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Yeni metadata'yı kaydet
    final_metadata_path = target_path / "metadata_words.csv"
    final_df.to_csv(final_metadata_path, index=False, encoding='utf-8')
    print(f"💾 Temizlenmiş veri kaydedildi: {final_metadata_path}")

if __name__ == "__main__":
    # 'Furkan' ve 'FurkanV1' verilerini birleştirip 'FurkanV1_Clean' adıyla yeni bir user oluşturalım
    clean_and_merge_data(['Furkan', 'FurkanV1'], 'FurkanV1_Clean', max_reps=10)
