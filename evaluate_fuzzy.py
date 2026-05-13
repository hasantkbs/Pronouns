# evaluate_fuzzy.py
import os
import pandas as pd
import evaluate
from tqdm import tqdm
from src.core.asr import ASRSystem
import config
from pathlib import Path

def main(user_id="FurkanV1"):
    print(f"--- {user_id} için Bulanık Mantık (Fuzzy) Destekli Değerlendirme ---")
    
    # ASR Sistemini yükle (Kişiselleştirilmiş model ile)
    model_path = f"data/models/personalized_models/{user_id}"
    if not os.path.exists(model_path):
        print(f"Hata: Model bulunamadı: {model_path}")
        return

    asr = ASRSystem(model_name=model_path, user_id=user_id)
    
    # Metrikleri yükle
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    # Eval veri setini oku
    eval_csv = Path(config.BASE_PATH) / user_id / "eval.csv"
    if not eval_csv.exists():
        print(f"Hata: {eval_csv} bulunamadı.")
        return
        
    df = pd.read_csv(eval_csv)
    # Sadece var olan dosyaları kontrol et
    df = df[df["file_path"].apply(os.path.exists)]
    
    predictions = []
    references = []
    
    print(f"Toplam {len(df)} örnek işleniyor...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row["file_path"]
        reference = row["transcript"]
        
        # Transkripsiyon (Fuzzy matching içerde yapılıyor)
        text, _ = asr.transcribe(audio_path)
        
        if text:
            predictions.append(text)
            references.append(reference)
        else:
            # Sessizlik veya hata durumunda boş tahmin
            predictions.append("")
            references.append(reference)

    # Metrikleri hesapla
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    
    print("\n" + "="*50)
    print(f"SONUÇLAR (Fuzzy Matching AKTİF):")
    print(f"WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
    print("="*50)
    
    # Karşılaştırmalı örnekler
    print("\nÖrnek Düzeltmeler:")
    for i in range(min(10, len(predictions))):
        if predictions[i] == references[i]:
            status = "✅ DOĞRU"
        else:
            status = "❌ HATALI"
        print(f"{i+1}. [Ref: {references[i]}] -> [Tahmin: {predictions[i]}] | {status}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, default="FurkanV1")
    args = parser.parse_args()
    main(args.user_id)
