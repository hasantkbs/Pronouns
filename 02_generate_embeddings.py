# -*- coding: utf-8 -*-

import os
import json
import torch
import re
import config  # Ayarları import et
from utils import EmbeddingGenerator  # Yardımcı sınıfı import et


def main():
    print("--- Embedding Üretimi Başladı ---")

    # Model ve yardımcı sınıfı başlat
    embed_generator = EmbeddingGenerator(model_name=config.MODEL_NAME)

    user_path = os.path.join(config.BASE_PATH, config.USER_ID)
    embeddings_path = os.path.join(user_path, "embeddings")
    os.makedirs(embeddings_path, exist_ok=True)

    meta_path = os.path.join(user_path, "meta.json")

    # Önce kalibrasyondan gelen meta.json'ı oku
    if not os.path.exists(meta_path):
        print(f"Hata: '{meta_path}' bulunamadı. Lütfen önce 01_calibrate.py scriptini çalıştırın.")
        return

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    wav_files_info = meta_data.get("kelimeler", [])
    print(f"{len(wav_files_info)} adet .wav dosyası için işlem yapılacak.")

    for i, file_info in enumerate(wav_files_info):
        file_name = file_info["dosya"]
        etiket = file_info["etiket"]
        wav_path = os.path.join(user_path, file_name)

        print(f"\n({i+1}/{len(wav_files_info)}) İşleniyor: {file_name}")

        try:
            embedding_vector = embed_generator.get_embedding_from_file(wav_path)

            if embedding_vector is None or embedding_vector.numel() == 0:
                print(f"  Uyarı: '{file_name}' için geçerli embedding üretilemedi. Atlanıyor.")
                continue

            embedding_file_name = f"{os.path.splitext(file_name)[0]}.pt"
            embedding_save_path = os.path.join(embeddings_path, embedding_file_name)
            torch.save(embedding_vector, embedding_save_path)
            print(f"  Embedding kaydedildi: {embedding_save_path}")

        except Exception as e:
            print(f"  Hata: '{file_name}' işlenirken bir sorun oluştu: {e}")

    print("\n--- İşlem Tamamlandı ---")


if __name__ == "__main__":
    main()
