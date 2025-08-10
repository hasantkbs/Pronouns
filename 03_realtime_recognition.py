# -*- coding: utf-8 -*-
import os
import json
import torch

import config
from utils import EmbeddingGenerator
from train_classifier import EnhancedClassifier  # Eğitilmiş sınıfı import et


def load_classifier_and_labels(user_path: str):
    """Eğitilmiş sınıflandırıcı modelini ve etiketleri yükler."""
    model_path = os.path.join(user_path, "classifier_model.pth")
    labels_path = os.path.join(user_path, "label_encoder.json")

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None

    # Etiketleri yükle
    with open(labels_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    output_dim = len(class_names)

    # Kaydedilmiş EnhancedClassifier modelini yükle
    classifier_model = EnhancedClassifier(
        input_dim=config.EMBEDDING_DIM,
        hidden_dim1=512,
        hidden_dim2=256,
        output_dim=output_dim
    )
    classifier_model.load_state_dict(torch.load(model_path, weights_only=True))
    classifier_model.eval()

    return classifier_model, class_names


def main():
    """Ana tanıma ve test fonksiyonu."""
    user_path = os.path.join(config.BASE_PATH, config.USER_ID)

    # Model ve etiketleri yükle
    classifier, class_names = load_classifier_and_labels(user_path)
    if classifier is None:
        print(f"Hata: Kullanıcı '{config.USER_ID}' için eğitilmiş model bulunamadı.")
        return

    embed_generator = EmbeddingGenerator(model_name=config.MODEL_NAME)

    meta_path = os.path.join(user_path, "meta.json")
    if not os.path.exists(meta_path):
        print("Hata: Test edilecek kelimeleri içeren meta.json dosyası bulunamadı.")
        return

    # Test verilerini oku
    with open(meta_path, "r", encoding="utf-8") as f:
        test_data = json.load(f).get("kelimeler", [])

    print(f"--- Kullanıcı '{config.USER_ID}' için Konuşma Tanıma Testi Başladı ---")
    correct_predictions = 0

    # Her dosya için test yap
    for item in test_data:
        wav_file = item["dosya"]
        expected_word = item["etiket"]
        test_wav_path = os.path.join(user_path, wav_file)

        if not os.path.exists(test_wav_path):
            continue

        print(f"\n-> Test ediliyor: '{wav_file}' (Beklenen: '{expected_word}')")

        embedding = embed_generator.get_embedding_from_file(test_wav_path)
        if embedding is None or embedding.numel() == 0:
            continue

        # Model ile tahmin yap
        with torch.no_grad():
            outputs = classifier(embedding.unsqueeze(0))
            _, predicted_idx = torch.max(outputs.data, 1)
            predicted_word = class_names[predicted_idx.item()]

        print(f"  Tahmin edilen: '{predicted_word}'")
        if predicted_word == expected_word:
            print("  Sonuç: ✅ DOĞRU")
            correct_predictions += 1
        else:
            print("  Sonuç: ❌ YANLIŞ")

    # Doğruluk oranını yazdır
    if test_data:
        total_accuracy = correct_predictions / len(test_data)
        print("\n--- Test Tamamlandı ---")
        print(f"Genel Doğruluk: {total_accuracy:.2%}")


if __name__ == "__main__":
    main()
