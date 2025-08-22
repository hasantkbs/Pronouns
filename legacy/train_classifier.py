# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
import config

from model import EnhancedClassifier

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def load_data(user_path, embeddings_dir, meta_path):
    print("--- Veri Yükleme Başladı ---")
    with open(meta_path, 'r', encoding='utf-8') as f: meta_data = json.load(f)
    embeddings, labels = [], []
    for info in meta_data["kelimeler"]:
        embedding_path = os.path.join(embeddings_dir, f"{os.path.splitext(info['dosya'])[0]}.pt")
        if os.path.exists(embedding_path):
            embedding = torch.load(embedding_path, weights_only=True)
            if embedding.ndim > 1: embedding = embedding.mean(dim=0)
            embeddings.append(embedding)
            labels.append(info["etiket"])
    print(f"--- Veri Yükleme Tamamlandı: {len(embeddings)} embedding yüklendi ---")
    return embeddings, labels

    # --- YENİDEN DÜZENLENMİŞ main FONKSİYONU ---
def main():
    user_path = os.path.join(config.BASE_PATH, config.USER_ID)
    embeddings_dir = os.path.join(user_path, "embeddings")
    meta_path = os.path.join(user_path, "meta.json")

    # 1. Tüm veriyi yükle
    all_embeddings, all_labels = load_data(user_path, embeddings_dir, meta_path)
    if not all_labels:
        print("Eğitilecek veri bulunamadı.")
        return

    # 2. Veriyi filtrele (az sayıda örneği olan sınıfları çıkar)
    label_counts = Counter(all_labels)

    final_embeddings = []
    final_labels = []
    for i, emb in enumerate(all_embeddings):
        if label_counts[all_labels[i]] >= config.MIN_SAMPLES_PER_CLASS:
            final_embeddings.append(emb)
            final_labels.append(all_labels[i])

    if not final_embeddings:
        print("Filtreleme sonrası eğitim için yeterli veri kalmadı.")
        return

    # 3. Veriyi PyTorch Tensor'larına çevir ve etiketleri kodla
    embeddings_tensor = torch.stack(final_embeddings)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(final_labels)
    output_dim = len(label_encoder.classes_)
    print(f"Toplam {output_dim} benzersiz etiket ile eğitim yapılacak.")

    # 4. Veriyi Eğitim ve Doğrulama setlerine ayır
    X_train, X_val, y_train, y_val = train_test_split(embeddings_tensor, labels_encoded, test_size=config.TEST_SIZE, random_state=42,stratify=labels_encoded)

    X_train, y_train = X_train.float(), torch.from_numpy(y_train).long()
    X_val, y_val = X_val.float(), torch.from_numpy(y_val).long()

    # 5. Modeli, optimizatörü ve Early Stopping'i hazırla
    model = EnhancedClassifier(input_dim=config.EMBEDDING_DIM, hidden_dim1=config.CLASSIFIER_HIDDEN_DIM_1, hidden_dim2=config.CLASSIFIER_HIDDEN_DIM_2,output_dim=output_dim, dropout_rate=0.5)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model_save_path = os.path.join(user_path, "classifier_model.pth")
    early_stopping = EarlyStopping(patience=20, verbose=True, path=model_save_path)

    # 6. Eğitim döngüsünü başlat
    print("--- Geliştirilmiş Model Eğitimi Başlatılıyor (Early Stopping Aktif) ---")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        print(f'Epoch: {epoch} \tTraining Loss: {loss.item():.6f} \tValidation Loss: {val_loss.item(): .6f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 7. En iyi modeli yükle ve test et
    model.load_state_dict(torch.load(model_save_path, weights_only=True))

    print("--- Eğitim Tamamlandı, Model Test Ediliyor ---")
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_val).sum().item() / y_val.size(0)
        print(f'En İyi Modelin Test Doğruluğu: {accuracy:.2%}')

    # 8. Label encoder'ı kaydet
    with open(os.path.join(user_path, "label_encoder.json"), 'w', encoding='utf-8') as f:
        json.dump(list(label_encoder.classes_), f, ensure_ascii=False, indent=4)
    print("En iyi model ve LabelEncoder başarıyla kaydedildi.")


if __name__ == "__main__":
    main()
