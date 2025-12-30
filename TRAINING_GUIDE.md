# Model Eğitim Kılavuzu - Konuşma Bozukluğu için Optimize Edilmiş

## 🎯 Genel Bakış

Bu kılavuz, konuşma bozukluğu olan bireyler için kişiselleştirilmiş ASR modeli eğitimi için optimize edilmiş pipeline'ı açıklar.

## 📋 Eğitim Öncesi Hazırlık

### 1. Veri Hazırlama

Furkan kullanıcısı için veri hazırlama:

```bash
python prepare_training_data.py Furkan
```

Bu komut:
- `metadata_words.csv` dosyasını okur
- Veriyi %80 eğitim, %20 değerlendirme olarak böler
- `train.csv` ve `eval.csv` dosyalarını oluşturur

### 2. Veri Kontrolü

Eğitim öncesi veri kalitesini kontrol edin:
- Ses dosyalarının `data/users/Furkan/words/` klasöründe olduğundan emin olun
- `metadata_words.csv` dosyasının doğru formatta olduğunu kontrol edin
- Minimum veri miktarı: 100+ kayıt önerilir

## 🚀 Model Eğitimi

### Temel Eğitim

```bash
python train_adapter.py Furkan
```

### Gelişmiş Özellikler

Eğitim sırasında şunlar otomatik olarak yapılır:

1. **Veri Augmentation** (Konfigürasyonda aktifse):
   - Hafif Gaussian gürültü ekleme
   - Zaman esnetme (konuşma hızı varyasyonu)
   - Pitch değişimi (hafif)
   - Zaman maskesi

2. **Validation**:
   - Her 50 adımda bir validation yapılır
   - WER (Word Error Rate) ve CER (Character Error Rate) hesaplanır
   - En iyi model otomatik olarak kaydedilir

3. **Early Stopping**:
   - Validation loss iyileşmezse eğitim durdurulur
   - Overfitting'i önler
   - Patience: 5 epoch (config'de ayarlanabilir)

4. **Learning Rate Scheduling**:
   - Warmup: İlk 100 adımda learning rate kademeli artar
   - Sonrasında linear decay

## 📊 Eğitim Metrikleri

Eğitim sırasında şu metrikler takip edilir:

- **Training Loss**: Her epoch sonunda gösterilir
- **Validation Loss**: Her 50 adımda bir hesaplanır
- **WER**: Kelime hata oranı (düşük = iyi)
- **CER**: Karakter hata oranı (düşük = iyi)

### İyi Performans Göstergeleri

- WER < 0.15 (%15'ten az kelime hatası)
- CER < 0.05 (%5'ten az karakter hatası)
- Validation loss training loss'a yakın (overfitting yok)

## ⚙️ Hyperparameter Ayarları

### Önerilen Ayarlar (config.py)

```python
# Konuşma bozukluğu için optimize edilmiş
NUM_FINETUNE_EPOCHS = 20          # Daha fazla epoch
FINETUNE_BATCH_SIZE = 4           # Daha büyük batch
FINETUNE_LEARNING_RATE = 5e-5     # Daha düşük LR (stabilite için)
ADAPTER_REDUCTION_FACTOR = 16    # Daha fazla parametre
GRADIENT_ACCUMULATION_STEPS = 4   # Efektif batch = 16
WARMUP_STEPS = 100                # Learning rate warmup
EARLY_STOPPING_PATIENCE = 5       # Early stopping
USE_AUGMENTATION = True           # Augmentation aktif
```

### Ayarlama İpuçları

**Düşük doğruluk durumunda:**
- Epoch sayısını artırın (20 → 30)
- Learning rate'i düşürün (5e-5 → 3e-5)
- Daha fazla veri toplayın
- Augmentation'ı aktif tutun

**Overfitting durumunda:**
- Early stopping patience'ı azaltın (5 → 3)
- Weight decay'i artırın (1e-3 → 5e-3)
- Augmentation'ı artırın
- Daha fazla veri toplayın

**Eğitim çok yavaşsa:**
- Batch size'ı artırın (4 → 8)
- Gradient accumulation'ı azaltın (4 → 2)
- Augmentation'ı kapatın (geçici olarak)

## 🔍 Model Değerlendirme

Eğitim sonrası modeli değerlendirin:

```bash
python evaluate_model.py Furkan
```

Sadece ilk 100 örnek için:
```bash
python evaluate_model.py Furkan --max_samples 100
```

## 📁 Çıktı Dosyaları

Eğitim sonrası şu dosyalar oluşturulur:

```
data/models/personalized_models/Furkan/
├── adapter_config.json          # LoRA adapter konfigürasyonu
├── adapter_model.bin            # Adapter ağırlıkları
└── checkpoints/
    └── best_model/              # En iyi model checkpoint'i
```

## 🐛 Sorun Giderme

### Eğitim sırasında hata

1. **CUDA out of memory**:
   - Batch size'ı azaltın (4 → 2)
   - Gradient accumulation'ı artırın (4 → 8)

2. **Validation loss artıyor**:
   - Learning rate'i düşürün
   - Early stopping çalışıyor olabilir (normal)

3. **WER/CER çok yüksek**:
   - Daha fazla veri toplayın
   - Epoch sayısını artırın
   - Model checkpoint'lerini kontrol edin

### Veri sorunları

1. **Ses dosyaları bulunamıyor**:
   - `prepare_training_data.py` çalıştırın
   - Dosya yollarını kontrol edin

2. **Boş transkriptler**:
   - `metadata_words.csv` dosyasını kontrol edin
   - Boş satırları temizleyin

## 💡 İpuçları

1. **İlk eğitim**: Varsayılan ayarlarla başlayın
2. **İteratif iyileştirme**: Her eğitimden sonra değerlendirme yapın
3. **Veri kalitesi**: Temiz, net kayıtlar önemli
4. **Düzenli checkpoint**: En iyi model otomatik kaydedilir
5. **Monitoring**: Eğitim sırasında metrikleri takip edin

## 📈 Performans İyileştirme Stratejisi

1. **Başlangıç**: Varsayılan ayarlarla eğitin
2. **Değerlendirme**: WER/CER metriklerini kontrol edin
3. **Ayarlama**: Gerekirse hyperparameter'ları optimize edin
4. **Veri toplama**: Düşük doğruluk varsa daha fazla veri toplayın
5. **Tekrar eğitim**: İyileştirilmiş ayarlarla tekrar eğitin

## 🎓 Kaynaklar

- Wav2Vec2: https://huggingface.co/docs/transformers/model_doc/wav2vec2
- LoRA: https://github.com/microsoft/LoRA
- PEFT: https://github.com/huggingface/peft

