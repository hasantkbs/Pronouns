# Eğitim Sorunları ve Düzeltmeler

## 🔴 Tespit Edilen Sorunlar

### 1. Yüksek WER/CER (99.76% / 82.87%)
- Model hiç öğrenmemiş
- Label encoding hatası
- Veri formatı uyumsuzluğu

### 2. Negatif Loss (-0.7099)
- Geçersiz loss değeri
- Label encoding sorunu
- CTC loss hesaplama hatası

## ✅ Yapılan Düzeltmeler

### 1. Label Encoding Düzeltmesi

**Önceki (Yanlış):**
```python
# Batch olarak tokenize ediliyordu
labels = processor.tokenizer(
    valid_transcripts, 
    return_tensors="pt", 
    padding=True
).input_ids
```

**Yeni (Doğru):**
```python
# Her örnek için ayrı ayrı tokenize ediliyor
for transcript in valid_transcripts:
    label_ids = processor.tokenizer(transcript).input_ids
    if isinstance(label_ids, torch.Tensor):
        label_ids = label_ids.tolist()
    if isinstance(label_ids[0], list):
        label_ids = label_ids[0]
    label_ids_list.append(label_ids)
```

### 2. Input Values Formatı Düzeltmesi

**Önceki (Yanlış):**
```python
# Batch olarak işleniyordu
inputs = processor(audio_arrays, ...)
input_values = inputs.input_values  # Tensor
```

**Yeni (Doğru):**
```python
# Her örnek için ayrı ayrı işleniyor
for audio in audio_arrays:
    inputs = processor(audio, padding=False, ...)
    input_values_list.append(inputs.input_values[0].tolist())
```

### 3. Loss Kontrolü Eklendi

```python
# Negatif veya invalid loss kontrolü
if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
    print(f"⚠️  Geçersiz loss: {loss.item()}, batch atlanıyor.")
    continue
```

### 4. Validation İyileştirmeleri

- Loss kontrolü eklendi
- Boş tahminler filtreleniyor
- Daha iyi hata yönetimi
- Debug bilgileri

## 🚀 Yeniden Eğitim

Düzeltmelerden sonra yeniden eğitim yapın:

```bash
# Eski modeli temizle (opsiyonel)
rm -rf data/models/personalized_models/Furkan/checkpoints

# Yeniden eğitim
python3 train_adapter.py Furkan
```

## 📊 Beklenen İyileştirmeler

Düzeltmelerden sonra:
- ✅ Loss pozitif ve azalan olmalı
- ✅ WER: 99.76% → <30% (ilk epoch'ta)
- ✅ CER: 82.87% → <15% (ilk epoch'ta)
- ✅ Model öğrenmeye başlamalı

## 🔍 Kontrol Listesi

Eğitim sırasında kontrol edin:

1. **Loss değerleri**:
   - Pozitif olmalı
   - Azalan trend göstermeli
   - 0.5-5.0 arası normal

2. **Validation metrikleri**:
   - WER: Her epoch'ta azalmalı
   - CER: Her epoch'ta azalmalı
   - Loss: Training loss'a yakın olmalı

3. **Veri kalitesi**:
   - Ses dosyaları yükleniyor mu?
   - Transkriptler doğru mu?
   - Label'lar doğru encode ediliyor mu?

## 🐛 Hala Sorun Varsa

Eğer hala yüksek WER/CER varsa:

1. **Veri kontrolü**:
   ```bash
   # İlk birkaç örneği kontrol et
   head -5 data/users/Furkan/train.csv
   ```

2. **Model kontrolü**:
   ```bash
   # Model yükleniyor mu?
   python3 -c "from transformers import Wav2Vec2ForCTC; print('OK')"
   ```

3. **Debug modu**:
   - İlk batch'i yazdır
   - Label'ları kontrol et
   - Input shape'leri kontrol et

## 📝 Notlar

- Label encoding çok kritik - her örnek ayrı işlenmeli
- CTC loss için -100 padding token'ları önemli
- Batch processing yerine individual processing daha güvenilir

