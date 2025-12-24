# Konuşma Bozukluğu ASR Sistemi - Kullanım Kılavuzu

## 🎯 Proje Hakkında

Bu proje, konuşma bozukluğu olan bireyler (örneğin Furkan) için özelleştirilmiş bir Otomatik Konuşma Tanıma (ASR) sistemidir. Wav2Vec2 tabanlı model kullanarak, kullanıcının ses kayıtlarıyla model eğitilir ve gerçek zamanlı konuşma tanıma yapılır.

## 📋 Sistem Gereksinimleri

- Python 3.9+
- CUDA destekli GPU (önerilir, CPU da çalışır)
- FFmpeg (ses işleme için)

## 🚀 Hızlı Başlangıç

### 1. Veri Hazırlama

Furkan'ın ses kayıtları zaten `data/users/Furkan/` klasöründe bulunuyor. Eğitim için verileri hazırlamak için:

```bash
python prepare_training_data.py Furkan
```

Bu komut:
- `metadata_words.csv` dosyasını okur
- Verileri eğitim (%80) ve değerlendirme (%20) setlerine ayırır
- `train.csv` ve `eval.csv` dosyalarını oluşturur

### 2. Model Eğitimi

Furkan için kişiselleştirilmiş model eğitmek için:

```bash
python train_adapter.py Furkan
```

Bu komut:
- Wav2Vec2 tabanlı temel modeli yükler
- LoRA adapter ile kişiselleştirme yapar
- Eğitilmiş modeli `data/models/personalized_models/Furkan/` klasörüne kaydeder

**Eğitim Parametreleri** (`config.py` dosyasında ayarlanabilir):
- `NUM_FINETUNE_EPOCHS`: 15 (epoch sayısı)
- `FINETUNE_BATCH_SIZE`: 2 (batch boyutu)
- `FINETUNE_LEARNING_RATE`: 1e-4 (öğrenme oranı)
- `ADAPTER_REDUCTION_FACTOR`: 32 (LoRA adapter boyutu)

### 3. Model Değerlendirme

Eğitilmiş modelin performansını değerlendirmek için:

```bash
python evaluate_model.py Furkan
```

Opsiyonel: Sadece ilk 100 örneği değerlendirmek için:
```bash
python evaluate_model.py Furkan --max_samples 100
```

Bu komut:
- WER (Word Error Rate) ve CER (Character Error Rate) metriklerini hesaplar
- Örnek tahminler gösterir
- İyileştirme önerileri sunar

### 4. Gerçek Zamanlı Kullanım

Eğitilmiş model ile gerçek zamanlı konuşma tanıma için:

```bash
python app.py
```

Sistem sizden kullanıcı kimliği ister. "Furkan" yazın ve ENTER'a basın. Sistem otomatik olarak:
- Kişiselleştirilmiş modeli yükler (varsa)
- Mikrofonu dinlemeye başlar
- Konuşmanızı metne dönüştürür
- Ekrana yazdırır

**Çıkmak için**: "çık" veya "exit" deyin.

## 📁 Proje Yapısı

```
Pronouns/
├── app.py                          # Ana uygulama
├── config.py                        # Yapılandırma dosyası
├── prepare_training_data.py         # Veri hazırlama scripti
├── train_adapter.py                 # Model eğitim scripti
├── evaluate_model.py                # Model değerlendirme scripti
├── src/
│   ├── core/
│   │   ├── asr.py                   # ASR sistemi (Wav2Vec2)
│   │   ├── nlu.py                   # Doğal dil anlama
│   │   └── actions.py               # Eylem yürütme
│   └── utils/
│       └── utils.py                 # Yardımcı fonksiyonlar
├── data/
│   ├── users/
│   │   └── Furkan/
│   │       ├── metadata_words.csv   # Ses dosyaları metadata
│   │       ├── train.csv            # Eğitim seti
│   │       ├── eval.csv             # Değerlendirme seti
│   │       └── words/               # Ses dosyaları
│   └── models/
│       └── personalized_models/
│           └── Furkan/              # Eğitilmiş model
└── requirements.txt                 # Python bağımlılıkları
```

## 🔧 Yapılan İyileştirmeler

### 1. Model Tutarlılığı
- ✅ Tüm sistem Wav2Vec2 tabanlı hale getirildi
- ✅ `asr.py` Whisper'dan Wav2Vec2'ye güncellendi
- ✅ Model yükleme mantığı iyileştirildi

### 2. Veri İşleme
- ✅ `prepare_training_data.py` iyileştirildi (hata kontrolü, dosya doğrulama)
- ✅ Eğitim scripti train.csv ve eval.csv dosyalarını otomatik kullanıyor
- ✅ Boş ve geçersiz veriler otomatik filtreleniyor

### 3. Eğitim İyileştirmeleri
- ✅ Daha iyi hata yönetimi ve loglama
- ✅ Progress bar ve epoch bazlı loss gösterimi
- ✅ Gradient accumulation desteği
- ✅ Veri ön işleme optimizasyonu

### 4. Değerlendirme
- ✅ Daha detaylı metrikler (WER, CER)
- ✅ Örnek tahminler gösterimi
- ✅ İyileştirme önerileri

### 5. Kullanıcı Deneyimi
- ✅ Daha açıklayıcı hata mesajları
- ✅ İlerleme göstergeleri
- ✅ Otomatik model algılama

## ⚙️ Yapılandırma

`config.py` dosyasından aşağıdaki ayarları yapabilirsiniz:

```python
# Model ayarları
MODEL_NAME = "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish"
ORNEKLEME_ORANI = 16000

# Eğitim ayarları
NUM_FINETUNE_EPOCHS = 15
FINETUNE_BATCH_SIZE = 2
FINETUNE_LEARNING_RATE = 1e-4
ADAPTER_REDUCTION_FACTOR = 32

# Ses kayıt ayarları
KAYIT_SURESI_SN = 5
SES_ESIK_DEGERI = 0.01
```

## 🐛 Sorun Giderme

### Model yüklenemiyor
- İnternet bağlantınızı kontrol edin (ilk yüklemede model indirilir)
- `data/models/personalized_models/Furkan/` klasörünün var olduğundan emin olun

### Eğitim sırasında hata
- Ses dosyalarının `data/users/Furkan/words/` klasöründe olduğundan emin olun
- `metadata_words.csv` dosyasının doğru formatta olduğunu kontrol edin
- Önce `prepare_training_data.py` çalıştırın

### Düşük doğruluk
- Daha fazla eğitim verisi toplayın
- Epoch sayısını artırın (`config.py`'de `NUM_FINETUNE_EPOCHS`)
- Öğrenme oranını ayarlayın (`FINETUNE_LEARNING_RATE`)

## 📊 Performans Metrikleri

İyi bir model için hedef metrikler:
- **WER < 0.15** (Word Error Rate %15'ten az)
- **CER < 0.05** (Character Error Rate %5'ten az)

## 📝 Notlar

- Model eğitimi GPU ile çok daha hızlıdır
- Küçük veri setleri için LoRA adapter kullanımı önerilir
- Eğitim sırasında sistem kaynaklarını kontrol edin
- Düzenli olarak model performansını değerlendirin

## 🤝 Destek

Sorun yaşarsanız:
1. Hata mesajlarını kontrol edin
2. Log dosyalarını inceleyin
3. `evaluate_model.py` ile model performansını kontrol edin

