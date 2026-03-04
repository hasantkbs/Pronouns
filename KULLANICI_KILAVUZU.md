# Kullanıcı Kılavuzu — Kişiselleştirilmiş Konuşma Tanıma Sistemi

Bu kılavuz, sistemi ilk kez kullanacak kişiler için adım adım hazırlanmıştır.  
Teknik bilgi gerekmez; yalnızca komutları sırayla çalıştırmanız yeterlidir.

---

## İçindekiler

1. [Sistem Nedir?](#1-sistem-nedir)
2. [Kurulum](#2-kurulum)
3. [Adım 1 — Ses Kaydı Toplama](#3-adım-1--ses-kaydı-toplama)
   - [Otomatik Kayıt (Önerilen)](#otomatik-kayıt-önerilen)
   - [Manuel Kayıt](#manuel-kayıt)
4. [Adım 2 — Veriyi Eğitime Hazırlama](#4-adım-2--veriyi-eğitime-hazırlama)
5. [Adım 3 — Modeli Eğitme](#5-adım-3--modeli-eğitme)
6. [Adım 4 — Uygulamayı Kullanma](#6-adım-4--uygulamayı-kullanma)
7. [Sık Karşılaşılan Sorunlar](#7-sık-karşılaşılan-sorunlar)
8. [Ayarları Değiştirme](#8-ayarları-değiştirme)

---

## 1. Sistem Nedir?

Bu sistem iki temel işi yapar:

**Dinler ve anlar:**  
Konuşma güçlüğü çeken bir kişinin sesini mikrofondan dinler, ne söylediğini metne dönüştürür ve anlar.

**Konuşur:**  
Kişinin kendi kaydedilmiş kelimelerini birleştirerek cevap oluşturur. Yani sistem, kişinin kendi sesiyle konuşur.

Bunun için sistemin önce o kişinin sesini tanımayı öğrenmesi gerekir. Bu öğrenme süreci üç adımdan oluşur: ses kaydet → hazırla → eğit.

---

## 2. Kurulum

Sistemi ilk kez kurmak için terminalde şu komutu çalıştırın:

```bash
pip install -r requirements.txt
```

> **Not:** Python 3.9 veya üstü gereklidir. Kurulum birkaç dakika sürebilir.

---

## 3. Adım 1 — Ses Kaydı Toplama

Sistemin bir kişiyi tanıyabilmesi için o kişinin sesini kaydetmeniz gerekir.  
İki yöntem vardır:

---

### Otomatik Kayıt (Önerilen)

Bu yöntemde sisteme kelime listesini verirsiniz, o kalanını halleder.  
Kişi sadece gösterilen kelimeyi söyler; ENTER'a basmak, onaylamak gerekmez.  
Kayıt kalitesiz veya boş ise sistem otomatik olarak reddeder ve tekrar ister.

**Komut:**

```bash
python auto_collect.py <kullanici_adi> <kelime_dosyasi.txt>
```

**Örnek:**

```bash
python auto_collect.py Furkan datasets/words_set/temel_kelimeler.txt
```

**Ekranda şunu görürsünüz:**

```
============================================================
         MERHABA
------------------------------------------------------------
    Tekrar 1/10  |  Deneme 1/3
============================================================
  Dinleniyor |
```

Sistem kelimeyi büyük harfle gösterir ve sizi dinlemeye başlar.  
Kişi kelimeyi söylediğinde kayıt otomatik başlar ve biter.

**Ek seçenekler:**

| Seçenek | Açıklama | Örnek |
|---|---|---|
| `--reps 5` | Her kelime için 5 tekrar iste (varsayılan: 10) | `--reps 5` |
| `--resume` | Kaldığı yerden devam et (varsayılan açık) | Ekstra yazmaya gerek yok |
| `--no-resume` | Baştan başla, mevcut kayıtları sayma | `--no-resume` |
| `--threshold 0.015` | Ses algılama hassasiyeti (0-1 arası) | `--threshold 0.015` |
| `--asr-verify` | Her kaydın doğru kelime olduğunu kontrol et | `--asr-verify` |

**Kullanıcı yokken kayıt almak için:**

Kişiyi bilgisayarın başına oturtun, komutu çalıştırın ve odadan çıkabilirsiniz.  
Sistem sırayla her kelimeyi gösterir, kaydeder ve bir sonrakine geçer.  
Program bittiğinde tüm kayıtlar `data/users/<kullanici_adi>/words/` klasöründe olur.

> **Önemli:** Program kapatılsa bile kayıtlar silinmez. Bir dahaki çalıştırmada kaldığı yerden devam eder.

---

### Manuel Kayıt

Her kayıt için ENTER'a basmak isterseniz bu yöntemi kullanın.  
Kayıtları dinleme ve onaylama seçeneği vardır.

```bash
python collect_data.py
```

Menüden kayıt türünü seçin (kelime / cümle / harf), ardından kelime dosyasını ve kullanıcı adını girin.

---

## 4. Adım 2 — Veriyi Eğitime Hazırlama

Kayıtları aldıktan sonra, eğitim için hazırlama scripti çalıştırılmalıdır.  
Bu script, kayıtları eğitim ve doğrulama setlerine ayırır.

```bash
python prepare_training_data.py <kullanici_adi>
```

**Örnek:**

```bash
python prepare_training_data.py Furkan
```

Bu komut çalıştıktan sonra `data/users/Furkan/` klasöründe `train.csv` ve `eval.csv` dosyaları oluşur.

---

## 5. Adım 3 — Modeli Eğitme

Eğitim komutu, kişinin sesini tanımayı öğrenecek modeli oluşturur.

```bash
python train_adapter.py <kullanici_adi>
```

**Örnek:**

```bash
python train_adapter.py Furkan
```

Eğitim sırasında ekranda şunları göreceksiniz:

```
Epoch 1/30 | Loss: 2.4521 | LR: 1.98e-05
   Validation: Loss: 2.1234 | WER: 45.23% | CER: 12.10%
   En iyi model kaydedildi! (WER: 45.23%)
```

Eğitim GPU varsa birkaç saat, CPU'da daha uzun sürebilir.  
Eğitim bittiğinde model `data/models/personalized_models/Furkan/` klasörüne kaydedilir.

> **Not:** Eğitim yarıda kesilse de en iyi model otomatik kaydedilir.

---

## 6. Adım 4 — Uygulamayı Kullanma

Model eğitildikten sonra gerçek zamanlı konuşma tanıma başlatılabilir.

```bash
python app.py
```

Program başladığında sizden kullanıcı adı ister:

```
Lutfen kullanici kimliginizi girin (orn: hasan): Furkan
```

Kullanıcı adını girip ENTER'a basın. Sistem kişiselleştirilmiş modeli yükler ve başlar.

```
=========================================
  Konusma Bozuklugu Ses Tanima Sistemi
=========================================
Hos geldin, Furkan!
Cikis icin 'cik' veya 'exit' deyin.

Konusmak icin ENTER'a basin ve konusun...
```

ENTER'a bastıktan sonra mikrofona konuşun. Sistem:
1. Sizi dinler
2. Ne söylediğinizi metne dönüştürür ve güven skorunu gösterir
3. Anladığı niyete göre cevap verir
4. Cevabı kişinin kendi kaydedilmiş sesiyle oynatır

**Tanınan komutlar (örnekler):**

| Ne söylenirse | Ne yapılır |
|---|---|
| "merhaba" | Selamlama yanıtı |
| "nasılsın" | Hal hatır sorar |
| "saat kaç" | Saati söyler |
| "not al ... diye" | Not kaydeder |
| "dosya listele" | Dosyaları gösterir |
| "çık" / "exit" | Programı kapatır |

---

## 7. Sık Karşılaşılan Sorunlar

### "Ses algılanamadı" mesajı çok sık çıkıyor

Ses eşiği çok yüksek olabilir. `config.py` dosyasında şu satırı düşürün:

```python
AUTO_SOUND_THRESHOLD = 0.008   # Varsayılan: 0.012
```

Veya komutu çalıştırırken doğrudan belirtin:

```bash
python auto_collect.py Furkan datasets/words_set/temel_kelimeler.txt --threshold 0.008
```

---

### "RED — Ses çok sessiz" mesajı çıkıyor

Mikrofon kişiye çok uzak olabilir. Mikrofonu 20-30 cm'ye yaklaştırın.  
Ayrıca `config.py` dosyasında minimum ses seviyesini düşürebilirsiniz:

```python
MIN_RMS_LEVEL = 200   # Varsayılan: 300
```

---

### Model yüklenmiyor

İnternet bağlantınızı kontrol edin. Temel model ilk açılışta internetten indirilir (~1.2 GB).  
İndirme tamamlandıktan sonra tekrar çalıştırın.

---

### Eğitim çok yavaş

Bilgisayarda NVIDIA GPU yoksa eğitim CPU üzerinde çalışır ve 10-20 saat sürebilir.  
GPU olan bir sunucuya aktarmak için `LINUX_SERVER_SETUP.md` dosyasına bakın.

---

### Tanıma doğruluğu düşük

- Daha fazla kayıt toplayın (her kelime için en az 10 tekrar önerilir).
- `train_adapter.py` komutunu tekrar çalıştırın.
- Kayıt ortamının sessiz olduğundan emin olun.

---

## 8. Ayarları Değiştirme

Tüm ayarlar `config.py` dosyasında bulunur. En sık ihtiyaç duyulan ayarlar:

| Ayar | Varsayılan | Açıklama |
|---|---|---|
| `IDEAL_REPETITIONS` | 10 | Her kelime için hedef kayıt sayısı |
| `QUALITY_THRESHOLD` | 40 | Minimum kabul edilebilir kalite skoru (0-100) |
| `AUTO_WORD_TIMEOUT_SEC` | 15 | Kelime başına bekleme süresi (saniye) |
| `AUTO_MAX_RETRIES` | 3 | Başarısız denemede otomatik tekrar sayısı |
| `AUTO_SILENCE_LIMIT_SEC` | 1.0 | Sessizlik algılandıktan sonra kaydı bitirme süresi |
| `NUM_FINETUNE_EPOCHS` | 30 | Eğitim tekrar sayısı (artırınca doğruluk artar) |
| `SYNTHESIS_PAUSE_MS` | 150 | Sentezlenen kelimelerin arasındaki duraklama (ms) |

Değişiklik yapmak için `config.py` dosyasını bir metin editörüyle açın ve ilgili satırı düzenleyin.
