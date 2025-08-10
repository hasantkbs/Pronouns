# Konusma Anlama ve Tanıma Sistemi 

Bu proje, `Wav2Vec2` modeli kullanarak ses dosyalarından kelime embedding'leri üreten, bu embedding'ler ile bir sınıflandırıcı eğiten ve sonuçta ses verisinden kelime tahmini yapan bir sistemdir.

## Proje Akışı

1.  **Veri Hazırlığı:** `users/user_001/` klasörüne `kelime_1.wav`,`kelime_2.wav` formatında ses dosyaları eklenir.
2.  **Kalibrasyon (`01_calibrate.py`):** Ses dosyalarını tarayarak hangi ses dosyasının hangi kelimeye ait olduğunu belirten bir `meta.json` dosyası oluşturur.
3.  **Embedding Üretimi (`02_generate_embeddings.py`):** Her ses dosyası için önceden eğitilmiş `Wav2Vec2` modelini kullanarak bir embedding (özellik vektörü) çıkarır ve bunları `embeddings/` klasörüne kaydeder.
4.  **Model Eğitimi (`train_classifier.py`):** Üretilen embedding'leri ve etiketleri kullanarak kelimeleri sınıflandıracak basit bir sinir ağı modelini eğitir. Eğitilmiş modeli (`classifier_model.pth`) ve etiket listesini (`label_encoder.json`) kaydeder.
5.  **Tanıma (`03_realtime_recognition.py`):** Eğitilen modeli ve yeni ses verisini kullanarak kelime tahmini yapar.

## Kurulum

Proje bağımlılıklarını yüklemek için aşağıdaki komutu çalıştırın:

pip install -r requirements.txt

## Kullanım

Scriptleri aşağıdaki sırayla çalıştırın:

python 01_calibrate.py
python 02_generate_embeddings.py
python train_classifier.py
python 03_realtime_recognition.py

# Konuşma Anlama ve Tanıma Sistemi

Bu proje, `Wav2Vec2` modeli kullanarak ses dosyalarından kelime embedding'leri üreten, bu embedding'ler ile bir sınıflandırıcı eğiten ve sonuçta ses verisinden kelime tahmini yapan bir sistemdir. Proje, modüler bir yapıda tasarlanmış olup, veri toplama, analiz, eğitim ve test süreçleri için yardımcı araçlar içerir.

## Dosyalar ve Görevleri

Proje, belirli bir akışa göre çalışan bir dizi script'ten oluşur:

### Veri Yönetimi Araçları

- **`collect_new_data.py`**: **(Veri Toplama Yardımcısı)** Yeni ses verisi toplama sürecini otomatikleştiren bir araçtır. Kullanıcıdan bir kelime alır, ses kaydı yapar ve dosyayı otomatik olarak doğru formatta ( `kelime_X.wav`) isimlendirerek kaydeder.

- **`analyze_data_distribution.py`**: **(Veri Analiz Aracı)** `meta.json` dosyasını okuyarak veri setindeki her bir kelimenin (etiketin) kaç adet ses örneğine sahip olduğunu analiz eder ve raporlar. Veri setindeki dengesizlikleri ve zayıf noktaları tespit etmek için kullanılır.

- **`00_cleanup_augmented_data.py`**: Veri artırma (augmentation) ile oluşturulmuş sentetik dosyaları (`_aug` içerenler) temizlemek için kullanılır.

### Ana İşlem Akışı Script'leri

- **`01_calibrate.py`**: `users/user_001/` klasöründeki ses dosyalarını tarayarak hangi ses dosyasının hangi kelimeye ait olduğunu belirten bir `meta.json` dosyası oluşturur.

- **`02_generate_embeddings.py`**: Her ses dosyası için `Wav2Vec2` modelini kullanarak bir embedding (özellik vektörü) çıkarır ve bunları `embeddings/` klasörüne kaydeder.

- **`train_classifier.py`**: Üretilen embedding'leri ve etiketleri kullanarak kelimeleri sınıflandıracak sinir ağı modelini eğitir. Eğitim sonunda en iyi modeli (`classifier_model.pth`) ve etiket listesini ( `label_encoder.json`) kaydeder.

- **`03_realtime_recognition.py`**: Eğitilen modeli ve test ses dosyalarını kullanarak kelime tahmini yapar ve modelin performansını değerlendirir.

### Yardımcı Modüller ve Klasörler

- **`config.py`**: Model adı, öğrenme oranı, dosya yolları gibi tüm proje ayarlarının ve hiperparametrelerin bulunduğu merkezi yapılandırma dosyası.

- **`utils.py`**: Model yükleme, embedding üretme gibi tekrar eden ve projenin farklı yerlerinde kullanılan yardımcı fonksiyonları ve sınıfları içerir.

- **`tests/`**: Projenin modüllerinin doğru çalıştığını kontrol eden `pytest` testlerini içerir.

## Önerilen Kullanım Akışı

1.  **Veri Toplama:** `python collect_new_data.py` script'ini kullanarak her kelime için yeterli sayıda (tavsiye edilen en az 15-20) ses kaydı yapın.

2.  **Veri Analizi:** `python analyze_data_distribution.py` script'i ile veri setinizin mevcut durumunu kontrol edin.

3.  **Kalibrasyon:** `python 01_calibrate.py` ile `meta.json` dosyasını oluşturun/güncelleyin.

4.  **Embedding Üretimi:** `python 02_generate_embeddings.py` ile tüm ses dosyaları için embedding'leri üretin. (Bu işlem uzun sürebilir).

5.  **Model Eğitimi:** `python train_classifier.py` ile sınıflandırıcı modelinizi eğitin ve test doğruluğunu gözlemleyin.

6.  **Test (Opsiyonel):** `pytest` komutunu çalıştırarak projenin temel fonksiyonlarının doğru çalıştığından emin olun.

