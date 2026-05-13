# Pronouns AI - Session Progress (12 May 2026)

Bu doküman, bu oturumda gerçekleştirilen teknik geliştirmeleri, model eğitimi durumunu ve sistem optimizasyonlarını özetler.

## 1. Veri Hazırlığı ve Ortam Kurulumu
- **FurkanV1 Veri Seti:** 6890 adet ses kaydı işlendi. `train.csv` (5502) ve `eval.csv` (1376) olarak ayrıldı.
- **Sistem Bağımlılıkları:** `pyaudio` için kritik olan `libportaudio2` kütüphanesi sunucuya kuruldu.
- **Dizin Yapısı:** `data/models/personalized_models/FurkanV1` altında eğitim çıktıları için klasörler oluşturuldu.

## 2. Eğitim Hattı (Training Pipeline) Optimizasyonları
- **Casting Hatası Giderildi:** `datasets` kütüphanesindeki `large_string` -> `Audio` dönüşüm hatası, açık string cast işlemi ile (`Value("string")`) `train_adapter.py` içinde çözüldü.
- **Multiprocessing Kararlılığı:** Preprocessing sırasında yaşanan `OSError: got end of file` hatası, `num_proc` yönetimi iyileştirilerek ve tek çekirdekli ana süreç kullanımı opsiyonu eklenerek giderildi.
- **Türkçe Normalizasyon:** `src/utils/utils.py` dosyasına `normalize_turkish_text` fonksiyonu eklendi. Büyük-küçük harf (İ-i, I-ı), noktalama işaretleri ve çift boşluk temizliği standart hale getirildi.

## 3. Model ve ASR İyileştirmeleri
- **LoRA Kapasitesi Artırıldı:** `config.py` içindeki `ADAPTER_REDUCTION_FACTOR` **64'ten 128'e** çıkarıldı. Modelin konuşma bozukluklarına ait özgün fonetik yapıları öğrenme kapasitesi artırıldı.
- **Hotwords Entegrasyonu:** `src/core/asr.py` içindeki Beam Search decoder'a kullanıcının kendi kelime haznesini (User Vocabulary) "hotwords" olarak önceliklendirme özelliği eklendi.
- **Fuzzy Matching Geliştirildi:** ASR çıktılarının kullanıcının kayıtlı kelimeleriyle karşılaştırılması ve Levenshtein mesafesine göre otomatik düzeltilmesi süreci `normalize_turkish_text` ile daha tutarlı hale getirildi.
- **API Parametreleri:** `api.py` içindeki `get_asr` fonksiyonuna `user_id` parametresi eklenerek, ASR sisteminin her kullanıcıya özel kelime haznesini yüklemesi sağlandı.

## 4. Güncel Durum (Deployment)
- **Backend API:** Port **8001** üzerinden aktif (Flutter mobil uygulamasındaki varsayılan portla senkronize edildi).
- **Eğitim (FurkanV1):** Wav2Vec2 adapter eğitimi GPU (L40S) üzerinde arka planda devam ediyor.
- **Log Takibi:** 
  - API: `api.log`
  - Eğitim: `training_furkan.log`

## 5. Gelecek Adımlar (TODO)
- **Model Değerlendirme:** Eğitim tamamlandığında (100 Epoch veya Early Stopping), WER oranının %10'un altına düşüp düşmediği test edilecek.
- **Mobil App Senkronizasyonu:** Model tamamlandığında API tarafında `asr_systems` cache'i yenilenerek yeni modelin mobil cihazlarda test edilmesi sağlanacak.
- **Dil Modeli Güncellemesi:** Mevcut 3.1MB'lık `lm.arpa` modelinin, Furkan'ın günlük kullandığı cümlelerle genişletilmesi WER oranını daha da düşürecektir.
