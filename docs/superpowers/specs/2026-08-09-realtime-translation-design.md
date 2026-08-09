# Furkanca Sayfası: Sonuç Sadeleştirme + Gerçek Zamanlı Dinleme

**Tarih:** 2026-08-09
**Kapsam:** `lib/main.dart` — yalnızca `_FurkancaPage`/`_FurkancaPageState`. `_KayitPage`, backend (`api.py`) **etkilenmiyor**.

## Sorun

1. Çeviri sonrası ekranda dört ayrı kutu gösteriliyor: Duyulan (ham ASR), Niyet (NLU), Düzeltilmiş (nihai metin), Eksik Kelimeler. Kullanıcı yalnızca nihai çeviriyi görmek istiyor.
2. Mevcut akış manuel: kullanıcı bir süre (slider ile 2-12 sn) seçip "Konuşmaya Başla"ya basıyor, sabit süre kayıt alınıyor, sonra gönderiliyor. İstenen: kullanıcı elini kullanmadan konuşabilsin — sistem konuşmayı kendisi algılayıp otomatik kaydetsin ve art arda birden fazla cümleyi arka arkaya çevirsin ("gerçek zamanlı" dinleme).

## Kapsam Dışı (bilinçli karar)

Gerçek streaming ASR (konuşurken kelime kelime ekrana dökülen canlı transkripsiyon) bu spec'in kapsamında **değil** — bu, sunucuda WebSocket + streaming-uyumlu bir ASR hattı gerektirir ve projenin "önce mobil, sonra sunucu AI modeli" planının önüne geçer. Bunun yerine: istemci tarafında konuşma algılama (VAD) ile otomatik segment kaydı, mevcut dosya-tabanlı `/translate` endpoint'ine değişmeden gönderilir. Sunucu tarafında hiçbir değişiklik yok.

## Tasarım

### 1. Sonuç Görünümü

- `_ResultTile` çağrıları: `Duyulan` (`_recognized`), `Niyet` (`_intent`), `Eksik Kelimeler` (`_missing`) kaldırılır.
- Sunucudan gelen `recognized_text`/`intent`/`missing_words` alanları hâlâ parse edilebilir (API sözleşmesi değişmiyor) ama state'e yazılmaz/gösterilmez — sadece `response_text` (`_corrected`) kullanılır.
- Ses geri oynatma (`_player.play(audioUrl)`) değişmeden kalır.
- Tek sonuç yerine **kayan geçmiş listesi**: her tamamlanan utterance, `List<_TranslationEntry>` gibi bir listeye eklenir (metin + zaman damgası + varsa hata durumu) ve `ListView` ile en yeni öğe görünür şekilde gösterilir.

### 2. Kontrol Modeli

- Süre `Slider`'ı ve "Konuşmaya Başla" butonu kaldırılır.
- Yerine tek bir toggle: **"Dinlemeyi Başlat" / "Dinlemeyi Durdur"**.
- Sayfa durumları: `idle` (dinlemiyor) → `listening` (mikrofon açık, konuşma bekleniyor) → `capturing` (konuşma algılandı, segment kaydediliyor) → `uploading` (segment `/translate`'e gönderiliyor) → `speaking` (TTS yanıtı çalıyor) → `listening` (döngü devam eder, kullanıcı durdurana kadar).
- Her durum için kısa bir durum metni gösterilir (`Dinleniyor...`, `Kaydediliyor...`, `İşleniyor...`, `Yanıt seslendiriliyor...`).

### 3. Konuşma Algılama (VAD) — istemci tarafı, sunucu değişmeden

`record` paketi (`^6.2.0`, proje zaten kullanıyor) `AudioRecorder.startStream(RecordConfig(encoder: AudioEncoder.pcm16bits, numChannels: 1, sampleRate: 16000))` ile ham `Stream<Uint8List>` PCM16LE verisi sağlıyor; ayrıca `pause()`/`resume()`/`stop()` destekliyor (paket kaynağında doğrulandı: `record-6.2.0/lib/src/record.dart`).

- "Dinlemeyi Başlat"a basınca **tek bir** `startStream` çağrısı yapılır ve kullanıcı durdurana kadar açık kalır (her utterance için ayrı `start`/`stop` **yok** — sürekli tek akış).
- Gelen her `Uint8List` chunk'ı Int16 örneklere çözülüp normalize edilir (`sample / 32768.0`) ve RMS hesaplanır — Python tarafındaki `config.py`'deki `AUTO_SOUND_THRESHOLD` (varsayılan `0.012`) ile **aynı birim ve varsayılan değer** kullanılır (tutarlılık, aynı "ne kadar ses = konuşma" tanımı).
- **Ön-tampon (pre-roll):** Son ~300ms'lik chunk'lar sürekli bir ring buffer'da tutulur; RMS eşiği aşıldığında (`onset`) bu tampon da segment'in başına eklenir — ilk hece kırpılmaz.
- **Konuşma sonu (offset):** RMS eşiğin altında kesintisiz `1.0` saniye kalırsa (Python'daki `AUTO_SILENCE_LIMIT_SEC` ile aynı değer) utterance biter.
- **Güvenlik sınırı:** Bir segment `12` saniyeyi geçerse (eski slider'ın max değeriyle tutarlı) zorla kesilir.
- **Minimum uzunluk:** `0.3` saniyeden kısa patlamalar (ör. öksürük/tık sesi) utterance sayılmaz, segment atılır ve dinlemeye devam edilir.
- Segment tamamlanınca biriken PCM verisi bir WAV dosyasına yazılır (44 byte'lık standart WAV header elle eklenir — `pcm16bits`/16kHz/mono parametreleriyle), mevcut `/translate` multipart isteğine **değişmeden** aynı şekilde gönderilir.
- **Geri besleme (feedback) önleme:** Segment gönderilip TTS yanıtı çalınırken (`uploading`/`speaking` durumları) akış `recorder.pause()` ile durdurulur; yanıt bitince `recorder.resume()` ile dinlemeye devam edilir — yoksa hoparlörden çıkan kendi sesini mikrofon algılayıp yeni bir "konuşma" sanabilir.

### 4. Hata Yönetimi

- Mikrofon izni reddedilirse: dinleme başlamaz, kısa bir hata mesajı gösterilir, buton `idle` durumuna döner.
- Bir segmentin `/translate` isteği başarısız olursa (ağ/sunucu hatası): geçmiş listesine kısa bir hata satırı eklenir, **dinleme durmaz**, döngü otomatik olarak bir sonraki konuşmayı beklemeye devam eder.
- Sayfa `dispose` edilirken: PCM stream aboneliği iptal edilir, `recorder.stop()` + `dispose()` çağrılır, bekleyen timer'lar/subscription'lar temizlenir — arka planda mikrofon açık kalmaz.

## Kapsam Dışı / Etkilenmeyenler

- `_KayitPage` (Kayıt sayfası) ve oradaki sabit süreli kayıt akışı — önceki turda düzeltilen "geri sayımla eş zamanlı kayıt" mantığı aynen kalıyor.
- Backend (`api.py`, `/translate` endpoint sözleşmesi) — hiçbir değişiklik yok.
- Gerçek streaming ASR / kelime kelime canlı transkripsiyon — bilinçli olarak kapsam dışı (yukarıda açıklandı).

## Test Planı

- RMS/VAD segmentasyon mantığı (onset/offset/pre-roll/max-süre/min-uzunluk) saf Dart fonksiyonu olarak ayrı bir dosyaya çıkarılır (ör. `lib/vad.dart`) ve **unit test** ile doğrulanır: sahte PCM örnek dizileriyle (sessiz → konuşma → sessiz, kısa patlama, çok uzun konuşma) doğru segment sınırlarının üretildiği kontrol edilir.
- Platform bağımlı kısımlar (`record` paketinin gerçek mikrofon akışı) otomatik test kapsamı dışında — önceki turlardaki gibi native plugin mocking'in getirdiği kırılganlık/orantısızlık nedeniyle. Cihazda manuel doğrulama gerekir: art arda birkaç cümle söyleyip her birinin ayrı ayrı, doğru sınırlarla çevrildiğinin ve dinlemenin kesintisiz sürdüğünün kontrolü.
- Sonuç listesi UI'ı için basit bir widget testi: sahte bir sonuç listesi state'e enjekte edilip `ListView`'de doğru sırada/sayıda göründüğü doğrulanabilir (VAD/mikrofon'a bağımlı değil).
