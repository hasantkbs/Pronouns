# Konuşmacı Doğrulamasıyla Sesle Tetikleme — Tasarım

**Tarih:** 2026-08-10
**Durum:** Onaylandı

## Amaç

Furkanca ekranında zaten çalışan VAD (Voice Activity Detection) tabanlı dinleme
akışına, yakalanan ses segmentinin gerçekten kullanıcıya (Furkan) ait olup
olmadığını kontrol eden bir "konuşmacı doğrulama" (speaker verification)
katmanı eklemek. Böylece:

- Arka plan gürültüsü veya başka bir kişinin konuşması sunucuya boşu boşuna
  yüklenip ağır ASR/NLU/sentez pipeline'ını çalıştırmaz.
- Uygulama pratikte "sadece Furkan'ın sesiyle tetiklenir" davranışı kazanır.

Mevcut VAD (`lib/vad.dart` → `VadSegmenter`) genlik (RMS) tabanlıdır ve kimin
konuştuğuna bakmaz; bu tasarım onun üstüne, sesin *kime ait olduğunu*
doğrulayan bir filtre ekler. VAD'ın kendisi değişmez.

## Kapsam Dışı

- Uygulamanın kapalı/arka plandayken sesle açılması (OS seviyesinde wake-word,
  foreground service vb.) — bu tasarımın konusu değil.
- "Başlat" butonunu kaldırıp dinlemeyi tamamen otomatik başlatmak — mevcut
  manuel başlatma akışı korunur, sadece VAD'ın yakaladığı segmentler
  sunucuda filtrelenir.

## Mimari

### Sunucu tarafı (Python)

Yeni servis: `src/services/speaker_verification_service.py` →
`SpeakerVerificationService`

- **Model:** SpeechBrain ECAPA-TDNN pretrained encoder
  (`speechbrain/spkrec-ecapa-voxceleb`). Tüm kullanıcılar arasında paylaşılan
  tek bir global singleton olarak yüklenir (model kullanıcıya özel değildir,
  sadece embedding çıkarır); `ASRSystem`'in aksine kullanıcı başına
  yeniden yüklenmez.
- **Referans profil (enrollment):** Kullanıcının `data/users/{user_id}/words/`
  altındaki mevcut .wav kayıtlarından (en fazla
  `config.SPEAKER_ENROLL_MAX_SAMPLES` adet) embedding çıkarılır, ortalaması
  (centroid) alınıp normalize edilir. Sonuç `data/users/{user_id}/voice_profile.pt`
  dosyasına yazılır (embedding vektörü + oluşturulma zamanı + kaç kayıttan
  üretildiği bilgisiyle birlikte) ve bellekte de önbelleklenir.
- **Doğrulama:** `verify(user_id, audio_path) -> (is_match: bool, score: float)`
  gelen segment için embedding çıkarır, kayıtlı centroid ile kosinüs
  benzerliği hesaplar, `config.SPEAKER_VERIFY_THRESHOLD` ile karşılaştırır.
- **Fail-open kuralı:** Referans profil hiç oluşturulmamışsa (yetersiz/hiç
  kayıt yoksa), enrollment sırasında hata oluşursa veya verify sırasında
  beklenmeyen bir hata olursa → doğrulama atlanır, `is_match=True` kabul
  edilir ve normal akış devam eder. Bu özellik hiçbir koşulda Furkan'ın
  uygulamayı kullanmasını engellememelidir.

### API (`api.py`)

- `get_speaker_service()`: modül seviyesinde bir kez oluşturulan global
  singleton (mevcut `get_asr()` / `asr_systems` desenine benzer, ama
  kullanıcı başına değil tek örnek).
- `/translate` içine, ASR çağrısından hemen önce doğrulama eklenir:
  - `SPEAKER_VERIFY_ENABLED` kapalıysa adım tamamen atlanır.
  - Eşleşmezse: `{"rejected": true, "reason": "speaker_mismatch"}` döner
    (HTTP 200 — bir hata değil, filtrelenmiş bir sonuçtur). ASR/NLU/sentez
    çalışmaz.
  - Eşleşirse veya profil yoksa (fail-open): mevcut akış (ASR → NLU →
    sentez) olduğu gibi devam eder.
- Yeni `POST /voice-profile/rebuild?user_id=...`: `enroll(user_id)` çağırır,
  `{"sample_count": N, "created_at": ...}` döner.
- Yeni `GET /voice-profile/status?user_id=...`: profilin var olup olmadığını,
  kaç örnekten oluşturulduğunu ve ne zaman oluşturulduğunu döner.

### Config (`config.py`)

```python
SPEAKER_VERIFY_ENABLED = True
SPEAKER_VERIFY_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_VERIFY_THRESHOLD = 0.72   # başlangıç değeri; gerçek verilerle kalibre edilecek
SPEAKER_ENROLL_MAX_SAMPLES = 40
```

`SPEAKER_VERIFY_ENABLED = False` yapılarak özellik tamamen kapatılabilir
(mevcut `AUTO_ASR_VERIFY` gibi açma/kapama desenine uygun).

### Bağımlılıklar

`requirements.txt`'e `speechbrain` eklenir. Model ilk kullanımda internet
üzerinden indirilir (mevcut Wav2Vec2 indirme deseniyle tutarlı).

### İstemci (`lib/main.dart`)

- `_FurkancaPageState._handleSegment`: sunucu yanıtında `rejected == true`
  ise:
  - Geçmişe (`_history`) hiçbir kayıt eklenmez.
  - Hata gösterilmez.
  - `_state` sessizce `listening`'e döner — tıpkı `VadEventType.discardedTooShort`
    davranışı gibi, segment hiç olmamış gibi.
- Kilitli Ayarlar sayfasına (`_AyarlarPage`) yeni "Ses Profili" bölümü:
  - `GET /voice-profile/status` ile mevcut durum gösterilir (örn. "42
    kayıttan oluşturuldu, son güncelleme: ...").
  - "Ses Profilini Yenile" butonu `POST /voice-profile/rebuild` çağırır,
    mevcut `_training`/`_trainStatus` deseniyle aynı şekilde yükleniyor
    durumu ve sonucu snackbar ile gösterir.

## Veri Akışı

1. Kullanıcı Furkanca sayfasında "Başlat"a basar → mevcut VAD dinlemeye
   devam eder (değişmedi).
2. VAD bir konuşma segmenti yakalar (`VadEventType.ended`) → mevcut gibi
   `/translate`'e yüklenir.
3. Sunucu: ASR'den önce speaker verification çalışır.
   - Eşleşmiyorsa → `{"rejected": true, "reason": "speaker_mismatch"}`,
     ağır pipeline çalışmaz.
   - Eşleşiyorsa veya profil yoksa (fail-open) → mevcut akış olduğu gibi
     devam eder.
4. İstemci: `rejected == true` ise sessizce yok sayar, aksi halde mevcut
   davranış (metne çevir, yanıtı seslendir, geçmişe ekle) sürer.
5. Ayarlar sayfasında kullanıcı istediği zaman "Ses Profilini Yenile"ye
   basarak profili güncel kayıtlarla yeniden oluşturabilir.

## Hata Yönetimi

- Enrollment sırasında tek tek dosya okuma/embedding hatası → o dosya
  atlanır, kalan dosyalarla devam edilir. Hiç örnek işlenemezse profil
  oluşturulmaz (fail-open korunur).
- Verify sırasında beklenmeyen hata (model/GPU hatası vb.) → loglanır,
  fail-open uygulanır (doğrulama atlanır, normal akışa devam edilir).
- SpeechBrain modeli indirilemezse (ilk çalıştırmada internet yoksa) → aynı
  fail-open kuralı işler; `SPEAKER_VERIFY_ENABLED` manuel olarak da
  kapatılabilir.

## Test Planı

- **Python birim testleri:** `SpeakerVerificationService` için enroll+verify
  akışı, eşik altı/üstü senaryoları, profil yokken fail-open davranışı,
  bozuk/eksik dosyalarla enrollment.
- **Eşik kalibrasyonu:** `SPEAKER_VERIFY_THRESHOLD` başlangıç değeri
  (0.72) bir tahmindir. Gerçek verilerle kalibre edilmelidir: FurkanV1
  kayıtlarından bir kısmını enrollment dışı tutup kendi sesiyle skor
  dağılımına bakmak, ardından farklı bir konuşmacının (örn. `Admin`
  klasöründeki kayıtlar) skorlarıyla karşılaştırıp eşiği ayarlamak.
- **Flutter/manuel test:** Cihazda Furkanca ekranında (a) başka biri
  konuşunca geçmişe hiçbir şey eklenmediğini, (b) Furkan konuşunca normal
  akışın çalıştığını, (c) Ayarlar sayfasındaki "Ses Profilini Yenile"
  butonunun durum bilgisini doğru güncellediğini doğrulamak.
