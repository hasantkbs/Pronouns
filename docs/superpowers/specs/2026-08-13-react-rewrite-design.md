# Furkanca Mobil/Web Uygulaması — React'e Geçiş Tasarımı

**Tarih:** 2026-08-13
**Durum:** Taslak (onay bekliyor)

## Amaç

Mevcut Flutter tabanlı Furkanca uygulamasını (3 panel: Furkanca, Kayıt, Ayarlar)
React tabanlı, tek kod tabanından hem web sitesi hem de Android APK üretebilen
bir uygulamayla değiştirmek. İkinci hedef: uygulamanın hangi ağdan olursa olsun
(VPN gerekmeden) backend'e ve ses kaydı/veri setine erişebilmesi.

## Keşif Bulguları (önemli, planı değiştiren)

- **`https://furkanca.algorixstudio.com` zaten canlı ve çalışıyor.** Host
  üzerindeki nginx, bu domaini `127.0.0.1:8001`'e (Pronouns API) proxy'liyor,
  geçerli Let's Encrypt sertifikası var (`/etc/nginx/sites-enabled/furkanca`).
  Dışarıdan doğrulandı: `GET /apk` → `200 OK`, doğru içerik. Bu, "VPN'siz
  erişim" hedefini (bkz. `2026-08-10-vpnsiz-erisim-design.md`) zaten
  karşılıyor — yeni domain almaya veya IP bazlı erişime gerek yok.
- Backend CORS zaten `allow_origins=["*"]` (api.py:45) — yeni bir web
  origin'i için ek backend değişikliği gerekmiyor.
- Ses upload endpoint'leri (`/record`, `/translate`) gelen dosyayı **hiç
  dönüştürmeden** (`shutil.copyfileobj`) `.wav` uzantısıyla kaydediyor —
  yani istemcinin gerçek WAV bytes göndermesi şart.
- **Furkanca paneli basit "kayda bas" değil, sürekli dinleme + VAD
  (Voice Activity Detection) ile çalışıyor** (`lib/vad.dart`): ham PCM16
  akışı RMS eşiğiyle (`onsetThreshold=0.012`) konuşma başlangıcı/bitişini
  algılıyor, 300ms pre-roll tutuyor, 1sn sessizlikte veya 12sn'de segmenti
  kapatıp WAV'a çevirip `/translate`'e yolluyor. Bu, platform bağımsız saf
  mantık (`VadSegmenter` sınıfı) — React'e birebir port edilecek.
- **Kayıt paneli** basit sabit süreli (geri sayımlı) kayıt kullanıyor, VAD
  yok. Ayrıca **çevrimdışı-öncelikli**: kayıtlar önce cihaza (manifest.json
  + wav dosyaları, `lib/pending_recordings.dart`) yazılıyor, "Bekleyen
  Kayıtlar" ekranından toplu olarak `/record`'a yükleniyor. Web'de bunun
  karşılığı IndexedDB.
- `/record` multipart alanları: `user_id`, `word`, dosya alanı `audio`
  (rep sayısı sunucu tarafında hesaplanıyor, istemci göndermiyor).
- API anahtarı (`X-API-Key`) zaten var (`vpnsiz-erisim` tasarımından); web
  tarafında JS bundle içine gömülecek. APK'nın aksine web'de bu tamamen
  görünürdür (DevTools). Proje zaten benzer risk seviyesini kabul etmiş
  durumda (Ayarlar şifresi düz metin) — bilinçli olarak aynı kabul devam
  ediyor, gerçek bir secret-management çözümü değil.

## Kapsam

**Dahil:**
- 3 panelin birebir işlevsel taşınması: Furkanca (dinle → düzelt → oynat),
  Kayıt (yeni kelime/ses toplama, set seçimi, ilerleme), Ayarlar (model
  bilgisi, fine-tune ayarları, ses profili, admin login kilidi).
- Görsel tasarımın yenilenmesi (frontend-design skill ile, implementasyon
  aşamasında ayrıca ele alınacak — bu doküman mimari/veri akışına odaklanır).
- Web dağıtımı: `https://app.furkanca.algorixstudio.com` (yeni subdomain,
  sadece React static build'i sunar).
- Android APK dağıtımı: Capacitor ile aynı React kod tabanından derlenir,
  mevcut `/apk` endpoint'inden dağıtılır (build artifact değişir, endpoint
  aynı kalır).
- Ses format güvenliği: istemci tarafında (web) kayıt WAV'a normalize edilir.

**Kapsam Dışı:**
- Backend/API'de işlevsel değişiklik (endpoint'ler, iş mantığı aynı kalır).
- Yeni özellik eklenmesi (kullanıcı "3 paneli birebir taşı" dedi).
- iOS desteği.
- Gerçek secret-management / kullanıcı bazlı auth (mevcut basit API-key
  modeli korunuyor).
- Flutter kodunun silinmesi (depoda kalır, sadece dağıtım/geliştirme durur).

## Mimari

### Kod tabanı
Tek React (Vite + TypeScript) projesi, **Capacitor** ile Android'e sarılır.
- Web: `npm run build` → statik dosyalar → nginx (`app.furkanca...`) sunar.
- Android: `npx cap sync android && ./gradlew assembleRelease` → APK →
  mevcut `data/apk/pronouns.apk` konumuna deploy edilir (aynı `/apk`
  endpoint'i, kullanıcı tarafında değişiklik yok).

### Platform farkı — ses kaydı
Web ve Capacitor/Android **aynı kodu** kullanır: Capacitor bir native
WebView'de çalıştığı için `navigator.mediaDevices.getUserMedia` ve Web
Audio API her iki platformda da çalışır (Android tarafında sadece
`AndroidManifest.xml`'e `RECORD_AUDIO` izni eklenir, native bir ses
plugin'i gerekmez). Bu, platform-özel kod miktarını ciddi azaltır.

Ortak ses hattı (`lib/vad.ts`, Dart'taki `vad.dart`'ın birebir portu):
1. `AudioContext` + `ScriptProcessorNode` ham PCM16 chunk'ları üretir
   (context'in kendi `sampleRate`'i kullanılır, 16000'e zorlanmaz — backend
   zaten `librosa.load(sr=...)` ile yeniden örnekliyor).
2. **Furkanca paneli:** chunk'lar `VadSegmenter`'a beslenir; `ended`
   olayında segment `pcmToWav()` ile WAV Blob'a çevrilip `/translate`'e
   gönderilir.
3. **Kayıt paneli:** VAD kullanılmaz; sabit süreli (geri sayımlı) kayıt
   PCM chunk'larını biriktirir, süre bitince `pcmToWav()` ile WAV Blob
   üretir, IndexedDB'ye ("pending recordings") yazılır.

Bu sayede backend hiç değişmeden (`shutil.copyfileobj` aynı kalır), her iki
platformdan da gerçek WAV içeriği garanti edilir; `MediaRecorder`/WebM yolu
kullanılmaz (VAD'ın ham PCM'e ihtiyacı olduğu için zaten uygun değildi).

### API iletişimi
Mevcut endpoint sözleşmesi birebir korunur (`/settings`, `/collect/*`,
`/translate`, `/record`, `/model/info`, `/voice-profile/*`, vb.), sadece
`kBaseUrl` yerine `VITE_API_BASE_URL=https://furkanca.algorixstudio.com`
ortam değişkeni kullanılır — hem web build'inde hem Capacitor build'inde
aynı, sabit HTTPS adresi.

### Sayfa/komponent yapısı (özellik paritesi)
```
src/
  pages/
    FurkancaPage.tsx      // sürekli dinle (VAD) → /translate → oynat, geçmiş
    KayitPage.tsx         // /collect/word-sets, /collect/next-word,
                           // /collect/progress, sabit süreli kayıt → IndexedDB
    PendingUploadsPage.tsx // IndexedDB'deki bekleyen kayıtları listeler,
                           // "Tümünü Yükle" → /record
    AyarlarLoginPage.tsx  // admin/adminpassword kilidi (mevcut basit model)
    AyarlarPage.tsx       // /settings, /model/info, /voice-profile/*
  lib/
    api.ts                 // fetch wrapper + X-API-Key header
    vad.ts                  // VadSegmenter + calculateRms + pcmToWav (vad.dart portu)
    audioCapture.ts          // getUserMedia + ScriptProcessorNode → PCM16 chunk stream
    pendingRecordingsStore.ts // IndexedDB CRUD (pending_recordings.dart portu)
  App.tsx                  // routing (3 buton ana ekran, aynı Flutter akışı)
```

## Hata Yönetimi

- Mevcut Flutter'daki hata gösterim deseni korunur: başarısız istekte
  kullanıcıya kısa bir toast/snackbar ("Hata: ..."), akışı durdurmayan
  (non-blocking) hatalar.
- Mikrofon izni reddedilirse (web: `getUserMedia` reddi; Android: Capacitor
  permission reddi) açık bir izin isteği ekranı gösterilir — sessiz
  başarısızlık yok.
- `getUserMedia` PCM chunk akışında beklenmedik biçimde kesilirse (ör. sekme
  arka plana alınır) kullanıcıya "kayıt işlenemedi, tekrar deneyin" mesajı
  gösterilir, eksik/yarım segment backend'e **gönderilmez**.

## Dağıtım / Geçiş Planı

1. `app.furkanca.algorixstudio.com` için nginx site config + DNS A kaydı
   eklenir (mevcut `algorixstudio.com` altında, aynı sunucu, aynı IP).
   Certbot ile SSL genişletilir.
2. React+Capacitor projesi, mevcut Pronouns deposu içinde `web/` dizininde
   (Flutter'ın `lib/`, `android/` dizinleriyle aynı seviyede, ayrı bir repo
   açılmaz) geliştirilir, backend'e dokunulmaz.
3. Web build → yeni subdomain'e deploy, Furkan ile test edilir.
4. Android build → `/apk` endpoint'ine deploy edilir (Flutter'ın yerini
   alır), Furkan'ın telefonunda test edilir.
5. Her iki platform da onaylanınca Flutter geliştirme/dağıtımı durdurulur
   (kod silinmez, `pubspec.yaml`/`android/` dizini arşiv olarak kalır).

## Test Planı

- **Manuel (öncelikli, tek gerçek kullanıcı — Furkan):** her 3 panelde uçtan
  uca akış, hem web hem APK'da; özellikle Kayıt panelinde ses kalitesi
  (SNR/duration) yeni pipeline ile eski Flutter kayıtlarıyla karşılaştırılır
  (regresyon kontrolü — bkz. dünkü SNR temizliği, yeni format hatası aynı
  soruna yol açmasın).
- **Otomatik:** `vad.ts` için birim testi — `calculateRms` (sessizlik/ses
  ayrımı), `VadSegmenter` (started/ended/discardedTooShort olayları,
  pre-roll dahil), `pcmToWav` (WAV header alanlarının doğruluğu). Bu
  fonksiyonlar saf mantık olduğundan tarayıcı ortamı gerekmeden (Vitest,
  Node) test edilir.
- CORS, API-key header'ın her istekte doğru gittiği manuel `curl`/DevTools
  Network sekmesi ile doğrulanır.
