# VPN'siz Erişim + API Anahtarı Koruması — Tasarım

**Tarih:** 2026-08-10
**Durum:** Onaylandı

## Amaç

Şu an mobil uygulama (`kBaseUrl`) sunucunun VPN-içi özel IP'sine
(`10.10.108.10:8001`) sabitlenmiş durumda; telefonun API'ye erişebilmesi
için OpenVPN bağlantısının (`KervanVPN.ovpn`) aktif olması gerekiyor. Bu,
APK'yı kuran normal bir kullanıcı için ekstra bir engel. Hedef: kullanıcı
APK'yı kurduktan sonra VPN'e hiç bağlanmadan, normal mobil veri/Wi-Fi ile
uygulamayı kullanabilsin.

## Keşif Bulguları (önemli)

- Sunucu (`mayasoftlnx01`) zaten `0.0.0.0:8001` üzerinde dinliyor ve makinenin
  genel IP'si (`88.255.236.143`) internetten doğrudan erişilebilir durumda.
  `ufw` aktif değil / kurulu değil.
- Servis logları bunu doğruluyor: `/.aws/credentials`, `/.git/config`,
  `/info.php` gibi path'lere rastgele IP'lerden (internet tarayıcı botları)
  istekler zaten geliyor.
- Sonuç: "VPN gerekliliği" bugüne kadar bir ağ güvenlik sınırı değil, sadece
  istemcinin adres seçimiydi. API zaten sızıntıya açık; VPN kaldırıldığında
  gerçek bir yeni saldırı yüzeyi *açılmıyor*, sadece adresin APK içine
  gömülüp dağıtılmasıyla bu erişim "keşfedilebilirliği" artıyor.
- `/fine-tune`, `/train`, `/settings` (POST), `/record` gibi durum
  değiştiren uçlarda hâlihazırda hiçbir kimlik doğrulama yok.

Bu nedenle, VPN gereksinimini kaldırmakla birlikte, dağıtılan adresin
APK'dan çıkarılabilir olması riskine karşı hafif bir API anahtarı koruması
da ekleniyor.

## Kapsam Dışı

- Gerçek kullanıcı bazlı kimlik doğrulama (OAuth, JWT, per-user token).
- TLS/HTTPS (domain adı, nginx reverse proxy, sertifika yönetimi) — mevcut
  düz HTTP korunuyor; `usesCleartextTraffic="true"` zaten ayarlı, ek Android
  config değişikliği gerekmiyor.
- Rate limiting / DDoS koruması.
- Sunucu tarafında firewall kurulumu (`ufw` vb.) — ayrı bir sistem yönetimi
  konusu, bu tasarımın kapsamında değil.

Bu proje zaten Ayarlar kilidini (`admin`/`adminpassword`) düz metin olarak
`main.dart` içinde tutuyor; API anahtarı da aynı basitlik/risk kabul
seviyesinde tasarlanıyor — gerçek bir secret-management çözümü değil,
rastgele bot/tarayıcı trafiğinin GPU'yu meşgul etmesini veya ayarları/model
eğitimini tetiklemesini engelleyen bir caydırıcı katman.

## Tasarım

### Sunucu (`api.py`, `config.py`)

- `config.py`'ye sabit bir değer eklenir:
  ```python
  API_SECRET_KEY = "<rastgele-üretilmiş-uzun-değer>"
  ```
- `api.py`'de bir Starlette middleware eklenir: gelen her istekte
  `X-API-Key` header'ı `config.API_SECRET_KEY` ile eşleşmiyorsa `401
  Unauthorized` (`{"error": "invalid_api_key"}`) döner.
- **Whitelist (anahtar gerektirmeyen yollar):**
  - `/apk` — APK'yı henüz kurmamış bir kullanıcı anahtarı bilemez; ilk
    kurulum indirmesi açık kalmalı.
  - `/download/{filename}` — sentezlenmiş ses `audioplayers` paketiyle
    `UrlSource` üzerinden oynatılıyor; bu akışta custom header eklemek
    platformlar arası güvenilir değil, bu yüzden bu uç anahtarsız kalır.
    Dosya adları tahmin edilemez (UUID tabanlı) ve içerik sadece
    sentezlenmiş ses parçası olduğu için kalan risk düşük kabul edilir.
  - Diğer her şey (`/translate`, `/record`, `/settings`, `/fine-tune`,
    `/train`, `/self-learn`, `/words`, `/collect/*`, `/model/info`,
    `/training/status`, ayrıca FastAPI'nin otomatik `/docs`,
    `/openapi.json`, `/redoc` uçları) middleware tarafından korunur —
    yeni bir endpoint eklendiğinde varsayılan olarak korumalı olması için
    whitelist'e değil, middleware'in kendisine güvenilir (secure-by-default).

### İstemci (`lib/main.dart`)

- `kBaseUrl` değişir:
  ```dart
  const String kBaseUrl = 'http://88.255.236.143:8001';
  ```
- Yeni sabit: `const String _kApiKey = '<config.py ile aynı değer>';`
- Yeni yardımcı fonksiyon:
  ```dart
  Map<String, String> _authHeaders([Map<String, String>? extra]) =>
      {'X-API-Key': _kApiKey, ...?extra};
  ```
- Anahtar gerektiren tüm mevcut çağrılar güncellenir (whitelist'teki
  `/download` ve APK indirme hariç):
  - `http.get(...)` çağrılarına `headers: _authHeaders()` eklenir
    (`/collect/word-sets`, `/collect/next-word`, `/collect/progress`,
    `/settings` GET, `/model/info`).
  - `http.post(...)` çağrılarına mevcut `Content-Type` header'ıyla
    birleştirilerek `_authHeaders({'Content-Type': 'application/json'})`
    kullanılır (`/settings` POST, `/fine-tune`).
  - `http.MultipartRequest` çağrılarına `req.headers.addAll(_authHeaders())`
    eklenir (`/translate`, `/record`).

### Veri Akışı

Değişiklik sadece istemcinin hangi adrese ve hangi header'larla istek
attığında; sunucu tarafındaki iş mantığı (ASR/NLU/sentez, kayıt, eğitim)
aynı kalır. Middleware, iş mantığından önce (route handler'lara girmeden)
çalışır.

## Hata Yönetimi

- Header eksik/yanlışsa: `401 Unauthorized`, JSON gövde. İstemci tarafında
  bu durumun operasyonel olarak oluşmaması beklenir (anahtar APK içine
  sabit gömülü); oluşursa mevcut genel hata gösterimi (`_snack('Hata: ...')`
  veya `_history`'ye hata girişi) zaten devrede olan akışları kullanır, ek
  özel bir UI dalı eklenmez.
- `/download/{filename}` ve `/apk` whitelist'te olduğu için bu uçlarda
  davranış değişmez.

## Test Planı

- **Sunucu:** middleware için birim/entegrasyon testi — doğru anahtarla
  200, eksik/yanlış anahtarla 401, whitelist'teki iki yol için anahtarsız
  200.
- **Manuel:** telefon VPN'den ayrılıp sadece mobil veriyle (a) Furkanca
  ekranında konuşma çevirisi, (b) Kayıt ekranında yeni kelime kaydı, (c)
  Ayarlar ekranında ayar kaydetme/fine-tune başlatma akışlarının çalıştığı
  doğrulanır.
- Var olan `curl` tabanlı `SUNUCU_TODO.md` test komutları, `X-API-Key`
  header'ı eklenerek yeniden doğrulanabilir.
