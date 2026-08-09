# Kayıt Sayfası: Cihazda Biriktirme + Dinle/Düzenle

**Tarih:** 2026-08-09
**Kapsam:** `lib/main.dart` — yalnızca `_KayitPage`/`_KayitPageState`. Ayrıca `api.py`'de `GET /collect/next-word` için minimal, geriye uyumlu bir eklenti. `_FurkancaPage`, `_AyarlarPage`, `/record`, `/collect/progress` sözleşmesi değişmiyor.

## Sorun

Kayıt sayfasında her tekrar kaydı alınır alınmaz otomatik olarak `/record` endpoint'ine yükleniyor. İstenen: kayıtlar önce cihazda birikecek, kullanıcı dinleyip beğenmediklerini silip yeniden kaydedebilecek, kelimeler arasında sunucuya yüklemeyi beklemeden serbestçe geçebilecek, ve istediği an bekleyen kayıtları toplu olarak sunucuya gönderebilecek.

## Ön Bulgu (koda bakılarak doğrulandı)

`POST /record` (`api.py:209-264`) `rep` numarasını **kendisi** hesaplıyor (`current_count + 1`, sunucudaki dosya sayısına göre) — istemciden `rep` alanı gelse de kullanmıyor. Bu, istemcinin rep numaralarını sunucuyla hassas şekilde senkronize etmesi gerekmediği anlamına gelir; istemci yalnızca bekleyen kayıtları **kaydedildikleri sırayla** yüklemekle yükümlü, sunucu doğru rep'i otomatik atar.

## Tasarım

### 1. Backend Eklentisi (minimal, geriye uyumlu)

`GET /collect/next-word` opsiyonel bir `exclude` query parametresi kabul eder (virgülle ayrılmış kelime listesi, varsayılan boş). `_get_next_word_and_rep(user_id, set_file, exclude=set())` bu kelimeleri atlayarak sıradaki gerçekten farklı, tamamlanmamış kelimeyi döndürür. Parametre verilmezse (`exclude=""`) davranış birebir mevcutla aynıdır — hiçbir mevcut çağıran (varsa) etkilenmez.

### 2. Yerel Depolama

- Bekleyen kayıtlar `getApplicationDocumentsDirectory()/pending_recordings/` altında kalıcı olarak saklanır (geçici dizin kullanılmaz — uygulama kapansa da kaybolmaz).
- Aynı klasörde `manifest.json`: `[{word, filePath, recordedAt}, ...]`. Uygulama açılışında bu dosyadan bekleyen kayıtlar listesi geri yüklenir.
- Silme: hem ses dosyası hem manifest girişi kaldırılır. Yükleme başarılı olunca ikisi de temizlenir.

### 3. Kayıt Akışı

- Mevcut kayıt mekanizması (süre slider'ı, `_startRecording`/`_stopRecording`, eş-zamanlı geri sayım) **değişmeden** kalır — yalnızca kayıt tamamlandıktan sonra ne olduğu değişir: `/record`'a POST yerine `pending_recordings/`e kaydedilip manifest'e eklenir.
- O kelime için gösterilecek "Tekrar: N / ideal" sayacı, sunucudan gelen `current_count` (yüklenmiş, onaylı) + o kelime için cihazdaki bekleyen kayıt sayısı toplanarak hesaplanır (yalnızca ekranda gösterim için; sunucuya rep numarası göndermiyoruz, yukarıdaki Ön Bulgu nedeniyle).
- Kelime kartının altında bekleyen tekrarların kısa listesi: her biri ▶ oynat + 🗑 sil.

### 4. Kelimeler Arası Geçiş

"Sonraki Kelime" butonu: cihazda bekleyen kayıtları olan **tüm kelimeler** `exclude` parametresi olarak `/collect/next-word`'e gönderilir; sunucu gerçekten farklı, tamamlanmamış bir kelime döndürür. Yükleme beklenmez. Sunucudaki tüm tamamlanmamış kelimeler zaten cihazda bekliyorsa (`done: true` ama aslında yerelde tamamlanmış kelimeler var), ekranda "Bu setteki kelimeler için yerel kayıt tamamlandı — yüklemek için Bekleyen Kayıtlar'a git" mesajı gösterilir (mevcut "Tüm kelimeler tamamlandı!" mesajından ayrıştırılır, çünkü henüz sunucuya hiçbir şey gitmemiş olabilir).

### 5. "Bekleyen Kayıtlar" Ekranı

Kayıt sayfasından bir üst-bar ikonuyla açılır (bekleyen toplam kayıt sayısı rozet olarak gösterilir). Kelimeye göre gruplanmış tam liste; her kayıt için oynat/sil. Üstte **"Tümünü Yükle"** butonu:

- Her bekleyen kaydı, kaydedildiği sırayla, mevcut `/record` multipart endpoint'ine (değişmeden) POST eder.
- Her kayıt için durum gösterilir: yükleniyor / başarılı / hata.
- Başarılı yüklenenler yerel depodan ve manifest'ten silinir. Hata alanlar listede kalır, kullanıcı tekrar "Tümünü Yükle"ye basarak yeniden deneyebilir.
- Tüm yüklemeler bitince `/collect/progress` ve (varsa güncel `set_file` için) `/collect/next-word` yeniden çağrılıp Kayıt sayfası tazelenir.

### 6. İlerleme Göstergesi

Mevcut ilerleme çubuğu (`_completedWords / _totalWords`, sunucu-onaylı) değişmeden kalır; yanına küçük bir ek not eklenir: "+N kelime cihazda bekliyor" (yalnızca istemci tarafında hesaplanan bekleyen kelime sayısı, sunucuya gitmiyor).

## Hata Yönetimi

- Yükleme sırasında ağ hatası: o kayıt "hata" durumunda listede kalır, diğer kayıtların yüklenmesi durmaz (best-effort, sıralı ama birbirinden bağımsız).
- Manifest dosyası bozuk/okunamazsa: boş bekleyen liste ile başlanır, hata sessizce loglanır (kullanıcı verisi kaybı olmaz çünkü ses dosyaları diskte kalır — yalnızca uygulama içi liste boş görünür; kapsam dışı bir kurtarma mekanizması gerekmez, bu proje ölçeğinde aşırı mühendislik olur).

## Kapsam Dışı

- Ses kırpma/kesme, kelime yeniden atama (önceki tur netleştirmede kapsam dışı bırakıldı).
- Birden fazla kullanıcı/cihaz arasında bekleyen kayıt senkronizasyonu.
- `/collect/progress` endpoint'inde backend değişikliği (yalnızca `/collect/next-word`'e `exclude` eklentisi var).

## Test Planı

- Backend: `_get_next_word_and_rep`'in `exclude` parametresiyle doğru kelimeyi atladığını doğrulayan birim/manuel test (mevcut Python test altyapısı `tests/test_utils.py` ile aynı tarzda, saf fonksiyon — gerçek dosya sistemi/model gerekmiyor).
- Mobil: manifest okuma/yazma/silme mantığı saf Dart fonksiyonları olarak çıkarılabiliyorsa unit test edilir (dosya sistemi gerektiren kısımlar, önceki turlardaki gibi platform mocking'in kırılganlığı nedeniyle otomatik test kapsamı dışında).
- Manuel: bir kelime için 2 kayıt al, birini sil, "Sonraki Kelime"ye bas (yükleme olmadan farklı kelime geldiğini doğrula), "Bekleyen Kayıtlar"a git, "Tümünü Yükle"ye bas, sunucuda dosyaların doğru rep numaralarıyla oluştuğunu ve ilerleme çubuğunun güncellendiğini doğrula.
