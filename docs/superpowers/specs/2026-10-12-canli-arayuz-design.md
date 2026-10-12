# Mobil Uygulama: Arayüzü Sunucudan Yükleme ve Uygulama İçi Güncelleme — Tasarım

Tarih: 2026-10-12
Durum: Onaylandı (sohbette bölüm bölüm), yazılı spec incelemede

## Sorun

Furkanca mobil uygulaması Capacitor ile paketleniyor ve arayüz kodu (`web/dist`)
APK'nın içine gömülü. Arayüzdeki her değişiklik yeni bir APK derleyip Furkan'ın
telefonuna elle kurmayı gerektiriyor. Kullanıcı bunu istemiyor: APK yalnızca
yapısal (native) değişikliklerde yeniden kurulsun.

## Kararlar (kullanıcıyla)

- **Yaklaşım 3:** Arayüz canlı web sitesinden yüklenir (günlük değişiklikler) +
  uygulama içi "güncelleme var" düğmesi (nadir native değişiklikler).
- **İnternet hep var:** çevrimdışı açılış (service worker) gerekmez.
- Mevcut APK'lar aynı anahtarla imzalı; yeni APK eskisinin **üstüne** kurulur,
  silme gerekmez (silmek bekleyen kayıtları siler).

## Başarı ölçütü

- Web'de `npm run build` yapıldıktan sonra Furkan uygulamayı yeniden açınca yeni
  arayüzü görür; APK kurulmaz.
- Native değişiklikte uygulamada şerit çıkar; dokununca yeni APK eskisinin
  üstüne kurulur.

## Tasarım

### Bölüm 1: Arayüzü sunucudan yükle (`.claude/worktrees/react-rewrite`, dal `worktree-react-rewrite`)

- **Kabuk:** `web/capacitor.config.ts`'e
  `server: { url: 'https://app.furkanca.algorixstudio.com', errorPath: 'offline.html' }`.
  Uygulama açılınca arayüzü `pronouns-web` servisinin sunduğu canlı siteden
  yükler. `webDir: 'dist'` kalır (Capacitor'ın derleme için ihtiyacı var;
  `offline.html` buradan APK'ya girer).
- **İnternet yoksa:** `web/public/offline.html` — APK içinden açılan, dış kaynak
  kullanmayan tek sayfa: "İnternet bağlantısı yok" + "Tekrar dene" düğmesi
  (`location.href` ile canlı adresi yeniden yükler).
- **Önbellek:** `web/public/serve.json` ile `index.html` için
  `Cache-Control: no-cache` (her açılışta sunucuya sorulur; ETag aynıysa 304).
  `assets/*` dosya adları her derlemede değiştiği için önbelleklenebilir.
  `serve -s dist` `dist/serve.json`'ı okur; nginx'e dokunulmaz.
- **API:** Sayfa kökeni `https://app.furkanca.algorixstudio.com` olur; API
  `https://furkanca.algorixstudio.com`'da ve CORS `*` (bugünkü web sürümüyle
  aynı). Değişiklik yok.
- **Tek seferlik geçiş:** Bekleyen kayıtlar (IndexedDB) kökene bağlı; kabuğa
  geçince eski kökendeki kayıtlar görünmez. Kurulumdan **önce** eski uygulamada
  Bekleyen Kayıtlar → "Tümünü Yükle".
- **Mikrofon riski:** Uzaktan yüklenen sayfada `getUserMedia` izninin
  çalıştığı planın ilk adımında telefonda doğrulanır. Çalışmazsa iş durur ve
  kullanıcıya danışılır (yaklaşımı değiştirebilecek tek risk).

### Bölüm 2: Uygulama içi güncelleme

- **Sürüm numarası:** `web/deploy-android.sh` her derlemede
  `/home/mayasoft/app/Pronouns/data/apk/version.json`'daki `versionCode`'u bir
  artırır (dosya yoksa 1'den başlar → ilk derleme 2), `versionName`
  `"1.<versionCode-1>"`, `built_at` ISO zaman. Gradle bu değerleri
  `-PversionCode=… -PversionName=…` ile alır; verilmezse `1` / `"1.0"`.
  İmza anahtarı (debug keystore) değişmez. `version.json` APK başarıyla
  kopyalandıktan **sonra** yazılır (derleme başarısızsa numara artmaz).
- **Sunucu (`/home/mayasoft/app/Pronouns`, dal `canli-arayuz`):**
  `GET /apk/version` → `version.json` içeriği; dosya yoksa ya da okunamazsa
  `{"versionCode": 0}`. API anahtarı istemez (`_API_KEY_EXEMPT_PATHS`'e eklenir).
- **Uygulama:**
  - Eklentiler: `@capacitor/app` (kurulu `build` = versionCode) ve
    `@capacitor/browser` (APK adresini telefonun tarayıcısında açar). Native
    değişiklik oldukları için bu APK bir kez elle kurulur (son elle kurulum).
  - `web/src/lib/appUpdate.ts`:
    - `needsUpdate(installed: number, server: number): boolean` — `server > installed`.
    - `checkForUpdate(deps): Promise<boolean>` — yalnızca native platformda
      (`Capacitor.isNativePlatform()`) kurulu sürümü ve `GET /apk/version`'ı
      karşılaştırır; web'de ya da herhangi bir hata/zaman aşımında `false`.
      Bağımlılıklar (platform kontrolü, kurulu sürüm, fetch) test için
      parametreyle verilir.
  - Ana sayfada (`HomePage.tsx`) açılışta kontrol; `true` ise üstte şerit:
    "**Uygulama güncellemesi var** · Güncelle". Dokununca
    `Browser.open({ url: <API>/apk })`; APK iner, Android güncellemeyi sorar.
  - Web tarayıcısında şerit hiç görünmez.

## Hata Yönetimi

| Durum | Davranış |
|---|---|
| Açılışta internet yok | `offline.html`: "İnternet bağlantısı yok" + "Tekrar dene" |
| `/apk/version` hata / zaman aşımı | Şerit çıkmaz, uygulama normal çalışır |
| `version.json` yok / bozuk | `{"versionCode": 0}` → şerit çıkmaz |
| APK derlemesi başarısız | `version.json` değişmez, eski APK yerinde kalır |
| Web'de (APK dışı) | Sürüm kontrolü yapılmaz |
| Uzaktan sayfada mikrofon izni çalışmıyor | İş durur, kullanıcıya danışılır |

## Test Planı

- **pytest:** `/apk/version` dosya varken içeriği, yokken ve bozukken
  `{"versionCode": 0}`; API anahtarı olmadan 200.
- **vitest:** `needsUpdate` (büyük / eşit / küçük); `checkForUpdate` web'de
  `false`, fetch hatasında `false`, sunucu daha yeniyse `true`, eşitse `false`.
- **Elle (telefon):** mikrofon (Furkanca ve Kayıt), web değişikliğinin yeniden
  açılışta gelmesi, internet kapalıyken `offline.html`, sürüm 3 derlenince
  şeridin çıkması ve üstüne güncelleme.

## Dağıtım

1. Sunucu: `canli-arayuz` → `master` merge; API'yi kullanıcı yeniden başlatır.
2. Web: `cd .claude/worktrees/react-rewrite/web && npm run build`
   (**`deploy-web.sh` kullanılmaz**).
3. Furkan eski uygulamada Bekleyen Kayıtlar → "Tümünü Yükle".
4. APK: `web/deploy-android.sh` (sürüm 2); Furkan `/apk`'dan **üstüne** kurar.
5. Sonrası: arayüz değişiklikleri yalnızca `npm run build`; native
   değişikliklerde `deploy-android.sh` + Furkan şeride dokunur.
- Geri alma: önceki APK yedeği (`pronouns_backup_*.apk`) üstüne kurulamaz
  (düşük sürüm); gerekirse aynı kodu `server.url`'süz yeni sürüm numarasıyla
  derlemek gerekir.

## Kapsam Dışı

- Çevrimdışı açılış (service worker), Play Store, sessiz otomatik güncelleme.
- Release keystore'a geçiş; nginx yapılandırması; Flutter uygulaması.
- `worktree-react-rewrite` dalını `master`'a merge etmek.
