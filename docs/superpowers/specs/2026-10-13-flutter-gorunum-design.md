# Mobil Arayüz: Flutter Görünümüne Dönüş — Tasarım

Tarih: 2026-10-13
Durum: Onaylandı (sohbette iki bölüm), yazılı spec incelemede

## Sorun

Furkan'ın telefonuna canlı arayüzlü APK (sürüm 2) kuruldu. Kullanıcının
gözlemi: eski arayüz (Flutter uygulaması, `lib/main.dart`) daha kolaydı ve
kullanıcı için korunmalı. React arayüzünde (`web/src/pages/*`) neredeyse hiç
tasarım yok: ana sayfa düz bir bağlantı listesi, sayfalarda CSS dosyası yok,
toplam 4 satır satır içi stil var. Flutter'da koyu mor tema, ikonlu büyük
renkli düğmeler, kartlar ve okunaklı yazılar vardı.

## Kararlar (kullanıcıyla)

- **Seçenek 1:** React arayüzü Flutter'ın görünümüne birebir getirilir.
  Flutter'a geri dönülmez (A, B1, C ve APK'sız güncelleme kaybolurdu).
- **Yalnızca görünüm değişir.** Kayıt, yükleme, dokun-başlat/bitir, cümle
  modu, bekleyen kayıt kuyruğu ve güncelleme kontrolünün mantığı aynen kalır.
- Dağıtım yalnızca `npm run build`; APK kurulmaz (canlı arayüz).

## Başarı ölçütü

- Furkan uygulamayı yeniden açınca altı ekranı Flutter'daki görünümle görür:
  koyu tema, renkli büyük düğmeler, ikonlar, kartlar.
- Bugün çalışan her akış (kelime ve cümle kaydı, bekleyen yükleme,
  Furkanca dokun-konuş, ayarlar) aynı davranır.

## Tasarım (`.claude/worktrees/react-rewrite`, dal `worktree-react-rewrite`, `web/src/`)

### Bölüm 1: Tema ve ortak parçalar

- **Tema** `src/theme.css` (tek dosya, `main.tsx`'te import edilir):
  - Flutter `ColorScheme.fromSeed(0xFF6C63FF, Brightness.dark)` karşılığı
    koyu zemin, yüzey ve yazı renkleri CSS değişkenleri olarak
    (`--accent: #6C63FF`, `--settings-accent: #F59E0B`, `--error` kırmızı).
  - Yazı: `Roboto, system-ui, sans-serif` (dış font yüklenmez).
  - Sayfa kenar boşluğu 24 px; telefon genişliğinde yatay kaydırma yok.
  - Yeni kütüphane eklenmez (Tailwind, MUI vb. yok).
- **İkonlar** `src/components/Icon.tsx`: Flutter'da kullanılan Material
  ikonlarının satır içi SVG kopyaları (`spatial_audio_off`,
  `record_voice_over`, `mic`, `mic_off`, `graphic_eq`, `stop`, `tune`,
  `arrow_forward_ios`, `arrow_back`, `folder_open`, `list_alt`, `timer`,
  `skip_next`, `check_circle`, `cloud_upload`, `hourglass_top`,
  `error_outline`, `play_arrow`, `delete_outline`, `lock_outline`,
  `person_outline`, `key`, `login`, `save`, `system_update`). Dış ikon fontu
  yüklenmez.
- **Ortak bileşenler** `src/components/`:

| Bileşen | Flutter karşılığı |
|---|---|
| `AppBar` | `AppBar`: geri oku (ana sayfaya ya da bir üst sayfaya), başlık, sağda isteğe bağlı eylem (rozetli ikon) |
| `BigButton` | `_BigButton`: geçişli renk, 20 px köşe, gölge, yarı saydam kutuda ikon, başlık + alt yazı, sağda ok |
| `FilledButton` / `OutlinedButton` | `FilledButton.icon` / `OutlinedButton.icon`: 14 px köşe, ikon + kalın yazı, devre dışı görünümü |
| `ResultTile` | `_ResultTile`: renkli çerçeveli kutu, küçük renkli etiket + metin |
| `SectionHeader` | `_SectionHeader`: 13 px kalın, ana renkte başlık |
| `ProgressBar` | `LinearProgressIndicator` (8 px, yuvarlak) |
| `SegmentedButton` | `SegmentedButton`: seçili parça dolu |

### Bölüm 2: Sayfalar

| Sayfa | Görünüm (davranış değişmez) |
|---|---|
| **Ana sayfa** (`HomePage`) | Nabız gibi atan mor yuvarlak logo (2 sn, CSS animasyonu), "Furkanca" (32 px, kalın, ana renk), "Konuşma Asistanı". Üç `BigButton`: **Furkanca** "Konuş → Düzeltilmiş metin & ses" (mor→mavi `#6C63FF→#3B82F6`), **Kayıt** "Kelime & ses kaydı yükle" (yeşil `#10B981→#059669`), **Ayarlar** "Model tercihi & ince ayar" (turuncu→kırmızı `#F59E0B→#EF4444`). Altta soluk "Kullanıcı: FurkanV1". Güncelleme şeridi en üstte, temaya uygun kart + "Güncelle" düğmesi. |
| **Furkanca** | `AppBar` "Furkanca". Açıklama: "Düğmeye dokun, konuş, bitince tekrar dokun — AI konuşmanı düzeltilmiş hâle çevirsin ve yüksek sesle okusun." İkonlu durum satırı (boşta `mic_off`, diğer durumlarda `graphic_eq`). Geniş `FilledButton`: boşta mor "Konuşmaya Başla" (`mic`), kayıtta kırmızı "Bitir" (`stop`) + süre; yükseklik bugünkü gibi büyük (en az ekranın %30'u). Kayıtta ses seviyesi `ProgressBar`. Hata `ResultTile` (kırmızı). Geçmiş: zamanlı (`SS:DD:ss`) renkli kartlar, en yeni üstte; boşken "Henüz çeviri yok." |
| **Kayıt** | `AppBar` "Kayıt"; sağda rozetli `folder_open` → Bekleyen Kayıtlar. "İlerleme: x / y kelime\|cümle" + yüzde, `ProgressBar`; bekleyen varsa "+N kayıt cihazda bekliyor". **Kelime \| Cümle** `SegmentedButton`. Çerçeveli, `list_alt` ikonlu set seçimi ("Kelime Seti" / "Cümle Seti"). Hedef kartı: kelime 30 px / cümle 24 px kalın, altında "Sıradaki Tekrar: n · Mevcut: m / ideal"; set bitince `check_circle` + "Tüm kelimeler tamamlandı!" kartı. Kelime modunda `timer` ikonlu "Kayıt süresi: n sn" kaydırıcı (bugünkü 1–5), kayıtta geri sayım `ProgressBar` + "Kayıt alınıyor... n sn kaldı", `FilledButton` "Kaydet (Cihazda Sakla)". Cümle modunda büyük başlat/bitir `FilledButton` ("Okumaya Başla" / "Bitir" + süre). Altında `OutlinedButton` "Sıradaki Kelime\|Cümle" (`skip_next`, bugünkü koşulla) ve cümle modunda "Atla". Durum mesajı `ResultTile` "Durum". |
| **Bekleyen Kayıtlar** | `AppBar` "Bekleyen Kayıtlar" (geri → Kayıt). "N kayıt, M kelime bekliyor." `FilledButton` "Tümünü Yükle" (`cloud_upload`, yüklenirken `hourglass_top` "Yükleniyor..."). Kelimeye göre gruplu liste: 18 px kalın başlık, "Tekrar n" satırları; yüklenirken dönen gösterge, hatada kırmızı `error_outline` (hata metni `title`), boştayken `play_arrow` ve `delete_outline` ikon düğmeleri. Boşken "Bekleyen kayıt yok." |
| **Ayarlar Girişi** | `AppBar` "Ayarlar Girişi". Ortada turuncu `lock_outline` (48 px), çerçeveli ikonlu "Kullanıcı Adı" ve "Şifre" alanları, kırmızı hata metni, turuncu `FilledButton` "Giriş Yap" (`login`). |
| **Ayarlar** | `AppBar` "Model Ayarları"; turuncu vurgu. Bugünkü alanlar, Flutter biçiminde: turuncu model bilgi kartı (`ResultTile`, bugünkü "Model • WER • Veri" metni); `SectionHeader` "Model Tercihi" + `SegmentedButton` (sunucudaki değer listede yoksa — bugün `whisper` — sona eklenip seçili gösterilir; kaydedilen değer değişmez); "Fine-Tune Algoritması" + `SegmentedButton` (büyük harf); "Kendi Kendine Öğrenme" anahtarı (alt yazı "Yeni kayıtlar otomatik modele eklenir" / "Manuel eğitim gerektirir"); "Hiperparametreler": Learning Rate, Epoch Sayısı, Batch Size bugünkü sayı alanları olarak kalır, satırda etiket + kalın değer düzeninde. `FilledButton` "Ayarları Kaydet" (`save`). "Ses Profili" `SectionHeader` + `ResultTile` "Durum" + `OutlinedButton` "Ses Profilini Güncelle". Mesaj ve hata `ResultTile`. Yüklenirken ortada dönen gösterge. |

- `offline.html` (APK içinde) aynı koyu tema renklerine getirilir (dış
  kaynak kullanmadan, satır içi stil).

## Hata Yönetimi

Davranış değişmediği için yeni hata durumu yok. Mevcut her hata metni
görünür kalır, yalnızca `ResultTile` (kırmızı) ya da kırmızı metin olarak
çizilir.

## Test Planı

- **vitest + tsc:** mevcut 64 test ve `tsc -b` temiz kalır. Görünüm için
  ayrılan küçük saf yardımcılar test edilir:
  - `progressPercent(completed, total)`: 0 toplamda 0, yuvarlama, %100 üstü
    yok.
  - `groupByWord(takes)`: sırayı koruyarak kelimeye göre gruplama
    (Bekleyen Kayıtlar sayfasındaki satır içi gruplama buraya taşınır).
  - `formatClock(date)`: `SS:DD:ss`, sıfır dolgulu.
  - `withCurrent(options, current)`: listede olmayan sunucu değeri sona eklenir.
  - Ortak bileşenler ve altı sayfa `react-dom/server` ile render edilip
    Flutter düzeninin işaretleri (geri bağlantısı, büyük düğmeler, segmentler,
    `role="alert"`/`role="status"`, devre dışı düğmeler) denetlenir; yeni
    kütüphane eklenmez.
- **Görsel kontrol (Claude yapar):** `npm run build` sonrası yerel önizlemede
  Playwright ile 412×915 boyutunda altı sayfanın ekran görüntüsü; Flutter
  düzeniyle karşılaştırılır; yatay kaydırma olmadığı denetlenir. Kullanıcının
  web'de denemesi gerekmez.
- **Elle (telefon):** Furkan uygulamayı yeniden açar; yeni görünüm gelir,
  kayıt ve Furkanca akışları çalışır.

## Dağıtım

1. Web: `cd .claude/worktrees/react-rewrite/web && npm run build`
   (**`deploy-web.sh` kullanılmaz**). `serve.json` değişmediği için web
   servisini yeniden başlatmak gerekmez.
2. APK kurulmaz. (`offline.html` değişikliği bir sonraki native derlemede
   APK'ya girer; o zamana dek eski `offline.html` kalır.)
3. Geri alma: önceki `dist` derlemesi (git'teki önceki commit'ten
   `npm run build`).

## Kapsam Dışı

- Flutter'da olup React'te olmayan davranışlar: hedef kartında cihazdaki
  kaydı dinleme/silme, "Yenile" düğmesi, "Fine-Tune Başlat",
  hiperparametre kaydırıcıları. İstenirse ayrı iş.
- Açık tema, Flutter uygulamasının kendisi, sunucu değişikliği.
- `worktree-react-rewrite` dalını `master`'a merge etmek.
