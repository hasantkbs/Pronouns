# Panelleri Bottom Sheet'ten Tam Sayfaya Çevirme

**Tarih:** 2026-08-09
**Kapsam:** `lib/main.dart` — Ana ekrandaki 3 buton (Furkanca, Kayıt, Ayarlar)

## Sorun

Ana sayfadaki `Furkanca`, `Kayıt`, `Ayarlar` butonları şu anda `showModalBottomSheet(...)` ile açılan popup panelleri gösteriyor (`_FurkancaPanel`, `_KayitPanel`, `_AyarlarPanel`, ortak kabuk: `_Sheet`). İstenen: bunların popup değil, tam sayfa (standart Flutter navigasyonu) olarak açılması.

## Tasarım

Tek yaklaşım: `showModalBottomSheet` çağrılarını `Navigator.push(MaterialPageRoute(...))` ile değiştirmek.

- `_FurkancaPanel`, `_KayitPanel`, `_AyarlarPanel` widget'ları `_FurkancaPage`, `_KayitPage`, `_AyarlarPage` olarak yeniden adlandırılır ve her biri `_Sheet` yerine kendi `Scaffold` + standart `AppBar`'ını (başlık + otomatik geri oku) döndürür.
- İç state/mantık (HTTP çağrıları, kayıt, oynatma, form state) değişmez — sadece dış kabuk (`_Sheet` bottom-sheet gövdesi → `Scaffold`/`AppBar`) değişir.
- `MainPage._openFurkanca()`, `_openKayit()`, `_openAyarlar()` fonksiyonları `showModalBottomSheet` yerine `Navigator.push(MaterialPageRoute(builder: (_) => const _XxxPage()))` çağırır.
- `_Sheet` widget'ı artık hiçbir yerde kullanılmadığı için kaldırılır.
- Geri dönüş: standart AppBar geri oku + Android sistem geri tuşu / iOS kaydırma jesti (ekstra kod gerekmez, `MaterialPageRoute`'un varsayılan davranışı).

## Kapsam Dışı

- Route isimlendirme / named routes, deep linking — gerekmiyor, üç sayfa da doğrudan push ediliyor.
- Panel içi mantık, API çağrıları, tasarım (buton renkleri, layout) değişmiyor.
- Diğer dosyalar (`android/`, `ios/`, backend) etkilenmiyor.

## Test Planı

- `flutter analyze` ile statik hata kontrolü.
- Uygulamayı çalıştırıp üç butona da basarak: sayfanın tam ekran açıldığını, AppBar'da geri okunun göründüğünü, geri tuşuyla ana sayfaya dönüldüğünü ve panel içi işlevlerin (kayıt, yükleme, ayar kaydetme) eskisi gibi çalıştığını doğrulama.
