# Ayarlar Sayfasını Kilitleme (admin/adminpassword)

**Tarih:** 2026-08-09
**Kapsam:** `lib/main.dart` — yalnızca `_MainPageState` (`_openAyarlar`) ve yeni bir `_AyarlarLoginPage`. `_AyarlarPage`, `_FurkancaPage`, `_KayitPage`, backend değişmiyor.

## Sorun

Ayarlar sayfası (model tercihi, fine-tune tetikleme gibi hassas ayarları içeriyor) şu an herkes tarafından açılabiliyor. Sadece `admin` / `adminpassword` bilgileriyle erişilebilir olması isteniyor.

## Tasarım

- `_MainPageState`'e bir oturum bayrağı eklenir: `bool _ayarlarUnlocked = false`.
- `_openAyarlar()`:
  - `_ayarlarUnlocked == true` ise doğrudan `Navigator.push(_AyarlarPage)` (mevcut davranış).
  - Değilse önce yeni `_AyarlarLoginPage`'e gidilir.
- **`_AyarlarLoginPage`** (yeni `StatefulWidget`, `_MainPage`'in aynı bölgesinde tanımlanır): kullanıcı adı ve şifre (`obscureText: true`) alanları + "Giriş Yap" butonu. Girilen değerler sabit kodlanmış `_kAdminUsername = 'admin'` / `_kAdminPassword = 'adminpassword'` (dosyanın üst kısmında, `kUserId`/`kBaseUrl` yanında tanımlı sabitler) ile karşılaştırılır.
  - Doğruysa: bir callback aracılığıyla `_MainPageState._ayarlarUnlocked = true` yapılır, sonra `Navigator.pushReplacement(_AyarlarPage)` ile giriş sayfası `_AyarlarPage` ile değiştirilir (geri tuşuna basınca tekrar giriş ekranına dönülmesin diye replacement kullanılır).
  - Yanlışsa: kısa bir hata metni gösterilir ("Kullanıcı adı veya şifre hatalı"), alanlar temizlenmez, tekrar denenebilir. Deneme sınırı/kilitlenme yok.
- Oturum kalıcılığı: `_ayarlarUnlocked` yalnızca bellekte (`_MainPageState` alanı) tutulur — uygulama tamamen kapatılıp yeniden açılınca sıfırlanır, yeniden giriş istenir. Aynı oturumda Ayarlar'dan çıkıp tekrar girildiğinde şifre sorulmaz.

## Kapsam Dışı

- Sunucu tarafı kimlik doğrulama / backend değişikliği — tamamen istemci tarafı bir kilit.
- Şifre değiştirme, çoklu kullanıcı, "beni hatırla" (uygulama yeniden açılınca hatırlama) — istenmedi.
- Deneme sınırlama / hesap kilitleme — kişisel kullanım için gereksiz.

## Test Planı

- Manuel: Ayarlar'a ilk basışta giriş ekranı açılmalı; yanlış bilgiyle hata gösterilmeli; doğru bilgiyle (`admin`/`adminpassword`) Ayarlar sayfası açılmalı; Ayarlar'dan geri dönüp tekrar Ayarlar'a basınca (aynı oturumda) giriş ekranı **çıkmamalı**; uygulama yeniden başlatılınca giriş ekranı tekrar çıkmalı.
- Otomatik: Furkanca/Kayıt sayfalarına ait mevcut testler (`panel_navigation_test.dart`, `vad_test.dart`, `pending_recordings_test.dart`) etkilenmemeli, aynen geçmeli. Giriş formunun kendisi için widget testi eklemek platform bağımlılığı gerektirmiyor (saf UI + string karşılaştırma) — basit bir widget testiyle doğru/yanlış giriş senaryoları kapsanabilir.
