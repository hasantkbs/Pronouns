# Ayarlar Sayfasını Kilitleme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ayarlar sayfası yalnızca `admin`/`adminpassword` girildikten sonra açılabilsin; oturum boyunca (uygulama kapanana kadar) tekrar sorulmasın.

**Architecture:** `_MainPageState`'e bellekte tutulan bir `_ayarlarUnlocked` bayrağı eklenir. `_openAyarlar()` bu bayrağa göre ya doğrudan `_AyarlarPage`'e ya da yeni bir `_AyarlarLoginPage`'e yönlendirir. Giriş başarılı olunca bayrak `true` yapılır ve `Navigator.pushReplacement` ile `_AyarlarPage`'e geçilir. Backend'e hiçbir dokunuş yok.

**Tech Stack:** Flutter/Dart, `flutter_test`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-09-ayarlar-login-lock-design.md`.
- Kimlik doğrulama tamamen istemci tarafında, sabit kodlanmış (`admin`/`adminpassword`) — sunucuya hiçbir istek gitmez, backend değişmez.
- `_AyarlarPage`, `_FurkancaPage`, `_KayitPage` içerikleri değişmez.
- Deneme sınırlama/kilitlenme yok; oturum kalıcılığı yalnızca bellekte (`_MainPageState` alanı), uygulama yeniden başlayınca sıfırlanır.

## Test Kapsamı Notu

Spec hem doğru hem yanlış giriş senaryolarının otomatik test edilebileceğini varsaymıştı. Uygulama sırasında şu kısıtlar netleşti: (1) `_AyarlarLoginPage` diğer sayfalar gibi `main.dart` içinde **private** bir sınıf — ayrı bir test dosyasından doğrudan izole test edilemez, yalnızca `PronounsApp` üzerinden `find.text(...)` gibi genel bulucularla dolaylı test edilebilir (mevcut `panel_navigation_test.dart` ile aynı desen). (2) Doğru giriş sonrası açılan `_AyarlarPage.initState()` gerçek ağ çağrıları (`GET /settings`, `GET /model/info`) yapıyor; bu çağrılar `mounted` kontrolü olmadan `setState` çağırıyor (mevcut, bu plana ait olmayan bir durum) — testte bu sayfaya geçişi tetiklemek, test bitip widget dispose olduktan sonra gecikmeli ağ hatasının başka bir testi etkilemesi riskini taşıyor. Bu yüzden **yalnızca "yanlış giriş" senaryosu** otomatik test edilecek (hiç ağ çağrısı tetiklemiyor); "doğru giriş" senaryosu manuel doğrulanacak.

---

### Task 1: Giriş kilidini ekle

**Files:**
- Modify: `lib/main.dart` (üst sabitler; `_MainPageState._openAyarlar` ve alanları; yeni `_AyarlarLoginPage`/`_AyarlarLoginPageState` eklenir)
- Create: `test/ayarlar_login_test.dart`

**Interfaces:**
- Consumes: mevcut `_AyarlarPage` (değişmeden, sadece hedef route olarak kullanılır), `PronounsApp` (public, test için).
- Produces: yok (bu son özellik, başka bir task ona bağlı değil).

- [ ] **Step 1: Sabit kimlik bilgilerini ekle**

`lib/main.dart` başında şu satırları bul:

```dart
const String kBaseUrl = 'http://10.10.108.10:8001';
const String kUserId = 'FurkanV1';
```

Şununla değiştir:

```dart
const String kBaseUrl = 'http://10.10.108.10:8001';
const String kUserId = 'FurkanV1';
const String _kAdminUsername = 'admin';
const String _kAdminPassword = 'adminpassword';
```

- [ ] **Step 2: `_MainPageState`'e oturum bayrağı ekle**

Şu bloğu bul:

```dart
class _MainPageState extends State<MainPage>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse;
```

Şununla değiştir:

```dart
class _MainPageState extends State<MainPage>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse;
  bool _ayarlarUnlocked = false;
```

- [ ] **Step 3: `_openAyarlar()`'ı güncelle**

Şu bloğu bul:

```dart
  void _openAyarlar() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const _AyarlarPage()),
    );
  }
```

Şununla değiştir:

```dart
  void _openAyarlar() {
    if (_ayarlarUnlocked) {
      Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => const _AyarlarPage()),
      );
      return;
    }
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => _AyarlarLoginPage(
          onSuccess: () => setState(() => _ayarlarUnlocked = true),
        ),
      ),
    );
  }
```

- [ ] **Step 4: `_AyarlarLoginPage`'i ekle**

`lib/main.dart` içinde `// 3) AYARLAR PANELİ` yorum satırıyla başlayan bölümü bul (`class _AyarlarPage extends StatefulWidget {` ile devam eder). Bu yorum bloğunun **hemen üstüne**, aşağıdaki yeni sınıfları ekle (mevcut `// 3) AYARLAR PANELİ` bölümüne dokunma):

```dart
// ════════════════════════════════════════════════════════════════════════════
// 2c) AYARLAR GİRİŞİ
// ════════════════════════════════════════════════════════════════════════════
class _AyarlarLoginPage extends StatefulWidget {
  final VoidCallback onSuccess;
  const _AyarlarLoginPage({required this.onSuccess});

  @override
  State<_AyarlarLoginPage> createState() => _AyarlarLoginPageState();
}

class _AyarlarLoginPageState extends State<_AyarlarLoginPage> {
  final _userController = TextEditingController();
  final _passController = TextEditingController();
  String? _error;

  @override
  void dispose() {
    _userController.dispose();
    _passController.dispose();
    super.dispose();
  }

  void _submit() {
    if (_userController.text == _kAdminUsername &&
        _passController.text == _kAdminPassword) {
      widget.onSuccess();
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const _AyarlarPage()),
      );
      return;
    }
    setState(() => _error = 'Kullanıcı adı veya şifre hatalı');
  }

  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFFF59E0B);
    return Scaffold(
      appBar: AppBar(title: const Text('Ayarlar Girişi')),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.lock_outline_rounded, size: 48, color: accent),
            const SizedBox(height: 16),
            TextField(
              controller: _userController,
              decoration: const InputDecoration(
                labelText: 'Kullanıcı Adı',
                prefixIcon: Icon(Icons.person_outline_rounded),
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _passController,
              obscureText: true,
              decoration: const InputDecoration(
                labelText: 'Şifre',
                prefixIcon: Icon(Icons.key_outlined),
                border: OutlineInputBorder(),
              ),
              onSubmitted: (_) => _submit(),
            ),
            if (_error != null) ...[
              const SizedBox(height: 12),
              Text(_error!,
                  style: const TextStyle(color: Colors.red, fontSize: 13)),
            ],
            const SizedBox(height: 20),
            FilledButton.icon(
              style: FilledButton.styleFrom(
                backgroundColor: accent,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
              ),
              onPressed: _submit,
              icon: const Icon(Icons.login_rounded),
              label: const Text('Giriş Yap',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
            ),
          ],
        ),
      ),
    );
  }
}

```

- [ ] **Step 5: Başarısız olacak testi yaz**

`test/ayarlar_login_test.dart` dosyasını oluştur:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pronouns_ai/main.dart';

void main() {
  testWidgets('Ayarlar butonu şifre olmadan doğrudan açılmaz, yanlış girişte hata gösterir',
      (WidgetTester tester) async {
    // MainPage'deki sonsuz "pulse" animasyonu pumpAndSettle() ile takılır;
    // sabit pump() adımları kullanılıyor (bkz. panel_navigation_test.dart).
    await tester.binding.setSurfaceSize(const Size(400, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    await tester.pumpWidget(const PronounsApp());
    await tester.pump();

    await tester.tap(find.text('Ayarlar'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 300));

    // Doğrudan Ayarlar sayfasına değil, giriş ekranına gitmeli.
    expect(find.text('Ayarlar Girişi'), findsOneWidget);
    expect(find.text('Model Ayarları'), findsNothing);

    // Yanlış kullanıcı adı/şifre.
    await tester.enterText(find.byType(TextField).first, 'admin');
    await tester.enterText(find.byType(TextField).last, 'yanlis-sifre');
    await tester.tap(find.text('Giriş Yap'));
    await tester.pump();

    expect(find.text('Kullanıcı adı veya şifre hatalı'), findsOneWidget);
    // Hâlâ giriş ekranındayız — Ayarlar sayfasına geçmedi.
    expect(find.text('Ayarlar Girişi'), findsOneWidget);
  });
}
```

> Not: Doğru giriş senaryosu (`admin`/`adminpassword`) burada otomatik test edilmiyor — yukarıdaki "Test Kapsamı Notu"na bakın. Adım 9'da manuel doğrulanacak.

- [ ] **Step 6: Testin şu an başarısız olduğunu doğrula**

Run: `flutter test test/ayarlar_login_test.dart`
Expected: FAIL — `_AyarlarLoginPage`/`_kAdminUsername`/`_kAdminPassword` henüz tanımlı değilse derleme hatası; tanımlıysa ama `_openAyarlar()` hâlâ eski haliyse `find.text('Ayarlar Girişi')` bulunamaz.

(Adım 1-4'ü henüz uygulamadıysanız bu adım doğal olarak başarısız olur — TDD sırasını Adım 1-4'ü Adım 5-6'dan önce uygulamak isterseniz plan sırasını kendi takdirinizle ayarlayabilirsiniz; önemli olan testin implementasyondan önce en az bir kez başarısız görülmesidir.)

- [ ] **Step 7: Testin geçtiğini doğrula**

Run: `flutter test test/ayarlar_login_test.dart`
Expected: PASS

- [ ] **Step 8: Statik analiz ve diğer testlerin etkilenmediğini doğrula**

Run: `flutter analyze lib/main.dart`
Expected: Yeni hata yok (var olan `withOpacity`/`curly_braces` info uyarıları kalabilir).

Run: `flutter test test/panel_navigation_test.dart test/pending_recordings_test.dart test/vad_test.dart test/ayarlar_login_test.dart`
Expected: Tümü geçer.

- [ ] **Step 9: Commit**

```bash
git add lib/main.dart test/ayarlar_login_test.dart
git commit -m "feat: lock Ayarlar page behind admin/adminpassword login"
```

- [ ] **Step 10: Manuel doğrulama (cihazda)**

```bash
flutter run
```

1. "Ayarlar"a bas — giriş ekranı açılmalı.
2. Yanlış bilgiyle "Giriş Yap"a bas — hata mesajı görünmeli, giriş ekranında kalmalı.
3. Doğru bilgiyle (`admin` / `adminpassword`) "Giriş Yap"a bas — Ayarlar sayfası açılmalı.
4. Geri dönüp tekrar "Ayarlar"a bas — **giriş ekranı çıkmamalı**, doğrudan Ayarlar açılmalı (oturum hatırlanıyor).
5. Uygulamayı tamamen kapatıp yeniden aç, "Ayarlar"a bas — giriş ekranı **tekrar çıkmalı**.

## Self-Review Notları

- **Spec kapsaması:** Spec'in tüm maddeleri (sabit kimlik, oturum bayrağı, yanlış girişte hata, doğru girişte `pushReplacement`, oturum kalıcılığı yalnızca bellekte) Task 1'de karşılanıyor.
- **Placeholder taraması:** Yok.
- **Test kapsamı sapması:** Spec "doğru ve yanlış giriş" ikisinin de otomatik test edilebileceğini varsaymıştı; implementasyon sırasında (private sınıf + `_AyarlarPage`'in ağ çağrılı `initState`'i) yalnızca yanlış-giriş senaryosunun güvenle otomatik test edilebildiği netleşti — bu sapma yukarıda "Test Kapsamı Notu" ile açıkça belgelendi, sessizce atlanmadı.
