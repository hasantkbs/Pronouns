# Panelleri Tam Sayfaya Çevirme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Furkanca`, `Kayıt`, `Ayarlar` butonlarıyla açılan bottom-sheet panellerini, geri oku olan standart tam sayfalara çevirmek.

**Architecture:** `lib/main.dart` içindeki `showModalBottomSheet(...)` çağrıları `Navigator.of(context).push(MaterialPageRoute(...))` ile değiştirilir. Her panel widget'ı (`_FurkancaPanel`, `_KayitPanel`, `_AyarlarPanel`) `...Page` olarak yeniden adlandırılır ve ortak `_Sheet` kabuğu yerine kendi `Scaffold(appBar: AppBar(...), body: SingleChildScrollView(...))` yapısını kullanır. İç state/mantık dokunulmadan kalır. Son adımda artık kullanılmayan `_Sheet` widget'ı silinir.

**Tech Stack:** Flutter/Dart, `flutter_test` (widget test).

## Global Constraints

- Named route / route ismi kullanılmayacak — sayfalar doğrudan `MaterialPageRoute` ile push edilecek (bkz. spec: `docs/superpowers/specs/2026-08-09-panels-to-pages-design.md`).
- Panel içi state, HTTP çağrıları, form mantığı, buton renkleri/layout değişmeyecek — sadece dış kabuk (Sheet → Scaffold/AppBar) değişecek.
- `android/`, `ios/`, backend (`api.py` vb.) dosyalarına dokunulmayacak.
- Geri navigasyon için ekstra kod yazılmayacak — `MaterialPageRoute`'un varsayılan AppBar geri oku / sistem geri tuşu davranışı kullanılacak.

---

### Task 1: Furkanca panelini sayfaya çevir + navigasyon testi ekle

**Files:**
- Create: `test/panel_navigation_test.dart`
- Modify: `lib/main.dart` (`_MainPageState._openFurkanca`, `_FurkancaPanel` class ve `_FurkancaPanelState.build`)

**Interfaces:**
- Consumes: mevcut `PronounsApp` (public, `lib/main.dart`'ta zaten tanımlı), mevcut `_BigButton` (metin etiketi `'Furkanca'`).
- Produces: `_FurkancaPage` / `_FurkancaPageState` sınıfları (Task 4'te `_Sheet` kaldırılırken hiçbir yerde referans kalmadığını doğrulamak için kullanılacak isimler).

- [ ] **Step 1: Başarısız olacak testi yaz**

`test/panel_navigation_test.dart` dosyasını oluştur:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pronouns_ai/main.dart';

void main() {
  testWidgets('Furkanca button opens a full page, not a bottom sheet',
      (WidgetTester tester) async {
    await tester.pumpWidget(const PronounsApp());
    await tester.pumpAndSettle();

    await tester.tap(find.text('Furkanca'));
    await tester.pumpAndSettle();

    // Tam sayfa push edildiğinde geri oklu bir AppBar görünür.
    expect(find.byType(AppBar), findsOneWidget);
    expect(find.byTooltip('Back'), findsOneWidget);
    // Modal bottom sheet olsaydı BottomSheet widget'ı bulunurdu.
    expect(find.byType(BottomSheet), findsNothing);
  });
}
```

- [ ] **Step 2: Testin şu an başarısız olduğunu doğrula**

Run: `flutter test test/panel_navigation_test.dart`
Expected: FAIL — `find.byType(AppBar)` `findsNothing` döner çünkü şu an `Furkanca` butonu bir `showModalBottomSheet` açıyor, `AppBar` yok.

- [ ] **Step 3: `_openFurkanca` metodunu güncelle**

`lib/main.dart` içinde şu bloğu bul:

```dart
  void _openFurkanca() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => const _FurkancaPanel(),
    );
  }
```

Şununla değiştir:

```dart
  void _openFurkanca() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const _FurkancaPage()),
    );
  }
```

- [ ] **Step 4: `_FurkancaPanel` sınıf tanımını yeniden adlandır**

Şu bloğu bul:

```dart
class _FurkancaPanel extends StatefulWidget {
  const _FurkancaPanel();

  @override
  State<_FurkancaPanel> createState() => _FurkancaPanelState();
}

class _FurkancaPanelState extends State<_FurkancaPanel> {
```

Şununla değiştir:

```dart
class _FurkancaPage extends StatefulWidget {
  const _FurkancaPage();

  @override
  State<_FurkancaPage> createState() => _FurkancaPageState();
}

class _FurkancaPageState extends State<_FurkancaPage> {
```

- [ ] **Step 5: `build()` metodunun açılışını `Scaffold`'a çevir**

Şu bloğu bul:

```dart
  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFF6C63FF);

    return _Sheet(
      title: 'Furkanca',
      icon: Icons.record_voice_over_rounded,
      iconColor: accent,
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
```

Şununla değiştir:

```dart
  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFF6C63FF);

    return Scaffold(
      appBar: AppBar(title: const Text('Furkanca')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
```

- [ ] **Step 6: `build()` metodunun kapanışına bir `)` daha ekle**

Şu bloğu bul (dosyanın sonunda değil, `_FurkancaPageState.build` metodunun sonunda — `_missing.isNotEmpty` satırından hemen sonra gelir):

```dart
          if (_missing.isNotEmpty)
            _ResultTile(
              label: 'Eksik Kelimeler',
              value: _missing.join(', '),
              color: Colors.orange,
            ),
        ],
      ),
    );
  }
}
```

Şununla değiştir (Step 5'te eklenen `SingleChildScrollView` için bir kapanış parantezi eklendi):

```dart
          if (_missing.isNotEmpty)
            _ResultTile(
              label: 'Eksik Kelimeler',
              value: _missing.join(', '),
              color: Colors.orange,
            ),
        ],
      ),
      ),
    );
  }
}
```

- [ ] **Step 7: Testin artık geçtiğini doğrula**

Run: `flutter test test/panel_navigation_test.dart`
Expected: PASS

- [ ] **Step 8: Statik analiz**

Run: `flutter analyze lib/main.dart`
Expected: `_FurkancaPanel`/`_FurkancaPanelState` için "yok" hatası çıkmaz (formatlama/indent uyarıları olabilir, Task 4'te `dart format` ile düzeltilecek — hata (error) olmamalı).

- [ ] **Step 9: Commit**

```bash
git add lib/main.dart test/panel_navigation_test.dart
git commit -m "feat: open Furkanca panel as a full page instead of a bottom sheet"
```

---

### Task 2: Kayıt panelini sayfaya çevir

**Files:**
- Modify: `lib/main.dart` (`_MainPageState._openKayit`, `_KayitPanel` class ve `_KayitPanelState.build`)

**Interfaces:**
- Consumes: Task 1'de kanıtlanmış dönüşüm deseni (Scaffold/AppBar/SingleChildScrollView). Bu panel `initState`'te ağ çağrısı yaptığı için (`_loadSets`, `_refreshWord`, `_refreshProgress`) otomatik widget testi eklenmiyor — gerçek sunucu olmadan flaky olur. Doğrulama `flutter analyze` + spec'teki manuel test planıyla yapılır.
- Produces: `_KayitPage` / `_KayitPageState` sınıfları.

- [ ] **Step 1: `_openKayit` metodunu güncelle**

Şu bloğu bul:

```dart
  void _openKayit() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => const _KayitPanel(),
    );
  }
```

Şununla değiştir:

```dart
  void _openKayit() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const _KayitPage()),
    );
  }
```

- [ ] **Step 2: `_KayitPanel` sınıf tanımını yeniden adlandır**

Şu bloğu bul:

```dart
class _KayitPanel extends StatefulWidget {
  const _KayitPanel();

  @override
  State<_KayitPanel> createState() => _KayitPanelState();
}

class _KayitPanelState extends State<_KayitPanel> {
```

Şununla değiştir:

```dart
class _KayitPage extends StatefulWidget {
  const _KayitPage();

  @override
  State<_KayitPage> createState() => _KayitPageState();
}

class _KayitPageState extends State<_KayitPage> {
```

- [ ] **Step 3: `build()` metodunun açılışını `Scaffold`'a çevir**

Şu bloğu bul:

```dart
  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFF10B981);
    final cs = Theme.of(context).colorScheme;

    final allDone = _word == null;
    final progress = _totalWords > 0 ? _completedWords / _totalWords : 0.0;

    return _Sheet(
      title: 'Kayıt',
      icon: Icons.mic_rounded,
      iconColor: accent,
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
```

Şununla değiştir:

```dart
  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFF10B981);
    final cs = Theme.of(context).colorScheme;

    final allDone = _word == null;
    final progress = _totalWords > 0 ? _completedWords / _totalWords : 0.0;

    return Scaffold(
      appBar: AppBar(title: const Text('Kayıt')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
```

- [ ] **Step 4: `build()` metodunun kapanışına bir `)` daha ekle**

Şu bloğu bul (`_KayitPageState.build` metodunun sonu — "Yenile" outlined butonundan hemen sonra):

```dart
          const SizedBox(height: 10),
          OutlinedButton.icon(
            onPressed: _busy
                ? null
                : () => Future.wait([_refreshWord(), _refreshProgress()]),
            icon: const Icon(Icons.refresh_rounded),
            label: const Text('Yenile'),
          ),

          if (_lastStatus != null) ...[
            const SizedBox(height: 16),
            _ResultTile(label: 'Durum', value: _lastStatus!, color: accent),
          ],
        ],
      ),
    );
  }
}
```

Şununla değiştir:

```dart
          const SizedBox(height: 10),
          OutlinedButton.icon(
            onPressed: _busy
                ? null
                : () => Future.wait([_refreshWord(), _refreshProgress()]),
            icon: const Icon(Icons.refresh_rounded),
            label: const Text('Yenile'),
          ),

          if (_lastStatus != null) ...[
            const SizedBox(height: 16),
            _ResultTile(label: 'Durum', value: _lastStatus!, color: accent),
          ],
        ],
      ),
      ),
    );
  }
}
```

- [ ] **Step 5: Statik analiz**

Run: `flutter analyze lib/main.dart`
Expected: `_KayitPanel`/`_KayitPanelState` için hata çıkmaz.

- [ ] **Step 6: Commit**

```bash
git add lib/main.dart
git commit -m "feat: open Kayıt panel as a full page instead of a bottom sheet"
```

---

### Task 3: Ayarlar panelini sayfaya çevir

**Files:**
- Modify: `lib/main.dart` (`_MainPageState._openAyarlar`, `_AyarlarPanel` class ve `_AyarlarPanelState.build`)

**Interfaces:**
- Consumes: Task 1'de kanıtlanmış dönüşüm deseni. Bu panel de `initState`'te ağ çağrısı yaptığı için (`_fetchSettings`) otomatik widget testi eklenmiyor.
- Produces: `_AyarlarPage` / `_AyarlarPageState` sınıfları.

- [ ] **Step 1: `_openAyarlar` metodunu güncelle**

Şu bloğu bul:

```dart
  void _openAyarlar() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => const _AyarlarPanel(),
    );
  }
```

Şununla değiştir:

```dart
  void _openAyarlar() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const _AyarlarPage()),
    );
  }
```

- [ ] **Step 2: `_AyarlarPanel` sınıf tanımını yeniden adlandır**

Şu bloğu bul:

```dart
class _AyarlarPanel extends StatefulWidget {
  const _AyarlarPanel();

  @override
  State<_AyarlarPanel> createState() => _AyarlarPanelState();
}

class _AyarlarPanelState extends State<_AyarlarPanel> {
```

Şununla değiştir:

```dart
class _AyarlarPage extends StatefulWidget {
  const _AyarlarPage();

  @override
  State<_AyarlarPage> createState() => _AyarlarPageState();
}

class _AyarlarPageState extends State<_AyarlarPage> {
```

- [ ] **Step 3: `build()` metodunun açılışını `Scaffold`'a çevir**

Şu bloğu bul (dikkat: `body` burada `_loading ? Center(...) : Column(...)` şeklinde bir ternary):

```dart
  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFFF59E0B);
    final cs = Theme.of(context).colorScheme;

    return _Sheet(
      title: 'Model Ayarları',
      icon: Icons.tune_rounded,
      iconColor: accent,
      body: _loading
          ? const Center(
              child: Padding(
                  padding: EdgeInsets.all(32),
                  child: CircularProgressIndicator()))
          : Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
```

Şununla değiştir:

```dart
  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFFF59E0B);
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(title: const Text('Model Ayarları')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: _loading
            ? const Center(
                child: Padding(
                    padding: EdgeInsets.all(32),
                    child: CircularProgressIndicator()))
            : Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
```

- [ ] **Step 4: `build()` metodunun kapanışına bir `)` daha ekle**

Şu bloğu bul (`_AyarlarPageState.build` metodunun sonu — "Fine-Tune Başlat" butonundan hemen sonra):

```dart
                  label: Text(
                    _training ? 'Eğitim başlatıldı...' : 'Fine-Tune Başlat',
                    style: const TextStyle(
                        fontSize: 15, fontWeight: FontWeight.w700),
                  ),
                ),
              ],
            ),
    );
  }
}
```

Şununla değiştir:

```dart
                  label: Text(
                    _training ? 'Eğitim başlatıldı...' : 'Fine-Tune Başlat',
                    style: const TextStyle(
                        fontSize: 15, fontWeight: FontWeight.w700),
                  ),
                ),
              ],
            ),
      ),
    );
  }
}
```

- [ ] **Step 5: Statik analiz**

Run: `flutter analyze lib/main.dart`
Expected: `_AyarlarPanel`/`_AyarlarPanelState` için hata çıkmaz.

- [ ] **Step 6: Commit**

```bash
git add lib/main.dart
git commit -m "feat: open Ayarlar panel as a full page instead of a bottom sheet"
```

---

### Task 4: `_Sheet` widget'ını kaldır, formatla, uçtan uca doğrula

**Files:**
- Modify: `lib/main.dart` (yalnızca `_Sheet` sınıfının silinmesi + `dart format`)

**Interfaces:**
- Consumes: Task 1-3'ün ürettiği `_FurkancaPage`, `_KayitPage`, `_AyarlarPage` — bu görev, üç panelin de artık `_Sheet` kullanmadığını doğrulayarak başlar.
- Produces: yok (temizlik görevi).

- [ ] **Step 1: `_Sheet` sınıfının artık hiç kullanılmadığını doğrula**

Run: `grep -n "_Sheet(" lib/main.dart`
Expected: Hiç sonuç dönmemeli (sadece sınıf tanımının kendisi `class _Sheet extends StatelessWidget {` satırı görünür, çağrı görünmemeli).

- [ ] **Step 2: `_Sheet` sınıfını sil**

`lib/main.dart` içinde şu bloğu bul ve tamamen sil (yorum satırı dahil):

```dart
// ─── Panel şablonu ────────────────────────────────────────────────────────────
class _Sheet extends StatelessWidget {
  final String title;
  final IconData icon;
  final Color iconColor;
  final Widget body;

  const _Sheet({
    required this.title,
    required this.icon,
    required this.iconColor,
    required this.body,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(context).size.height * 0.92,
      ),
      decoration: BoxDecoration(
        color: cs.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Handle
          Center(
            child: Container(
              margin: const EdgeInsets.only(top: 12),
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: cs.onSurface.withOpacity(0.2),
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(24, 16, 24, 0),
            child: Row(
              children: [
                Icon(icon, color: iconColor, size: 26),
                const SizedBox(width: 10),
                Text(title,
                    style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                        color: cs.onSurface)),
              ],
            ),
          ),
          const Divider(height: 24),
          Flexible(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(24, 0, 24, 32),
              child: body,
            ),
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 3: Kodu formatla**

Run: `dart format lib/main.dart`
Expected: Task 1-3'teki manuel düzenlemelerden kalan girinti (indentation) tutarsızlıkları otomatik düzelir.

- [ ] **Step 4: Tam proje analizi**

Run: `flutter analyze`
Expected: `No issues found!` (ya da yalnızca bu değişiklikle ilgisiz, önceden var olan uyarılar — yeni hata olmamalı).

- [ ] **Step 5: Tüm testleri çalıştır**

Run: `flutter test test/panel_navigation_test.dart`
Expected: PASS

> Not: `test/widget_test.dart` bu projeden önce kalma, artık var olmayan `MyApp` sınıfına referans veren varsayılan Flutter sayaç testidir ve bu planın kapsamı dışındadır — dokunulmuyor.

- [ ] **Step 6: Commit**

```bash
git add lib/main.dart
git commit -m "refactor: remove unused _Sheet widget after panel-to-page migration"
```

- [ ] **Step 7: Manuel doğrulama (cihaz/emülatör üzerinde)**

Otomatik testler yalnızca navigasyon davranışını doğrular; gerçek sunucu ile uçtan uca akışı manuel kontrol et (spec'teki test planı):

```bash
flutter run
```

1. Üç butona da bas: `Furkanca`, `Kayıt`, `Ayarlar` — her biri tam ekran açılmalı, üstte geri oku olan bir AppBar görünmeli.
2. Geri okuna bas (veya Android sistem geri tuşu / iOS kaydırma) — ana sayfaya dönmeli.
3. `Kayıt` sayfasında kelime setinin sunucudan geldiğini kontrol et.
4. `Furkanca` sayfasında bir ses kaydı alıp "Düzeltilmiş" metnin geldiğini kontrol et.
5. `Ayarlar` sayfasında ayarları kaydet / Fine-Tune başlat işlevlerinin eskisi gibi çalıştığını kontrol et.

---

## Self-Review Notları

- **Spec kapsaması:** Spec'teki tek gereksinim ("3 buton popup yerine tam sayfa açılsın, standart AppBar ile") Task 1-3'te karşılanıyor; `_Sheet`'in kaldırılması ve manuel test planı spec'in "Test Planı" bölümünü karşılıyor (Task 4).
- **Placeholder taraması:** "TBD"/"TODO" yok; her adımda tam kod veya tam komut var.
- **Tip/isim tutarlılığı:** `_FurkancaPage`/`_KayitPage`/`_AyarlarPage` isimleri Task 1-3 boyunca ve Task 4'ün "Consumes" bölümünde tutarlı kullanıldı.
