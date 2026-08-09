# Kayıt Sayfası: Cihazda Biriktirme + Dinle/Düzenle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kayıt sayfasında alınan tekrarlar artık anında sunucuya gitmesin; cihazda birikip dinlenip silinebilsin, kelimeler arasında yükleme beklemeden serbestçe geçilebilsin, ve istenen an topluca yüklenebilsin.

**Architecture:** `api.py`'de `GET /collect/next-word`'e minimal, geriye uyumlu bir `exclude` parametresi eklenir. Mobil tarafta yeni bir `lib/pending_recordings.dart` modülü (`PendingRecordingsStore`) bekleyen kayıtları kalıcı depoda (uygulama belgeler dizini) ve bir JSON manifest'te yönetir. `_KayitPageState` artık kayıt sonrası `/record`'a POST etmek yerine bu store'a ekler; yeni bir `_PendingUploadsPage` tüm bekleyen kayıtları listeleyip toplu yükleme yapar.

**Tech Stack:** Flutter/Dart (`path_provider`, `audioplayers` `DeviceFileSource`), Python/FastAPI.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-09-local-recording-review-design.md`.
- `POST /record` sözleşmesi değişmiyor (zaten `rep`'i kendisi hesaplıyor — bkz. spec'teki Ön Bulgu).
- `GET /collect/progress`, `_FurkancaPage`, `_AyarlarPage`, kayıt mekanizması (`_startRecording`/`_stopRecording`, eş-zamanlı geri sayım) değişmiyor.
- Ses kırpma, kelime yeniden atama, çoklu cihaz senkronizasyonu kapsam dışı.

---

### Task 1: Backend — `/collect/next-word`'e `exclude` parametresi

**Files:**
- Modify: `api.py`

**Interfaces:**
- Produces: `GET /collect/next-word?...&exclude=kelime1,kelime2` (opsiyonel, varsayılan boş — mevcut davranışla tam geriye uyumlu). Task 3 mobil tarafta bu parametreyi kullanacak.

- [ ] **Step 1: `typing` importuna `Set` ekle**

`api.py` başında şu satırı bul:

```python
from typing import List, Optional, Dict, Any
```

Şununla değiştir:

```python
from typing import List, Optional, Dict, Any, Set
```

- [ ] **Step 2: `_get_next_word_and_rep`'e `exclude` parametresi ekle**

Şu fonksiyonu bul:

```python
def _get_next_word_and_rep(user_id: str, set_file: str) -> Dict[str, Any]:
    """
    collect_data.py'deki resume/eksik kayıt mantığının API versiyonu:
    - Setteki kelimeleri sırayla gezer
    - metadata_words.csv'den o kelime için kaç kayıt var bakar
    - IDEAL_REPETITIONS tamamlanmamışsa o kelimeyi + sıradaki rep'i döndürür
    """
    words = _read_word_set_file(set_file)
    details = _get_recorded_details_words(user_id)
    ideal = int(config.IDEAL_REPETITIONS)

    for w in words:
        current = int(details.get(w, 0))
        if current < ideal:
            return {
                "done": False,
                "word": w,
                "rep": current + 1,
                "current_count": current,
                "ideal_repetitions": ideal,
                "set_file": set_file,
                "total_words": len(words),
            }

    return {
        "done": True,
        "word": None,
        "rep": None,
        "ideal_repetitions": ideal,
        "set_file": set_file,
        "total_words": len(words),
    }
```

Şununla değiştir:

```python
def _get_next_word_and_rep(user_id: str, set_file: str, exclude: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    collect_data.py'deki resume/eksik kayıt mantığının API versiyonu:
    - Setteki kelimeleri sırayla gezer
    - metadata_words.csv'den o kelime için kaç kayıt var bakar
    - IDEAL_REPETITIONS tamamlanmamışsa o kelimeyi + sıradaki rep'i döndürür
    - [exclude] içindeki kelimeler atlanır (mobil tarafta cihazda bekleyen
      kayıtları olan kelimeler için kullanılır; kullanıcı yüklemeyi
      beklemeden bir sonraki kelimeye geçebilsin diye)
    """
    words = _read_word_set_file(set_file)
    details = _get_recorded_details_words(user_id)
    ideal = int(config.IDEAL_REPETITIONS)
    exclude = exclude or set()

    for w in words:
        if w in exclude:
            continue
        current = int(details.get(w, 0))
        if current < ideal:
            return {
                "done": False,
                "word": w,
                "rep": current + 1,
                "current_count": current,
                "ideal_repetitions": ideal,
                "set_file": set_file,
                "total_words": len(words),
            }

    return {
        "done": True,
        "word": None,
        "rep": None,
        "ideal_repetitions": ideal,
        "set_file": set_file,
        "total_words": len(words),
    }
```

- [ ] **Step 3: `/collect/next-word` endpoint'ine `exclude` parametresi ekle**

Şu bloğu bul:

```python
@app.get("/collect/next-word")
async def get_next_word(user_id: str, set_file: str = "wordSet.txt"):
    """
    Mobil tarafın 'collect_data.py' akışına benzer biçimde sıradaki kelimeyi alması için.
    """
    try:
        return _get_next_word_and_rep(user_id=user_id, set_file=set_file)
    except Exception as e:
        return {"error": str(e)}
```

Şununla değiştir:

```python
@app.get("/collect/next-word")
async def get_next_word(user_id: str, set_file: str = "wordSet.txt", exclude: str = ""):
    """
    Mobil tarafın 'collect_data.py' akışına benzer biçimde sıradaki kelimeyi alması için.
    `exclude`: virgülle ayrılmış, atlanacak kelimeler (cihazda bekleyen kayıtları
    olan kelimeler için — kullanıcı yüklemeden bir sonraki kelimeye geçebilsin).
    """
    try:
        exclude_set = {w.strip() for w in exclude.split(",") if w.strip()}
        return _get_next_word_and_rep(user_id=user_id, set_file=set_file, exclude=exclude_set)
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Söz dizimi kontrolü**

Run: `python3 -m py_compile api.py`
Expected: Hata çıktısı yok (sessizce başarılı biter).

> Not: `api.py`'yi gerçekten import edip çalıştırmak ağır ML bağımlılıkları (ASR modeli vb.) yüklediği için burada otomatik bir unit test kurmuyoruz — bu proje ölçeğinde orantısız olur. Doğrulama Task 4'te sunucu ayağa kaldırılıp `curl` ile yapılacak.

- [ ] **Step 5: Commit**

```bash
git add api.py
git commit -m "feat(api): add optional exclude param to /collect/next-word"
```

---

### Task 2: `PendingRecordingsStore` modülü (TDD)

**Files:**
- Create: `lib/pending_recordings.dart`
- Create: `test/pending_recordings_test.dart`

**Interfaces:**
- Produces: `class PendingTake {final String word; final String filePath; final DateTime recordedAt;}`, `class PendingRecordingsStore {PendingRecordingsStore(Directory baseDir); Future<void> load(); List<PendingTake> get takes; List<PendingTake> forWord(String word); Set<String> get pendingWords; Future<PendingTake> add(String word, File audioFile); Future<void> remove(PendingTake take);}` — Task 3 bu API'yi doğrudan kullanacak.

- [ ] **Step 1: Başarısız olacak testleri yaz**

`test/pending_recordings_test.dart` dosyasını oluştur:

```dart
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:pronouns_ai/pending_recordings.dart';

void main() {
  late Directory tempDir;

  setUp(() {
    tempDir = Directory.systemTemp.createTempSync('pending_recordings_test');
  });

  tearDown(() {
    if (tempDir.existsSync()) tempDir.deleteSync(recursive: true);
  });

  File writeFakeAudio(String name) {
    final f = File('${tempDir.path}/$name');
    f.writeAsBytesSync([1, 2, 3, 4]);
    return f;
  }

  test('load() on an empty store returns no takes', () async {
    final store = PendingRecordingsStore(tempDir);
    await store.load();
    expect(store.takes, isEmpty);
  });

  test('add() copies the file and persists it in the manifest', () async {
    final store = PendingRecordingsStore(tempDir);
    await store.load();

    final source = writeFakeAudio('rec1.wav');
    final take = await store.add('elma', source);

    expect(store.takes, hasLength(1));
    expect(store.takes.first.word, 'elma');
    expect(File(take.filePath).existsSync(), isTrue);
    expect(
        File('${tempDir.path}/pending_recordings/manifest.json').existsSync(),
        isTrue);
  });

  test('a new store instance restores takes saved by a previous one',
      () async {
    final store1 = PendingRecordingsStore(tempDir);
    await store1.load();
    await store1.add('elma', writeFakeAudio('rec1.wav'));
    await store1.add('armut', writeFakeAudio('rec2.wav'));

    final store2 = PendingRecordingsStore(tempDir);
    await store2.load();

    expect(store2.takes, hasLength(2));
    expect(store2.pendingWords, {'elma', 'armut'});
  });

  test('remove() deletes the file and updates the manifest', () async {
    final store = PendingRecordingsStore(tempDir);
    await store.load();
    final take1 = await store.add('elma', writeFakeAudio('rec1.wav'));
    await store.add('elma', writeFakeAudio('rec2.wav'));

    await store.remove(take1);

    expect(store.takes, hasLength(1));
    expect(File(take1.filePath).existsSync(), isFalse);

    final store2 = PendingRecordingsStore(tempDir);
    await store2.load();
    expect(store2.takes, hasLength(1));
  });

  test('forWord() filters by word', () async {
    final store = PendingRecordingsStore(tempDir);
    await store.load();
    await store.add('elma', writeFakeAudio('rec1.wav'));
    await store.add('armut', writeFakeAudio('rec2.wav'));
    await store.add('elma', writeFakeAudio('rec3.wav'));

    expect(store.forWord('elma'), hasLength(2));
    expect(store.forWord('armut'), hasLength(1));
    expect(store.forWord('kayısı'), isEmpty);
  });
}
```

- [ ] **Step 2: Testlerin şu an derlenmediğini doğrula**

Run: `flutter test test/pending_recordings_test.dart`
Expected: FAIL — `package:pronouns_ai/pending_recordings.dart` bulunamıyor.

- [ ] **Step 3: `lib/pending_recordings.dart` modülünü yaz**

```dart
import 'dart:convert';
import 'dart:io';

/// Cihazda birikmiş, henüz sunucuya yüklenmemiş tek bir kayıt.
class PendingTake {
  final String word;
  final String filePath;
  final DateTime recordedAt;

  PendingTake({
    required this.word,
    required this.filePath,
    required this.recordedAt,
  });

  Map<String, dynamic> toJson() => {
        'word': word,
        'filePath': filePath,
        'recordedAt': recordedAt.toIso8601String(),
      };

  factory PendingTake.fromJson(Map<String, dynamic> json) => PendingTake(
        word: json['word'] as String,
        filePath: json['filePath'] as String,
        recordedAt: DateTime.parse(json['recordedAt'] as String),
      );
}

/// Cihazda biriken, henüz sunucuya yüklenmemiş kayıtları yönetir.
/// [baseDir] genelde `getApplicationDocumentsDirectory()` sonucudur.
class PendingRecordingsStore {
  final Directory baseDir;

  PendingRecordingsStore(this.baseDir);

  Directory get _recordingsDir =>
      Directory('${baseDir.path}/pending_recordings');
  File get _manifestFile => File('${_recordingsDir.path}/manifest.json');

  List<PendingTake> _takes = [];
  List<PendingTake> get takes => List.unmodifiable(_takes);

  /// Diskteki manifest dosyasından bekleyen kayıtları geri yükler.
  Future<void> load() async {
    _takes = [];
    if (!await _manifestFile.exists()) return;
    try {
      final content = await _manifestFile.readAsString();
      final list = jsonDecode(content) as List;
      _takes = list
          .map((e) => PendingTake.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) {
      _takes = [];
    }
  }

  Future<void> _save() async {
    await _recordingsDir.create(recursive: true);
    final content = jsonEncode(_takes.map((t) => t.toJson()).toList());
    await _manifestFile.writeAsString(content);
  }

  List<PendingTake> forWord(String word) =>
      _takes.where((t) => t.word == word).toList();

  Set<String> get pendingWords => _takes.map((t) => t.word).toSet();

  /// [audioFile]'ı kalıcı depoya kopyalar ve manifest'e ekler.
  Future<PendingTake> add(String word, File audioFile) async {
    await _recordingsDir.create(recursive: true);
    final destPath =
        '${_recordingsDir.path}/${word}_${DateTime.now().millisecondsSinceEpoch}.wav';
    await audioFile.copy(destPath);
    final take = PendingTake(
        word: word, filePath: destPath, recordedAt: DateTime.now());
    _takes.add(take);
    await _save();
    return take;
  }

  /// [take]'i hem diskten hem manifest'ten kaldırır.
  Future<void> remove(PendingTake take) async {
    _takes.removeWhere((t) => t.filePath == take.filePath);
    final file = File(take.filePath);
    if (await file.exists()) await file.delete();
    await _save();
  }
}
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `flutter test test/pending_recordings_test.dart`
Expected: `All tests passed!` (5 test)

- [ ] **Step 5: Statik analiz**

Run: `flutter analyze lib/pending_recordings.dart test/pending_recordings_test.dart`
Expected: Hata yok.

- [ ] **Step 6: Commit**

```bash
git add lib/pending_recordings.dart test/pending_recordings_test.dart
git commit -m "feat: add PendingRecordingsStore for on-device recording accumulation"
```

---

### Task 3: Kayıt sayfasını cihazda-biriktirme akışına çevir + Bekleyen Kayıtlar sayfası

**Files:**
- Modify: `lib/main.dart` (üst importlar; `_KayitPage`/`_KayitPageState`'in tamamı; ayrıca aynı bölgeye yeni `_PendingUploadsPage`/`_PendingUploadsPageState` eklenir)

**Interfaces:**
- Consumes: Task 1'in `exclude` parametresi, Task 2'nin `PendingRecordingsStore`/`PendingTake` API'si, mevcut `_ensureMic()`, `_startRecording()`/`_stopRecording()`, `_base()`, `kUserId`, `_ResultTile` (değişmiyor).
- Produces: yok.

- [ ] **Step 1: Import ekle**

`lib/main.dart` başındaki importlar bloğunda şu satırı bul:

```dart
import 'vad.dart';
```

Şununla değiştir:

```dart
import 'vad.dart';
import 'pending_recordings.dart';
```

- [ ] **Step 2: `_KayitPage`/`_KayitPageState`'in tamamını değiştir, `_PendingUploadsPage`'i ekle**

`lib/main.dart` içinde `// 2) KAYIT PANELİ` yorum satırıyla başlayan bölümü bul. Bu bölüm `class _KayitPage extends StatefulWidget {` ile başlar ve `// 3) AYARLAR PANELİ` yorumundan hemen önceki `}` ile biter (yani section-header'dan `_KayitPageState`'in kapanış parantezine kadar olan her şey — mevcut dosyada `_setFile`, `_word`, `_rep`, `_currentCount`, `_idealReps`, `_seconds`, `_busy`, `_recording`, `_remaining`, `_lastStatus` alanlarını ve `_init`, `_loadSets`, `_refreshWord`, `_refreshProgress`, `_upload`, `build` metotlarını içerir). Bu bloğun **tamamını sil** ve yerine şunu koy:

```dart
// ════════════════════════════════════════════════════════════════════════════
// 2) KAYIT SAYFASI — cihazda biriktir, dinle/sil, sonra topluca yükle
// ════════════════════════════════════════════════════════════════════════════
class _KayitPage extends StatefulWidget {
  const _KayitPage();

  @override
  State<_KayitPage> createState() => _KayitPageState();
}

class _KayitPageState extends State<_KayitPage> {
  final AudioPlayer _player = AudioPlayer();

  String _setFile = 'wordSet.txt';
  List<String> _sets = const [];

  String? _word;
  int _currentCount = 0;
  int _idealReps = 0;
  int _completedWords = 0;
  int _totalWords = 0;
  bool _allWordsCoveredLocally = false;

  int _seconds = 2;
  bool _busy = false;
  bool _recording = false;
  int _remaining = 0;
  String? _lastStatus;

  PendingRecordingsStore? _store;

  @override
  void initState() {
    super.initState();
    _init();
  }

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  Future<void> _init() async {
    final dir = await getApplicationDocumentsDirectory();
    final store = PendingRecordingsStore(dir);
    await store.load();
    if (!mounted) return;
    setState(() => _store = store);

    await _loadSets();
    await Future.wait([_refreshWord(), _refreshProgress()]);
  }

  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  List<PendingTake> _wordPending(String word) =>
      _store?.forWord(word) ?? const [];

  Future<void> _loadSets() async {
    try {
      final res = await http.get(Uri.parse('${_base()}/collect/word-sets'));
      final d = jsonDecode(res.body) as Map<String, dynamic>;
      final s = (d['sets'] as List?)?.map((e) => e.toString()).toList() ?? [];
      if (!mounted) return;
      setState(() {
        _sets = s;
        if (s.isNotEmpty && !s.contains(_setFile)) _setFile = s.first;
      });
    } catch (_) {}
  }

  Future<void> _refreshWord() async {
    try {
      final exclude = _store?.pendingWords.join(',') ?? '';
      final uri = Uri.parse('${_base()}/collect/next-word?user_id=$kUserId'
          '&set_file=$_setFile&exclude=${Uri.encodeQueryComponent(exclude)}');
      final res = await http.get(uri);
      final d = jsonDecode(res.body) as Map<String, dynamic>;
      if (d['error'] != null) throw Exception(d['error']);
      if (!mounted) return;
      setState(() {
        _word = d['word']?.toString();
        _currentCount = _parseInt(d['current_count']) ?? 0;
        _idealReps = _parseInt(d['ideal_repetitions']) ?? 0;
        _totalWords = _parseInt(d['total_words']) ?? 0;
        _allWordsCoveredLocally =
            _word == null && (_store?.pendingWords.isNotEmpty ?? false);
      });
    } catch (e) {
      _snack('Kelime alınamadı: $e');
    }
  }

  Future<void> _refreshProgress() async {
    try {
      final uri = Uri.parse(
          '${_base()}/collect/progress?user_id=$kUserId&set_file=$_setFile');
      final res = await http.get(uri);
      final d = jsonDecode(res.body) as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _completedWords = _parseInt(d['completed_words']) ?? 0;
        _totalWords = _parseInt(d['total_words']) ?? _totalWords;
        _idealReps = _parseInt(d['ideal_repetitions']) ?? _idealReps;
      });
    } catch (_) {}
  }

  Future<void> _record() async {
    if (_busy) return;
    final word = _word?.trim() ?? '';
    if (word.isEmpty) {
      _snack('Hedef kelime alınamadı.');
      return;
    }

    setState(() {
      _busy = true;
      _recording = true;
      _remaining = _seconds;
      _lastStatus = null;
    });

    try {
      await _ensureMic();

      // Mikrofon geri sayımla eş zamanlı açılır: kayıt hemen başlar,
      // geri sayım gerçek kayıt penceresini gösterir.
      final session = await _startRecording();
      for (var i = _seconds; i > 0; i--) {
        if (!mounted) return;
        setState(() => _remaining = i);
        await Future.delayed(const Duration(seconds: 1));
      }
      final file = await _stopRecording(session);
      if (!mounted) return;
      setState(() {
        _recording = false;
        _remaining = 0;
      });

      await _store!.add(word, file);
      if (!mounted) return;
      final count = _wordPending(word).length;
      setState(
          () => _lastStatus = 'Cihazda kaydedildi: $word (Tekrar $count)');
      _snack('Cihazda kaydedildi: $word');
    } catch (e) {
      _snack('Hata: $e');
    } finally {
      if (mounted) {
        setState(() {
          _busy = false;
          _recording = false;
          _remaining = 0;
        });
      }
    }
  }

  Future<void> _deleteTake(PendingTake take) async {
    await _store?.remove(take);
    if (!mounted) return;
    setState(() {});
  }

  Future<void> _openPendingUploads() async {
    final store = _store;
    if (store == null) return;
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => _PendingUploadsPage(store: store)),
    );
    if (!mounted) return;
    setState(() {});
    await Future.wait([_refreshWord(), _refreshProgress()]);
  }

  static int? _parseInt(dynamic v) =>
      v is int ? v : int.tryParse(v?.toString() ?? '');

  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFF10B981);
    final cs = Theme.of(context).colorScheme;

    final allDone = _word == null;
    final progress = _totalWords > 0 ? _completedWords / _totalWords : 0.0;
    final pendingCount = _store?.takes.length ?? 0;
    final wordPending =
        _word == null ? const <PendingTake>[] : _wordPending(_word!);
    final wordFull =
        _word != null && (_currentCount + wordPending.length) >= _idealReps;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Kayıt'),
        actions: [
          IconButton(
            tooltip: 'Bekleyen Kayıtlar',
            onPressed: _store == null ? null : _openPendingUploads,
            icon: Badge(
              label: Text('$pendingCount'),
              isLabelVisible: pendingCount > 0,
              child: const Icon(Icons.folder_open_rounded),
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // İlerleme
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('İlerleme: $_completedWords / $_totalWords kelime'),
                Text('${(progress * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(fontWeight: FontWeight.w700)),
              ],
            ),
            if (pendingCount > 0) ...[
              const SizedBox(height: 4),
              Text('+$pendingCount kayıt cihazda bekliyor',
                  style: TextStyle(
                      fontSize: 12, color: cs.onSurface.withOpacity(0.5))),
            ],
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: LinearProgressIndicator(
                value: progress,
                minHeight: 8,
                color: accent,
                backgroundColor: accent.withOpacity(0.15),
              ),
            ),
            const SizedBox(height: 16),

            // Set seçimi
            DropdownButtonFormField<String>(
              initialValue: _sets.isEmpty
                  ? null
                  : (_sets.contains(_setFile) ? _setFile : _sets.first),
              decoration: const InputDecoration(
                labelText: 'Kelime Seti',
                prefixIcon: Icon(Icons.list_alt_rounded),
                border: OutlineInputBorder(),
              ),
              items: (_sets.isEmpty ? [_setFile] : _sets)
                  .map((s) => DropdownMenuItem(value: s, child: Text(s)))
                  .toList(),
              onChanged: _busy
                  ? null
                  : (v) async {
                      if (v == null) return;
                      setState(() => _setFile = v);
                      await Future.wait([_refreshWord(), _refreshProgress()]);
                    },
            ),
            const SizedBox(height: 16),

            // Hedef kelime kartı
            Container(
              decoration: BoxDecoration(
                color: allDone
                    ? accent.withOpacity(0.12)
                    : cs.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: accent.withOpacity(0.4)),
              ),
              padding: const EdgeInsets.all(20),
              child: allDone
                  ? Column(
                      children: [
                        const Icon(Icons.check_circle_rounded,
                            color: accent, size: 28),
                        const SizedBox(height: 8),
                        Text(
                          _allWordsCoveredLocally
                              ? 'Bu setteki kelimeler için yerel kayıt tamamlandı.'
                              : 'Tüm kelimeler tamamlandı!',
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: accent),
                        ),
                        if (_allWordsCoveredLocally) ...[
                          const SizedBox(height: 6),
                          Text(
                            'Yüklemek için sağ üstteki klasör ikonuna dokun.',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                                fontSize: 12,
                                color: cs.onSurface.withOpacity(0.6)),
                          ),
                        ],
                      ],
                    )
                  : Column(
                      children: [
                        Text(
                          _word ?? '-',
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                              fontSize: 30, fontWeight: FontWeight.w900),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Sıradaki Tekrar: ${_currentCount + wordPending.length + 1}'
                          '  ·  Mevcut: ${_currentCount + wordPending.length} / $_idealReps',
                          style: TextStyle(
                              fontSize: 13,
                              color: cs.onSurface.withOpacity(0.6)),
                        ),
                        if (wordPending.isNotEmpty) ...[
                          const SizedBox(height: 12),
                          ...wordPending.asMap().entries.map((entry) {
                            final take = entry.value;
                            return Padding(
                              padding: const EdgeInsets.only(bottom: 4),
                              child: Row(
                                children: [
                                  Expanded(
                                    child: Text(
                                      'Tekrar ${_currentCount + entry.key + 1} (cihazda)',
                                      style: const TextStyle(fontSize: 12),
                                    ),
                                  ),
                                  IconButton(
                                    icon: const Icon(Icons.play_arrow_rounded,
                                        size: 20),
                                    onPressed: () => _player
                                        .play(DeviceFileSource(take.filePath)),
                                  ),
                                  IconButton(
                                    icon: const Icon(
                                        Icons.delete_outline_rounded,
                                        size: 20),
                                    onPressed: () => _deleteTake(take),
                                  ),
                                ],
                              ),
                            );
                          }),
                        ],
                      ],
                    ),
            ),
            const SizedBox(height: 16),

            // Süre slider
            Row(
              children: [
                const Icon(Icons.timer_outlined, size: 18),
                const SizedBox(width: 8),
                Text('Kayıt süresi: $_seconds sn'),
              ],
            ),
            Slider(
              value: _seconds.toDouble(),
              min: 1,
              max: 6,
              divisions: 5,
              label: '$_seconds sn',
              activeColor: accent,
              onChanged:
                  _busy ? null : (v) => setState(() => _seconds = v.round()),
            ),

            // Progress bar
            if (_recording) ...[
              LinearProgressIndicator(
                value:
                    _seconds == 0 ? null : (_seconds - _remaining) / _seconds,
                color: accent,
              ),
              const SizedBox(height: 6),
              Text('Kayıt alınıyor... $_remaining sn kaldı',
                  style: const TextStyle(fontSize: 12)),
              const SizedBox(height: 8),
            ],

            const SizedBox(height: 8),
            FilledButton.icon(
              style: FilledButton.styleFrom(
                backgroundColor: accent,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
              ),
              onPressed: (_busy || allDone || wordFull) ? null : _record,
              icon: Icon(_busy ? Icons.hourglass_top : Icons.mic_rounded),
              label: Text(
                _busy
                    ? (_recording ? 'Kaydediliyor...' : 'İşleniyor...')
                    : 'Kaydet (Cihazda Sakla)',
                style: const TextStyle(
                    fontSize: 16, fontWeight: FontWeight.w700),
              ),
            ),
            const SizedBox(height: 10),
            OutlinedButton.icon(
              onPressed: _busy ? null : () => _refreshWord(),
              icon: const Icon(Icons.skip_next_rounded),
              label: const Text('Sonraki Kelime'),
            ),
            const SizedBox(height: 10),
            OutlinedButton.icon(
              onPressed: _busy ? null : _refreshProgress,
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

// ════════════════════════════════════════════════════════════════════════════
// 2b) BEKLEYEN KAYITLAR SAYFASI
// ════════════════════════════════════════════════════════════════════════════

enum _UploadStatus { uploading, error }

class _PendingUploadsPage extends StatefulWidget {
  final PendingRecordingsStore store;
  const _PendingUploadsPage({required this.store});

  @override
  State<_PendingUploadsPage> createState() => _PendingUploadsPageState();
}

class _PendingUploadsPageState extends State<_PendingUploadsPage> {
  final AudioPlayer _player = AudioPlayer();
  bool _uploadingAll = false;
  final Map<String, _UploadStatus> _status = {};
  final Map<String, String> _errors = {};

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  Future<void> _deleteTake(PendingTake take) async {
    await widget.store.remove(take);
    if (!mounted) return;
    setState(() {});
  }

  Future<void> _uploadAll() async {
    if (_uploadingAll) return;
    setState(() => _uploadingAll = true);

    final takes = List<PendingTake>.from(widget.store.takes);
    for (final take in takes) {
      setState(() => _status[take.filePath] = _UploadStatus.uploading);
      try {
        final req =
            http.MultipartRequest('POST', Uri.parse('${_base()}/record'))
              ..fields['user_id'] = kUserId
              ..fields['word'] = take.word
              ..files.add(
                  await http.MultipartFile.fromPath('audio', take.filePath));

        final res = await req.send();
        final body = await res.stream.bytesToString();
        if (res.statusCode != 200) {
          throw Exception('Sunucu: ${res.statusCode} $body');
        }

        await widget.store.remove(take);
        if (!mounted) return;
        setState(() => _status.remove(take.filePath));
      } catch (e) {
        if (!mounted) return;
        setState(() {
          _status[take.filePath] = _UploadStatus.error;
          _errors[take.filePath] = '$e';
        });
      }
    }

    if (!mounted) return;
    setState(() => _uploadingAll = false);
  }

  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFF10B981);
    final cs = Theme.of(context).colorScheme;
    final takes = widget.store.takes;

    final byWord = <String, List<PendingTake>>{};
    for (final t in takes) {
      byWord.putIfAbsent(t.word, () => []).add(t);
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Bekleyen Kayıtlar')),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              '${takes.length} kayıt, ${byWord.length} kelime bekliyor.',
              style: TextStyle(
                  fontSize: 13, color: cs.onSurface.withOpacity(0.65)),
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              style: FilledButton.styleFrom(
                backgroundColor: accent,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
              ),
              onPressed: (_uploadingAll || takes.isEmpty) ? null : _uploadAll,
              icon: Icon(_uploadingAll
                  ? Icons.hourglass_top
                  : Icons.cloud_upload_rounded),
              label: Text(
                _uploadingAll ? 'Yükleniyor...' : 'Tümünü Yükle',
                style: const TextStyle(
                    fontSize: 16, fontWeight: FontWeight.w700),
              ),
            ),
            const SizedBox(height: 20),
            Expanded(
              child: takes.isEmpty
                  ? Center(
                      child: Text(
                        'Bekleyen kayıt yok.',
                        style: TextStyle(
                            fontSize: 13,
                            color: cs.onSurface.withOpacity(0.4)),
                      ),
                    )
                  : ListView(
                      children: byWord.entries.map((entry) {
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 16),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                entry.key,
                                style: const TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.w800),
                              ),
                              const SizedBox(height: 6),
                              ...entry.value.asMap().entries.map((e) {
                                final take = e.value;
                                final status = _status[take.filePath];
                                return Padding(
                                  padding: const EdgeInsets.only(bottom: 6),
                                  child: Row(
                                    children: [
                                      Expanded(
                                          child: Text('Tekrar ${e.key + 1}')),
                                      if (status == _UploadStatus.uploading)
                                        const SizedBox(
                                          width: 16,
                                          height: 16,
                                          child: CircularProgressIndicator(
                                              strokeWidth: 2),
                                        )
                                      else if (status == _UploadStatus.error)
                                        Tooltip(
                                          message:
                                              _errors[take.filePath] ??
                                                  'Hata',
                                          child: const Icon(
                                              Icons.error_outline_rounded,
                                              color: Colors.red,
                                              size: 20),
                                        )
                                      else ...[
                                        IconButton(
                                          icon: const Icon(
                                              Icons.play_arrow_rounded,
                                              size: 20),
                                          onPressed: () => _player.play(
                                              DeviceFileSource(
                                                  take.filePath)),
                                        ),
                                        IconButton(
                                          icon: const Icon(
                                              Icons.delete_outline_rounded,
                                              size: 20),
                                          onPressed: () => _deleteTake(take),
                                        ),
                                      ],
                                    ],
                                  ),
                                );
                              }),
                            ],
                          ),
                        );
                      }).toList(),
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
```

- [ ] **Step 3: Diğer sayfaların (Furkanca/Ayarlar) hâlâ derlendiğini/etkilenmediğini doğrula**

Run: `flutter test test/panel_navigation_test.dart`
Expected: PASS (Furkanca navigasyonu bu değişiklikten etkilenmemeli).

- [ ] **Step 4: Statik analiz**

Run: `flutter analyze lib/main.dart`
Expected: `_KayitPanel`/`_rep`/`_upload` gibi eski isimlerle veya yeni `_PendingUploadsPage` ile ilgili hata yok. (Var olan `withOpacity` deprecation info'ları kalabilir.)

- [ ] **Step 5: Commit**

```bash
git add lib/main.dart
git commit -m "feat: accumulate Kayıt recordings on-device with listen/delete before upload"
```

---

### Task 4: Temizlik ve uçtan uca doğrulama

**Files:**
- Modify: `lib/main.dart` (yalnızca `dart format`)

**Interfaces:**
- Consumes: Task 1-3'ün tüm çıktıları.
- Produces: yok.

- [ ] **Step 1: Kodu formatla**

Run: `dart format lib/main.dart lib/pending_recordings.dart test/pending_recordings_test.dart`

- [ ] **Step 2: Tam proje analizi**

Run: `flutter analyze`
Expected: Yeni hata yok (yalnızca önceden var olan `withOpacity`/`curly_braces` info'ları ve kapsam dışı `test/widget_test.dart` hatası kalabilir).

- [ ] **Step 3: Tüm ilgili testleri çalıştır**

Run: `flutter test test/pending_recordings_test.dart test/vad_test.dart test/panel_navigation_test.dart`
Expected: `All tests passed!`

- [ ] **Step 4: Commit**

```bash
git add lib/main.dart
git commit -m "chore: format after local recording accumulation feature"
```

(Eğer Step 1 hiçbir dosyayı değiştirmediyse bu adımı atla.)

- [ ] **Step 5: Manuel doğrulama (sunucu + cihaz)**

Sunucu tarafı (curl ile, `exclude` parametresinin çalıştığını doğrulamak için):

```bash
curl "http://<sunucu>:8001/collect/next-word?user_id=FurkanV1&set_file=wordSet.txt"
# Dönen "word" değerini not et, sonra:
curl "http://<sunucu>:8001/collect/next-word?user_id=FurkanV1&set_file=wordSet.txt&exclude=<o kelime>"
# Farklı bir kelime dönmeli (veya set küçükse done:true).
```

Cihazda:

```bash
flutter run
```

1. Kayıt sayfasını aç, bir kelime için 2 tekrar kaydet — her ikisi de kelime kartının altında "(cihazda)" etiketiyle görünmeli, ▶ ile dinlenebilmeli.
2. Birini sil — listeden kalkmalı, "Mevcut: N/ideal" sayacı azalmalı.
3. "Sonraki Kelime"ye bas — **hiçbir şey yüklenmeden** farklı bir kelime gelmeli; ilerleme çubuğu değişmemeli ("+N kayıt cihazda bekliyor" notu artmalı).
4. Sağ üstteki rozetli klasör ikonuna bas → "Bekleyen Kayıtlar" sayfası açılmalı, iki kelime altında kayıtlar görünmeli.
5. "Tümünü Yükle"ye bas — her kayıt sırayla yüklenmeli, başarılı olanlar listeden kalkmalı.
6. Kayıt sayfasına dön — ilerleme çubuğu artık sunucudan gelen gerçek sayıyı yansıtmalı, "+N cihazda bekliyor" notu kaybolmalı (hepsi yüklendiyse).
7. Uygulamayı tamamen kapatıp yeniden aç, birkaç kayıt alıp yüklemeden kapat, tekrar aç — kayıtların hâlâ "Bekleyen Kayıtlar"da durduğunu doğrula (kalıcılık kontrolü).

## Self-Review Notları

- **Spec kapsaması:** Spec'in tüm bölümleri (1: backend eklentisi → Task 1; 2: yerel depolama → Task 2; 3: kayıt akışı → Task 3; 4: kelimeler arası geçiş → Task 1+3; 5: Bekleyen Kayıtlar ekranı → Task 3; 6: ilerleme göstergesi → Task 3) karşılanıyor.
- **Placeholder taraması:** Yok.
- **Tip/isim tutarlılığı:** `PendingRecordingsStore`/`PendingTake` isimleri Task 2'de tanımlandığı gibi Task 3'te birebir kullanıldı; `exclude` parametre adı Task 1 (backend) ve Task 3 (mobil çağrı) arasında tutarlı.
