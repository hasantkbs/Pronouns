# Furkanca Sonuç Sadeleştirme + Gerçek Zamanlı Dinleme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Furkanca sayfasında Duyulan/Niyet/Eksik Kelimeler kutularını kaldırıp yalnızca çeviriyi göstermek; sabit süreli manuel kayıt akışını, konuşmayı otomatik algılayıp art arda çeviren sürekli bir dinleme döngüsüyle değiştirmek.

**Architecture:** İstemci tarafında saf-Dart, platformdan bağımsız bir VAD (konuşma algılama) modülü (`lib/vad.dart`) eklenir. `record` paketinin `startStream()` API'siyle sürekli ham PCM16 akışı alınır, bu modülle konuşma başlangıcı/sonu segmentlenir, her segment WAV'a çevrilip mevcut `/translate` endpoint'ine değişmeden gönderilir. `_FurkancaPageState` bu döngüyü yönetir ve sonuçları kayan bir geçmiş listesinde gösterir. Backend ve `_KayitPage` değişmez.

**Tech Stack:** Flutter/Dart, `record: ^6.2.0` (`startStream`/`pause`/`resume`), `flutter_test`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-09-realtime-translation-design.md`.
- Backend (`api.py`, `/translate` sözleşmesi) değişmeyecek.
- `_KayitPage` ve oradaki sabit süreli kayıt akışı (önceki turda düzeltilen eş-zamanlı kayıt mantığı dahil) değişmeyecek.
- Onset/silence eşikleri sunucudaki `config.py` değerleriyle aynı: `AUTO_SOUND_THRESHOLD=0.012`, `AUTO_SILENCE_LIMIT_SEC=1.0`.
- Gerçek streaming ASR (kelime kelime canlı transkripsiyon, WebSocket) kapsam dışı.

---

### Task 1: VAD segmentleyici modülü + WAV encoder (TDD, saf Dart)

**Files:**
- Create: `lib/vad.dart`
- Create: `test/vad_test.dart`

**Interfaces:**
- Produces: `double calculateRms(Uint8List)`, `Duration pcmChunkDuration(Uint8List, {int sampleRate})`, `Uint8List pcmToWav(Uint8List, {int sampleRate, int numChannels, int bitsPerSample})`, `enum VadEventType {started, ended, discardedTooShort}`, `class VadEvent {final VadEventType type; final Uint8List? pcmData;}`, `class VadSegmenter {VadEvent? addChunk(Uint8List chunk); VadSegmenter({double onsetThreshold, Duration silenceTimeout, Duration maxDuration, Duration minSpeechDuration, Duration preRollDuration, int sampleRate})}` — Task 2 bu API'yi doğrudan kullanacak.

- [ ] **Step 1: Başarısız olacak testleri yaz**

`test/vad_test.dart` dosyasını oluştur:

```dart
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:pronouns_ai/vad.dart';

// 16kHz, 16-bit mono => 100ms = 1600 örnek = 3200 bayt.
const _sampleRate = 16000;
const _samplesPerChunk = _sampleRate * 100 ~/ 1000;
const _bytesPerChunk = _samplesPerChunk * 2;

Uint8List _chunk(double amplitude) {
  final data = ByteData(_samplesPerChunk * 2);
  final value = (amplitude * 32767).round().clamp(-32768, 32767);
  for (var i = 0; i < _samplesPerChunk; i++) {
    data.setInt16(i * 2, value, Endian.little);
  }
  return data.buffer.asUint8List();
}

Uint8List _loud() => _chunk(0.5);
Uint8List _silent() => _chunk(0.0);

void main() {
  group('calculateRms', () {
    test('silence is below the default onset threshold', () {
      expect(calculateRms(_silent()), lessThan(0.012));
    });

    test('loud audio is above the default onset threshold', () {
      expect(calculateRms(_loud()), greaterThan(0.012));
    });
  });

  group('pcmToWav', () {
    test('produces a 44-byte header followed by the PCM data', () {
      final pcm = _loud();
      final wav = pcmToWav(pcm);

      expect(wav.length, 44 + pcm.length);
      expect(String.fromCharCodes(wav.sublist(0, 4)), 'RIFF');
      expect(String.fromCharCodes(wav.sublist(8, 12)), 'WAVE');
      expect(wav.sublist(44), pcm);
    });
  });

  group('VadSegmenter', () {
    test('a short blip followed by silence is discarded', () {
      final segmenter = VadSegmenter();

      expect(segmenter.addChunk(_loud())!.type, VadEventType.started);
      for (var i = 0; i < 9; i++) {
        expect(segmenter.addChunk(_silent()), isNull);
      }
      final ended = segmenter.addChunk(_silent());
      expect(ended!.type, VadEventType.discardedTooShort);
      expect(ended.pcmData, isNull);
    });

    test('speech of at least minSpeechDuration produces a valid segment', () {
      final segmenter = VadSegmenter();

      expect(segmenter.addChunk(_loud())!.type, VadEventType.started);
      for (var i = 0; i < 3; i++) {
        expect(segmenter.addChunk(_loud()), isNull);
      }
      for (var i = 0; i < 9; i++) {
        expect(segmenter.addChunk(_silent()), isNull);
      }
      final ended = segmenter.addChunk(_silent());

      expect(ended!.type, VadEventType.ended);
      // 4 gürültülü + 10 sessiz parça = 14 * 3200 bayt.
      expect(ended.pcmData!.length, 14 * _bytesPerChunk);
    });

    test('pre-roll audio before onset is included in the final segment', () {
      final segmenter = VadSegmenter();

      expect(segmenter.addChunk(_silent()), isNull);
      expect(segmenter.addChunk(_silent()), isNull);
      expect(segmenter.addChunk(_loud())!.type, VadEventType.started);
      for (var i = 0; i < 3; i++) {
        expect(segmenter.addChunk(_loud()), isNull);
      }
      for (var i = 0; i < 9; i++) {
        expect(segmenter.addChunk(_silent()), isNull);
      }
      final ended = segmenter.addChunk(_silent());

      expect(ended!.type, VadEventType.ended);
      // 2 ön-tampon + 4 gürültülü + 10 sessiz parça = 16 * 3200 bayt.
      expect(ended.pcmData!.length, 16 * _bytesPerChunk);
    });

    test('continuous speech is force-ended at maxDuration', () {
      final segmenter = VadSegmenter();

      expect(segmenter.addChunk(_loud())!.type, VadEventType.started);
      // chunk 1 zaten gönderildi; maxDuration (12.0s) tam 120 parçada dolar.
      for (var i = 0; i < 118; i++) {
        expect(segmenter.addChunk(_loud()), isNull);
      }
      final ended = segmenter.addChunk(_loud());

      expect(ended!.type, VadEventType.ended);
      expect(ended.pcmData!.length, 120 * _bytesPerChunk);
    });
  });
}
```

- [ ] **Step 2: Testlerin şu an derlenmediğini/başarısız olduğunu doğrula**

Run: `flutter test test/vad_test.dart`
Expected: FAIL — `package:pronouns_ai/vad.dart` bulunamadığı için derleme hatası.

- [ ] **Step 3: `lib/vad.dart` modülünü yaz**

```dart
import 'dart:math' show sqrt;
import 'dart:typed_data';

/// Ham PCM16 (little-endian, mono) örneklerinin ses şiddetini (RMS) hesaplar.
/// Sunucudaki (config.py) RMS tanımıyla aynı birimde: örnekler -1.0..1.0
/// aralığına normalize edilip karelerinin ortalamasının karekökü alınır.
double calculateRms(Uint8List pcm16Bytes) {
  final sampleCount = pcm16Bytes.length ~/ 2;
  if (sampleCount == 0) return 0.0;

  final data = ByteData.sublistView(pcm16Bytes);
  double sumSquares = 0;
  for (var i = 0; i < sampleCount; i++) {
    final normalized = data.getInt16(i * 2, Endian.little) / 32768.0;
    sumSquares += normalized * normalized;
  }
  return sqrt(sumSquares / sampleCount);
}

/// Bir PCM16 mono parçanın süresi.
Duration pcmChunkDuration(Uint8List pcm16Bytes, {int sampleRate = 16000}) {
  final sampleCount = pcm16Bytes.length ~/ 2;
  final seconds = sampleCount / sampleRate;
  return Duration(
      microseconds: (seconds * Duration.microsecondsPerSecond).round());
}

Uint8List _wavHeader(
  int pcmDataLength, {
  required int sampleRate,
  required int numChannels,
  required int bitsPerSample,
}) {
  final byteRate = sampleRate * numChannels * bitsPerSample ~/ 8;
  final blockAlign = numChannels * bitsPerSample ~/ 8;
  final header = ByteData(44);

  void writeString(int offset, String s) {
    for (var i = 0; i < s.length; i++) {
      header.setUint8(offset + i, s.codeUnitAt(i));
    }
  }

  writeString(0, 'RIFF');
  header.setUint32(4, 36 + pcmDataLength, Endian.little);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  header.setUint32(16, 16, Endian.little);
  header.setUint16(20, 1, Endian.little);
  header.setUint16(22, numChannels, Endian.little);
  header.setUint32(24, sampleRate, Endian.little);
  header.setUint32(28, byteRate, Endian.little);
  header.setUint16(32, blockAlign, Endian.little);
  header.setUint16(34, bitsPerSample, Endian.little);
  writeString(36, 'data');
  header.setUint32(40, pcmDataLength, Endian.little);

  return header.buffer.asUint8List();
}

/// Ham PCM16 verisini geçerli bir WAV dosyası (byte dizisi) haline getirir.
Uint8List pcmToWav(
  Uint8List pcmData, {
  int sampleRate = 16000,
  int numChannels = 1,
  int bitsPerSample = 16,
}) {
  final header = _wavHeader(
    pcmData.length,
    sampleRate: sampleRate,
    numChannels: numChannels,
    bitsPerSample: bitsPerSample,
  );
  final result = Uint8List(header.length + pcmData.length);
  result.setRange(0, header.length, header);
  result.setRange(header.length, result.length, pcmData);
  return result;
}

enum VadEventType { started, ended, discardedTooShort }

class VadEvent {
  final VadEventType type;
  final Uint8List? pcmData;
  const VadEvent(this.type, {this.pcmData});
}

/// Sürekli PCM16 akışını dinleyip konuşma başlangıcını/sonunu algılayan,
/// platformdan bağımsız (test edilebilir) konuşma segmentleyici.
///
/// [onsetThreshold]/[silenceTimeout] varsayılanları, sunucu tarafındaki
/// config.py'deki AUTO_SOUND_THRESHOLD/AUTO_SILENCE_LIMIT_SEC ile aynıdır.
class VadSegmenter {
  final double onsetThreshold;
  final Duration silenceTimeout;
  final Duration maxDuration;
  final Duration minSpeechDuration;
  final Duration preRollDuration;
  final int sampleRate;

  VadSegmenter({
    this.onsetThreshold = 0.012,
    this.silenceTimeout = const Duration(seconds: 1),
    this.maxDuration = const Duration(seconds: 12),
    this.minSpeechDuration = const Duration(milliseconds: 300),
    this.preRollDuration = const Duration(milliseconds: 300),
    this.sampleRate = 16000,
  });

  final List<Uint8List> _preRoll = [];
  Duration _preRollBuffered = Duration.zero;

  bool _capturing = false;
  final List<int> _buffer = [];
  Duration _capturedDuration = Duration.zero;
  Duration _speechDuration = Duration.zero;
  Duration _silenceElapsed = Duration.zero;

  /// Yeni bir ham PCM16 parçası bildirir. Bir olay oluşmadıysa `null` döner.
  VadEvent? addChunk(Uint8List chunk) {
    final chunkDuration = pcmChunkDuration(chunk, sampleRate: sampleRate);
    final isSpeech = calculateRms(chunk) >= onsetThreshold;

    if (!_capturing) {
      if (!isSpeech) {
        _preRoll.add(chunk);
        _preRollBuffered += chunkDuration;
        while (_preRollBuffered > preRollDuration && _preRoll.isNotEmpty) {
          _preRollBuffered -=
              pcmChunkDuration(_preRoll.removeAt(0), sampleRate: sampleRate);
        }
        return null;
      }

      _capturing = true;
      _buffer.clear();
      for (final c in _preRoll) {
        _buffer.addAll(c);
      }
      _buffer.addAll(chunk);
      _capturedDuration = _preRollBuffered + chunkDuration;
      _speechDuration = chunkDuration;
      _silenceElapsed = Duration.zero;
      _preRoll.clear();
      _preRollBuffered = Duration.zero;
      return const VadEvent(VadEventType.started);
    }

    _buffer.addAll(chunk);
    _capturedDuration += chunkDuration;
    if (isSpeech) {
      _speechDuration += chunkDuration;
      _silenceElapsed = Duration.zero;
    } else {
      _silenceElapsed += chunkDuration;
    }

    final shouldEnd =
        _silenceElapsed >= silenceTimeout || _capturedDuration >= maxDuration;
    if (!shouldEnd) return null;

    _capturing = false;
    final pcm = Uint8List.fromList(_buffer);
    final tooShort = _speechDuration < minSpeechDuration;
    _buffer.clear();
    _capturedDuration = Duration.zero;
    _speechDuration = Duration.zero;
    _silenceElapsed = Duration.zero;

    return tooShort
        ? const VadEvent(VadEventType.discardedTooShort)
        : VadEvent(VadEventType.ended, pcmData: pcm);
  }
}
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `flutter test test/vad_test.dart`
Expected: `All tests passed!` (9 test)

- [ ] **Step 5: Statik analiz**

Run: `flutter analyze lib/vad.dart test/vad_test.dart`
Expected: Hata yok.

- [ ] **Step 6: Commit**

```bash
git add lib/vad.dart test/vad_test.dart
git commit -m "feat: add platform-independent VAD segmenter + WAV encoder"
```

---

### Task 2: Furkanca sayfasını sürekli dinleme döngüsüne çevir

**Files:**
- Modify: `lib/main.dart` (üst importlar; `_FurkancaPage`/`_FurkancaPageState` sınıflarının tamamı — mevcut hâliyle satır 314-524 arası, bu görevin önceki adımlarındaki değişikliklerle satır numaraları kayabilir, sınıf içeriğine göre eşleştirin)

**Interfaces:**
- Consumes: Task 1'in `VadSegmenter`, `VadEvent`, `VadEventType`, `pcmToWav` API'leri; dosyanın başındaki mevcut `_ensureMic()`, `_base()` yardımcı fonksiyonları (değişmiyor, `_KayitPage` de kullanıyor).
- Produces: yok (bu görev sadece Furkanca sayfasını tüketiyor).

- [ ] **Step 1: Gerekli importları ekle**

`lib/main.dart` başındaki şu bloğu bul:

```dart
import 'dart:async';
import 'dart:convert';
import 'dart:io' show File;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:permission_handler/permission_handler.dart';
```

Şununla değiştir:

```dart
import 'dart:async';
import 'dart:convert';
import 'dart:io' show File;
import 'dart:typed_data' show Uint8List;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:permission_handler/permission_handler.dart';

import 'vad.dart';
```

- [ ] **Step 2: `_FurkancaPage`/`_FurkancaPageState` sınıflarının tamamını değiştir**

`lib/main.dart` içinde şu satırla başlayan bölümü bul:

```dart
// ════════════════════════════════════════════════════════════════════════════
// 1) FURKANCA PANELİ
// ════════════════════════════════════════════════════════════════════════════
class _FurkancaPage extends StatefulWidget {
```

... ve `class _KayitPage extends StatefulWidget {` satırından hemen önceki `}` ile biten bölümün tamamını (yani section-header yorumundan `_FurkancaPageState`'in kapanış `}`'ine kadar — mevcut dosyada bu blok `_player`, `_seconds`, `_busy`, `_recording`, `_remaining`, `_recognized`, `_corrected`, `_intent`, `_missing` alanlarını ve `_run()`, `build()` metotlarını içerir) **tamamen sil** ve yerine şunu koy:

```dart
// ════════════════════════════════════════════════════════════════════════════
// 1) FURKANCA SAYFASI — sürekli dinle / algıla / çevir
// ════════════════════════════════════════════════════════════════════════════

enum _ListenState { idle, listening, capturing, uploading, speaking }

class _TranslationEntry {
  final String text;
  final bool isError;
  final DateTime time;

  _TranslationEntry(this.text, {this.isError = false, required this.time});
}

class _FurkancaPage extends StatefulWidget {
  const _FurkancaPage();

  @override
  State<_FurkancaPage> createState() => _FurkancaPageState();
}

class _FurkancaPageState extends State<_FurkancaPage> {
  final AudioPlayer _player = AudioPlayer();

  AudioRecorder? _recorder;
  StreamSubscription<Uint8List>? _pcmSub;
  VadSegmenter? _segmenter;

  _ListenState _state = _ListenState.idle;
  final List<_TranslationEntry> _history = [];

  @override
  void dispose() {
    _pcmSub?.cancel();
    _recorder?.dispose();
    _player.dispose();
    super.dispose();
  }

  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  Future<void> _toggleListening() async {
    if (_state == _ListenState.idle) {
      await _startListening();
    } else {
      await _stopListening();
    }
  }

  Future<void> _startListening() async {
    try {
      await _ensureMic();
    } catch (e) {
      _snack('Hata: $e');
      return;
    }

    final recorder = AudioRecorder();
    final stream = await recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        numChannels: 1,
        sampleRate: 16000,
      ),
    );

    _recorder = recorder;
    _segmenter = VadSegmenter();
    _pcmSub = stream.listen(_onPcmChunk);

    if (!mounted) return;
    setState(() => _state = _ListenState.listening);
  }

  Future<void> _stopListening() async {
    await _pcmSub?.cancel();
    _pcmSub = null;
    await _recorder?.stop();
    await _recorder?.dispose();
    _recorder = null;
    _segmenter = null;

    if (!mounted) return;
    setState(() => _state = _ListenState.idle);
  }

  void _onPcmChunk(Uint8List chunk) {
    final segmenter = _segmenter;
    if (segmenter == null) return;

    final event = segmenter.addChunk(chunk);
    if (event == null) return;

    switch (event.type) {
      case VadEventType.started:
        // Sadece normal dinleme akışındayken durumu güncelle; bir segment
        // yüklenirken/seslendirilirken gelen sapma parçaları yok say.
        if (mounted && _state == _ListenState.listening) {
          setState(() => _state = _ListenState.capturing);
        }
        break;
      case VadEventType.discardedTooShort:
        if (mounted && _state == _ListenState.capturing) {
          setState(() => _state = _ListenState.listening);
        }
        break;
      case VadEventType.ended:
        unawaited(_handleSegment(event.pcmData!));
        break;
    }
  }

  Future<void> _handleSegment(Uint8List pcmData) async {
    if (!mounted) return;
    setState(() => _state = _ListenState.uploading);
    await _recorder?.pause();

    try {
      final wavBytes = pcmToWav(pcmData);
      final dir = await getTemporaryDirectory();
      final path =
          '${dir.path}/live_${DateTime.now().millisecondsSinceEpoch}.wav';
      final file = File(path);
      await file.writeAsBytes(wavBytes);

      final uri = Uri.parse('${_base()}/translate');
      final req = http.MultipartRequest('POST', uri)
        ..fields['user_id'] = kUserId
        ..files.add(await http.MultipartFile.fromPath('audio', file.path));

      final res = await req.send();
      final body = await res.stream.bytesToString();

      if (res.statusCode != 200) {
        throw Exception('Sunucu hatası (${res.statusCode}): $body');
      }

      final d = jsonDecode(body) as Map<String, dynamic>;
      final corrected = d['response_text']?.toString() ?? '';

      if (!mounted) return;
      setState(() {
        _history.insert(
          0,
          _TranslationEntry(
            corrected.isEmpty ? '(boş yanıt)' : corrected,
            time: DateTime.now(),
          ),
        );
      });

      final audioUrl = d['audio_url'];
      if (audioUrl is String && audioUrl.isNotEmpty) {
        if (!mounted) return;
        setState(() => _state = _ListenState.speaking);
        final full =
            audioUrl.startsWith('http') ? audioUrl : '${_base()}$audioUrl';
        final completeFuture = _player.onPlayerComplete.first;
        await _player.play(UrlSource(full));
        await completeFuture;
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _history.insert(
          0,
          _TranslationEntry('Hata: $e', isError: true, time: DateTime.now()),
        );
      });
    } finally {
      if (mounted && _state != _ListenState.idle) {
        await _recorder?.resume();
        setState(() => _state = _ListenState.listening);
      }
    }
  }

  String _statusText() {
    switch (_state) {
      case _ListenState.idle:
        return 'Dinlemek için başlat\'a bas.';
      case _ListenState.listening:
        return 'Dinleniyor... konuşabilirsin.';
      case _ListenState.capturing:
        return 'Kaydediliyor...';
      case _ListenState.uploading:
        return 'İşleniyor...';
      case _ListenState.speaking:
        return 'Yanıt seslendiriliyor...';
    }
  }

  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFF6C63FF);
    final cs = Theme.of(context).colorScheme;
    final isListening = _state != _ListenState.idle;

    return Scaffold(
      appBar: AppBar(title: const Text('Furkanca')),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Dinlemeyi başlat, konuş — AI konuşmanı düzeltilmiş hâle çevirsin ve yüksek sesle okusun.',
              style: TextStyle(
                  fontSize: 13, color: cs.onSurface.withOpacity(0.65)),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Icon(
                  isListening
                      ? Icons.graphic_eq_rounded
                      : Icons.mic_off_rounded,
                  color: accent,
                  size: 18,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child:
                      Text(_statusText(), style: const TextStyle(fontSize: 13)),
                ),
              ],
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              style: FilledButton.styleFrom(
                backgroundColor: isListening ? Colors.redAccent : accent,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
              ),
              onPressed: _toggleListening,
              icon: Icon(isListening ? Icons.stop_rounded : Icons.mic_rounded),
              label: Text(
                isListening ? 'Dinlemeyi Durdur' : 'Dinlemeyi Başlat',
                style:
                    const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
              ),
            ),
            const SizedBox(height: 20),
            Expanded(
              child: _history.isEmpty
                  ? Center(
                      child: Text(
                        'Henüz çeviri yok.',
                        style: TextStyle(
                            fontSize: 13,
                            color: cs.onSurface.withOpacity(0.4)),
                      ),
                    )
                  : ListView.separated(
                      itemCount: _history.length,
                      separatorBuilder: (_, __) => const SizedBox(height: 10),
                      itemBuilder: (_, i) {
                        final entry = _history[i];
                        final color = entry.isError ? Colors.red : accent;
                        return Container(
                          padding: const EdgeInsets.all(14),
                          decoration: BoxDecoration(
                            color: color.withOpacity(0.08),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: color.withOpacity(0.3)),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                '${entry.time.hour.toString().padLeft(2, '0')}:'
                                '${entry.time.minute.toString().padLeft(2, '0')}:'
                                '${entry.time.second.toString().padLeft(2, '0')}',
                                style: TextStyle(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w700,
                                    color: color),
                              ),
                              const SizedBox(height: 6),
                              SelectableText(
                                entry.text,
                                style: TextStyle(
                                    fontSize: 14, color: cs.onSurface),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
```

- [ ] **Step 3: `panel_navigation_test.dart`'ın hâlâ geçtiğini doğrula**

Run: `flutter test test/panel_navigation_test.dart`
Expected: PASS — bu test yalnızca "Furkanca butonuna basınca tam sayfa açılıyor mu" kontrolü yapıyor (buton etiketi/sublabel'i, AppBar'ı değişmedi), yeni dinleme mantığından etkilenmemeli.

- [ ] **Step 4: Statik analiz**

Run: `flutter analyze lib/main.dart`
Expected: `_recognized`/`_intent`/`_missing`/`_seconds` (Furkanca'ya özgü) veya `_FurkancaPage`/`_FurkancaPageState` ile ilgili hata yok. (Var olan `withOpacity` deprecation info'ları kalabilir, bunlar bu görevin kapsamı dışında.)

- [ ] **Step 5: Commit**

```bash
git add lib/main.dart
git commit -m "feat: replace fixed-duration recording with continuous VAD-driven listening on Furkanca page"
```

---

### Task 3: Temizlik ve uçtan uca doğrulama

**Files:**
- Modify: `lib/main.dart` (yalnızca `dart format`)

**Interfaces:**
- Consumes: Task 1 + Task 2'nin tüm çıktıları.
- Produces: yok (doğrulama görevi).

- [ ] **Step 1: Kodu formatla**

Run: `dart format lib/main.dart lib/vad.dart test/vad_test.dart`

- [ ] **Step 2: Tam proje analizi**

Run: `flutter analyze`
Expected: Yeni hata yok (yalnızca önceden var olan `withOpacity`/`curly_braces` info uyarıları kalabilir; `test/widget_test.dart`'taki önceden var olan `MyApp` hatası da kapsam dışı, dokunulmuyor).

- [ ] **Step 3: Tüm ilgili testleri çalıştır**

Run: `flutter test test/vad_test.dart test/panel_navigation_test.dart`
Expected: `All tests passed!`

- [ ] **Step 4: Commit**

```bash
git add lib/main.dart lib/vad.dart test/vad_test.dart
git commit -m "chore: format after VAD listening migration"
```

(Eğer Step 1 hiçbir dosyayı değiştirmediyse bu adımı atla — commit edilecek bir şey olmaz.)

- [ ] **Step 5: Manuel doğrulama (cihazda)**

```bash
flutter run
```

1. Furkanca sayfasını aç, "Dinlemeyi Başlat"a bas — durum metni "Dinleniyor..." olmalı.
2. Bir şey söyle — konuşurken durum "Kaydediliyor..." olmalı, sustuktan ~1 saniye sonra "İşleniyor..." ve ardından (varsa) "Yanıt seslendiriliyor..." görünmeli.
3. Sonuç geçmiş listesine eklenmeli; ekranda yalnızca çeviri metni var, Duyulan/Niyet/Eksik Kelimeler yok.
4. Yanıt bittikten hemen sonra tekrar "Dinleniyor..." durumuna dönüp, tuşa basmadan ikinci bir cümle söyleyebilmeli — bu da geçmiş listesine (en üste) eklenmeli.
5. "Dinlemeyi Durdur"a bas — dinleme tamamen kesilmeli, mikrofon kapanmalı.
6. Kayıt sayfasında hiçbir davranış değişikliği olmadığını (önceki turda düzeltilen eş-zamanlı kayıt akışının aynen çalıştığını) doğrula.

## Self-Review Notları

- **Spec kapsaması:** Spec'in "1. Sonuç Görünümü" bölümü Task 2'nin UI'sinde (yalnızca `_corrected`, kayan liste) karşılanıyor; "2. Kontrol Modeli" ve "3. VAD" bölümleri Task 1 + Task 2'de; "4. Hata Yönetimi" (`_handleSegment`'in `catch`/`finally` bloğu, dinlemenin durmaması) karşılanıyor. "Kapsam Dışı" maddeleri (Kayıt sayfası, backend, streaming ASR) hiçbir task'ta değiştirilmiyor.
- **Placeholder taraması:** Yok.
- **Tip/isim tutarlılığı:** `VadSegmenter`/`VadEvent`/`VadEventType`/`pcmToWav` isimleri Task 1'de tanımlandığı gibi Task 2'de birebir kullanıldı.
