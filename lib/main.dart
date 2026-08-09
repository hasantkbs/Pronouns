// ignore_for_file: use_build_context_synchronously
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
import 'pending_recordings.dart';

// ─── Sabitler ────────────────────────────────────────────────────────────────
const String kBaseUrl = 'http://10.10.108.10:8001';
const String kUserId = 'FurkanV1';

// ─── Uygulama ────────────────────────────────────────────────────────────────
void main() => runApp(const PronounsApp());

class PronounsApp extends StatelessWidget {
  const PronounsApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Furkanca',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6C63FF),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
        fontFamily: 'Roboto',
      ),
      home: const MainPage(),
    );
  }
}

// ─── Yardımcı fonksiyonlar ───────────────────────────────────────────────────
String _base() => kBaseUrl.trimRight().replaceAll(RegExp(r'/*$'), '');

Future<void> _ensureMic() async {
  final s = await Permission.microphone.request();
  if (!s.isGranted) throw Exception('Mikrofon izni reddedildi.');
}

/// Aktif bir kayıt oturumunu temsil eder. Mikrofon [_startRecording] ile
/// hemen açılır; [_stopRecording] çağrılana kadar (geri sayım süresince)
/// gerçekten kayıt almaya devam eder.
class _RecordingSession {
  final AudioRecorder recorder;
  const _RecordingSession(this.recorder);
}

Future<_RecordingSession> _startRecording() async {
  final recorder = AudioRecorder();
  final dir = await getTemporaryDirectory();
  final path = '${dir.path}/rec_${DateTime.now().millisecondsSinceEpoch}.wav';

  final ok = await recorder.hasPermission(request: true);
  if (!ok) {
    await recorder.dispose();
    throw Exception('Mikrofon izni alınamadı.');
  }

  await recorder.start(
    const RecordConfig(
        encoder: AudioEncoder.wav, numChannels: 1, sampleRate: 16000),
    path: path,
  );
  return _RecordingSession(recorder);
}

Future<File> _stopRecording(_RecordingSession session) async {
  final out = await session.recorder.stop();
  await session.recorder.dispose();

  if (out == null) throw Exception('Kayıt dosyası oluşturulamadı.');
  return File(out);
}

// ─── Ana Sayfa ───────────────────────────────────────────────────────────────
class MainPage extends StatefulWidget {
  const MainPage({super.key});

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  void _openFurkanca() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const _FurkancaPage()),
    );
  }

  void _openKayit() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const _KayitPage()),
    );
  }

  void _openAyarlar() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const _AyarlarPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final size = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: cs.surface,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const SizedBox(height: 16),
              // ── Logo / başlık ──
              AnimatedBuilder(
                animation: _pulse,
                builder: (_, __) => Container(
                  width: 90,
                  height: 90,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [
                        const Color(0xFF6C63FF)
                            .withOpacity(0.4 + 0.2 * _pulse.value),
                        const Color(0xFF6C63FF).withOpacity(0.05),
                      ],
                    ),
                  ),
                  child: const Icon(Icons.spatial_audio_off_rounded,
                      size: 46, color: Color(0xFF6C63FF)),
                ),
              ),
              const SizedBox(height: 14),
              Text(
                'Furkanca',
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.w800,
                  color: cs.primary,
                  letterSpacing: 1.2,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'Konuşma Asistanı',
                style: TextStyle(
                    fontSize: 14, color: cs.onSurface.withOpacity(0.55)),
              ),
              const SizedBox(height: 36),

              // ── 3 Ana Buton ──
              Expanded(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _BigButton(
                      label: 'Furkanca',
                      sublabel: 'Konuş → Düzeltilmiş metin & ses',
                      icon: Icons.record_voice_over_rounded,
                      gradient: const LinearGradient(
                        colors: [Color(0xFF6C63FF), Color(0xFF3B82F6)],
                      ),
                      width: size.width - 48,
                      onTap: _openFurkanca,
                    ),
                    const SizedBox(height: 18),
                    _BigButton(
                      label: 'Kayıt',
                      sublabel: 'Kelime & ses kaydı yükle',
                      icon: Icons.mic_rounded,
                      gradient: const LinearGradient(
                        colors: [Color(0xFF10B981), Color(0xFF059669)],
                      ),
                      width: size.width - 48,
                      onTap: _openKayit,
                    ),
                    const SizedBox(height: 18),
                    _BigButton(
                      label: 'Ayarlar',
                      sublabel: 'Model tercihi & ince ayar',
                      icon: Icons.tune_rounded,
                      gradient: const LinearGradient(
                        colors: [Color(0xFFF59E0B), Color(0xFFEF4444)],
                      ),
                      width: size.width - 48,
                      onTap: _openAyarlar,
                    ),
                  ],
                ),
              ),

              // ── Alt bilgi ──
              const SizedBox(height: 12),
              Text(
                'Kullanıcı: $kUserId  •  Sunucu: $kBaseUrl',
                style: TextStyle(
                  fontSize: 10,
                  color: cs.onSurface.withOpacity(0.35),
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── Büyük Buton ─────────────────────────────────────────────────────────────
class _BigButton extends StatelessWidget {
  final String label;
  final String sublabel;
  final IconData icon;
  final Gradient gradient;
  final double width;
  final VoidCallback onTap;

  const _BigButton({
    required this.label,
    required this.sublabel,
    required this.icon,
    required this.gradient,
    required this.width,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: width,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
        decoration: BoxDecoration(
          gradient: gradient,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.25),
              blurRadius: 16,
              offset: const Offset(0, 6),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 52,
              height: 52,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.2),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon, color: Colors.white, size: 28),
            ),
            const SizedBox(width: 18),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(label,
                      style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w800,
                          color: Colors.white)),
                  const SizedBox(height: 3),
                  Text(sublabel,
                      style: TextStyle(
                          fontSize: 12, color: Colors.white.withOpacity(0.82))),
                ],
              ),
            ),
            const Icon(Icons.arrow_forward_ios_rounded,
                color: Colors.white54, size: 16),
          ],
        ),
      ),
    );
  }
}

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
                            fontSize: 13, color: cs.onSurface.withOpacity(0.4)),
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
      if (!mounted) return;
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

// ════════════════════════════════════════════════════════════════════════════
// 3) AYARLAR PANELİ
// ════════════════════════════════════════════════════════════════════════════
class _AyarlarPage extends StatefulWidget {
  const _AyarlarPage();

  @override
  State<_AyarlarPage> createState() => _AyarlarPageState();
}

class _AyarlarPageState extends State<_AyarlarPage> {
  bool _loading = true;
  bool _saving = false;
  bool _training = false;
  String? _error;

  // Model bilgisi
  String _modelInfo = '';

  // Ayarlar
  String _selectedModel = 'Furkan';
  String _selectedAlgo = 'lora';
  bool _selfLearning = true;
  double _learningRate = 0.0001;
  int _epochs = 3;
  int _batchSize = 8;
  String _trainStatus = '';

  final List<String> _modelOptions = ['Furkan', 'base', 'large'];
  final List<String> _algoOptions = ['lora', 'qlora', 'full'];

  @override
  void initState() {
    super.initState();
    _fetchSettings();
  }

  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  Future<void> _fetchSettings() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      // GET /settings
      final res = await http.get(
        Uri.parse('${_base()}/settings?user_id=$kUserId'),
      );
      if (res.statusCode == 200) {
        final d = jsonDecode(res.body) as Map<String, dynamic>;
        setState(() {
          _selectedModel = d['model'] ?? _selectedModel;
          _selectedAlgo = d['algorithm'] ?? _selectedAlgo;
          _selfLearning = d['self_learning'] ?? _selfLearning;
          _learningRate =
              (d['learning_rate'] as num?)?.toDouble() ?? _learningRate;
          _epochs = d['epochs'] is int
              ? d['epochs'] as int
              : int.tryParse('${d['epochs']}') ?? _epochs;
          _batchSize = d['batch_size'] is int
              ? d['batch_size'] as int
              : int.tryParse('${d['batch_size']}') ?? _batchSize;
        });
      }

      // GET /model/info
      final infoRes = await http.get(
        Uri.parse('${_base()}/model/info?user_id=$kUserId'),
      );
      if (infoRes.statusCode == 200) {
        final info = jsonDecode(infoRes.body) as Map<String, dynamic>;
        setState(() {
          _modelInfo =
              'Model: ${info['model'] ?? '-'}  •  WER: ${info['wer'] ?? '-'}  •  Veri: ${info['sample_count'] ?? '-'} ses';
        });
      }
    } catch (e) {
      setState(() => _error = 'Ayarlar yüklenemedi: $e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _saveSettings() async {
    setState(() => _saving = true);
    try {
      // POST /settings
      final res = await http.post(
        Uri.parse('${_base()}/settings'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_id': kUserId,
          'model': _selectedModel,
          'algorithm': _selectedAlgo,
          'self_learning': _selfLearning,
          'learning_rate': _learningRate,
          'epochs': _epochs,
          'batch_size': _batchSize,
        }),
      );
      if (res.statusCode == 200) {
        _snack('Ayarlar kaydedildi.');
      } else {
        _snack('Kayıt hatası: ${res.statusCode}');
      }
    } catch (e) {
      _snack('Hata: $e');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Future<void> _startFineTune() async {
    setState(() {
      _training = true;
      _trainStatus = 'Eğitim başlatılıyor...';
    });
    try {
      // POST /fine-tune
      final res = await http.post(
        Uri.parse('${_base()}/fine-tune'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_id': kUserId,
          'algorithm': _selectedAlgo,
          'learning_rate': _learningRate,
          'epochs': _epochs,
          'batch_size': _batchSize,
        }),
      );
      final d = jsonDecode(res.body) as Map<String, dynamic>;
      setState(() {
        _trainStatus = d['status']?.toString() ??
            (res.statusCode == 200
                ? 'Eğitim kuyruğa alındı.'
                : 'Hata: ${res.statusCode}');
      });
    } catch (e) {
      setState(() => _trainStatus = 'Hata: $e');
    } finally {
      if (mounted)
        setState(() {
          _training = false;
        });
    }
  }

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
                  // Model bilgi kartı
                  if (_modelInfo.isNotEmpty)
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: accent.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: accent.withOpacity(0.3)),
                      ),
                      child: Text(_modelInfo,
                          style: TextStyle(
                              fontSize: 12,
                              color: cs.onSurface.withOpacity(0.8))),
                    ),
                  if (_error != null) ...[
                    const SizedBox(height: 8),
                    Text(_error!,
                        style:
                            const TextStyle(color: Colors.red, fontSize: 12)),
                  ],
                  const SizedBox(height: 20),

                  // ── Model seçimi ──
                  const _SectionHeader(label: 'Model Tercihi'),
                  const SizedBox(height: 10),
                  SegmentedButton<String>(
                    segments: _modelOptions
                        .map((m) => ButtonSegment(value: m, label: Text(m)))
                        .toList(),
                    selected: {_selectedModel},
                    onSelectionChanged: (s) =>
                        setState(() => _selectedModel = s.first),
                  ),
                  const SizedBox(height: 20),

                  // ── Algoritma ──
                  const _SectionHeader(label: 'Fine-Tune Algoritması'),
                  const SizedBox(height: 10),
                  SegmentedButton<String>(
                    segments: _algoOptions
                        .map((a) => ButtonSegment(
                            value: a, label: Text(a.toUpperCase())))
                        .toList(),
                    selected: {_selectedAlgo},
                    onSelectionChanged: (s) =>
                        setState(() => _selectedAlgo = s.first),
                  ),
                  const SizedBox(height: 20),

                  // ── Self-learning ──
                  const _SectionHeader(label: 'Kendi Kendine Öğrenme'),
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Self-learning aktif'),
                    subtitle: Text(
                      _selfLearning
                          ? 'Yeni kayıtlar otomatik modele eklenir'
                          : 'Manuel eğitim gerektirir',
                      style: TextStyle(
                          fontSize: 12, color: cs.onSurface.withOpacity(0.6)),
                    ),
                    value: _selfLearning,
                    activeThumbColor: accent,
                    onChanged: (v) => setState(() => _selfLearning = v),
                  ),
                  const SizedBox(height: 12),

                  // ── Hiperparametreler ──
                  const _SectionHeader(label: 'Hiperparametreler'),
                  const SizedBox(height: 10),

                  // Learning rate
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Learning Rate'),
                      Text(
                        _learningRate.toStringAsExponential(1),
                        style: const TextStyle(fontWeight: FontWeight.w700),
                      ),
                    ],
                  ),
                  Slider(
                    value: _learningRate,
                    min: 0.000001,
                    max: 0.001,
                    divisions: 20,
                    activeColor: accent,
                    onChanged: (v) => setState(() => _learningRate = v),
                  ),

                  // Epochs
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Epoch Sayısı'),
                      Text('$_epochs',
                          style: const TextStyle(fontWeight: FontWeight.w700)),
                    ],
                  ),
                  Slider(
                    value: _epochs.toDouble(),
                    min: 1,
                    max: 20,
                    divisions: 19,
                    activeColor: accent,
                    onChanged: (v) => setState(() => _epochs = v.round()),
                  ),

                  // Batch size
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Batch Size'),
                      Text('$_batchSize',
                          style: const TextStyle(fontWeight: FontWeight.w700)),
                    ],
                  ),
                  Slider(
                    value: _batchSize.toDouble(),
                    min: 1,
                    max: 32,
                    divisions: 5,
                    activeColor: accent,
                    onChanged: (v) => setState(() => _batchSize = v.round()),
                  ),

                  const SizedBox(height: 20),

                  // Eğitim durumu
                  if (_trainStatus.isNotEmpty) ...[
                    _ResultTile(
                        label: 'Eğitim Durumu',
                        value: _trainStatus,
                        color: accent),
                    const SizedBox(height: 12),
                  ],

                  // Kaydet butonu
                  FilledButton.icon(
                    style: FilledButton.styleFrom(
                      backgroundColor: accent,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14)),
                    ),
                    onPressed: (_saving || _training) ? null : _saveSettings,
                    icon: Icon(
                        _saving ? Icons.hourglass_top : Icons.save_rounded),
                    label: Text(
                      _saving ? 'Kaydediliyor...' : 'Ayarları Kaydet',
                      style: const TextStyle(
                          fontSize: 15, fontWeight: FontWeight.w700),
                    ),
                  ),
                  const SizedBox(height: 10),

                  // Fine-tune başlat
                  OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      foregroundColor: accent,
                      side: const BorderSide(color: accent),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14)),
                    ),
                    onPressed: (_saving || _training) ? null : _startFineTune,
                    icon: Icon(_training
                        ? Icons.hourglass_top
                        : Icons.model_training_rounded),
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

// ─── Yardımcı widget'lar ─────────────────────────────────────────────────────
class _ResultTile extends StatelessWidget {
  final String label;
  final String value;
  final Color color;

  const _ResultTile({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: TextStyle(
                  fontSize: 11, fontWeight: FontWeight.w700, color: color)),
          const SizedBox(height: 6),
          SelectableText(
            value,
            style: TextStyle(fontSize: 14, color: cs.onSurface),
          ),
        ],
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String label;
  const _SectionHeader({required this.label});

  @override
  Widget build(BuildContext context) {
    return Text(
      label,
      style: TextStyle(
        fontSize: 13,
        fontWeight: FontWeight.w700,
        color: Theme.of(context).colorScheme.primary,
        letterSpacing: 0.5,
      ),
    );
  }
}
