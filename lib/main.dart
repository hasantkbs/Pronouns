// ignore_for_file: use_build_context_synchronously
import 'dart:async';
import 'dart:convert';
import 'dart:io' show File;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:permission_handler/permission_handler.dart';

// ─── Sabitler ────────────────────────────────────────────────────────────────
const String kBaseUrl = 'http://91.241.50.187:8001';
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

Future<File> _recordWav(int seconds) async {
  final recorder = AudioRecorder();
  final dir = await getTemporaryDirectory();
  final path = '${dir.path}/rec_${DateTime.now().millisecondsSinceEpoch}.wav';

  final ok = await recorder.hasPermission(request: true);
  if (!ok) {
    await recorder.dispose();
    throw Exception('Mikrofon izni alınamadı.');
  }

  await recorder.start(
    const RecordConfig(encoder: AudioEncoder.wav, numChannels: 1, sampleRate: 16000),
    path: path,
  );
  await Future.delayed(Duration(seconds: seconds));
  final out = await recorder.stop();
  await recorder.dispose();

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
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => const _FurkancaPanel(),
    );
  }

  void _openKayit() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => const _KayitPanel(),
    );
  }

  void _openAyarlar() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => const _AyarlarPanel(),
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
                style: TextStyle(fontSize: 14, color: cs.onSurface.withOpacity(0.55)),
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
                          fontSize: 12,
                          color: Colors.white.withOpacity(0.82))),
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

// ════════════════════════════════════════════════════════════════════════════
// 1) FURKANCA PANELİ
// ════════════════════════════════════════════════════════════════════════════
class _FurkancaPanel extends StatefulWidget {
  const _FurkancaPanel();

  @override
  State<_FurkancaPanel> createState() => _FurkancaPanelState();
}

class _FurkancaPanelState extends State<_FurkancaPanel> {
  final AudioPlayer _player = AudioPlayer();
  int _seconds = 4;
  bool _busy = false;
  bool _recording = false;
  int _remaining = 0;

  String? _recognized;
  String? _corrected;
  String? _intent;
  List<String> _missing = const [];

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  Future<void> _run() async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _recording = true;
      _remaining = _seconds;
      _recognized = null;
      _corrected = null;
      _intent = null;
      _missing = const [];
    });

    try {
      await _ensureMic();

      // Geri sayım
      for (var i = _seconds; i > 0; i--) {
        if (!mounted) return;
        setState(() => _remaining = i);
        await Future.delayed(const Duration(seconds: 1));
      }

      final file = await _recordWav(_seconds);
      if (!mounted) return;
      setState(() {
        _recording = false;
        _remaining = 0;
      });

      // POST /translate
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
      final recognized = d['recognized_text']?.toString() ?? '';
      final corrected = d['response_text']?.toString() ?? '';
      final intent = d['intent']?.toString() ?? '';
      final missing = (d['missing_words'] is List)
          ? (d['missing_words'] as List).map((e) => e.toString()).toList()
          : <String>[];

      setState(() {
        _recognized = recognized.isEmpty ? null : recognized;
        _corrected = corrected.isEmpty ? null : corrected;
        _intent = intent.isEmpty ? null : intent;
        _missing = missing;
      });

      final audioUrl = d['audio_url'];
      if (audioUrl is String && audioUrl.isNotEmpty) {
        final full =
            audioUrl.startsWith('http') ? audioUrl : '${_base()}$audioUrl';
        await _player.play(UrlSource(full));
      }
    } catch (e) {
      _snack('Hata: $e');
    } finally {
      if (mounted) setState(() { _busy = false; _recording = false; _remaining = 0; });
    }
  }

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
          Text(
            'Konuş, AI konuşmanı düzeltilmiş hâle çevirsin ve yüksek sesle okusun.',
            style: TextStyle(
                fontSize: 13,
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.65)),
          ),
          const SizedBox(height: 20),

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
            min: 2,
            max: 12,
            divisions: 10,
            label: '$_seconds sn',
            activeColor: accent,
            onChanged:
                _busy ? null : (v) => setState(() => _seconds = v.round()),
          ),
          const SizedBox(height: 8),

          // Progress bar
          if (_recording) ...[
            LinearProgressIndicator(
              value: _seconds == 0
                  ? null
                  : (_seconds - _remaining) / _seconds,
              color: accent,
            ),
            const SizedBox(height: 6),
            Text(
              'Kayıt alınıyor... $_remaining sn kaldı',
              style: const TextStyle(fontSize: 12),
            ),
            const SizedBox(height: 12),
          ],

          // Ana buton
          FilledButton.icon(
            style: FilledButton.styleFrom(
              backgroundColor: accent,
              padding: const EdgeInsets.symmetric(vertical: 16),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14)),
            ),
            onPressed: _busy ? null : _run,
            icon: Icon(_busy ? Icons.hourglass_top : Icons.mic_rounded),
            label: Text(
              _busy
                  ? (_recording ? 'Kaydediliyor...' : 'İşleniyor...')
                  : 'Konuşmaya Başla',
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
            ),
          ),

          const SizedBox(height: 24),

          // Sonuçlar
          if (_recognized != null) _ResultTile(label: 'Duyulan', value: _recognized!, color: Colors.blue),
          if (_intent != null) _ResultTile(label: 'Niyet', value: _intent!, color: Colors.purple),
          if (_corrected != null) _ResultTile(label: 'Düzeltilmiş', value: _corrected!, color: const Color(0xFF10B981)),
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

// ════════════════════════════════════════════════════════════════════════════
// 2) KAYIT PANELİ
// ════════════════════════════════════════════════════════════════════════════
class _KayitPanel extends StatefulWidget {
  const _KayitPanel();

  @override
  State<_KayitPanel> createState() => _KayitPanelState();
}

class _KayitPanelState extends State<_KayitPanel> {
  String _setFile = 'wordSet.txt';
  List<String> _sets = const [];

  String? _word;
  int _rep = 1;
  int _currentCount = 0;
  int _idealReps = 0;
  int _completedWords = 0;
  int _totalWords = 0;

  int _seconds = 2;
  bool _busy = false;
  bool _recording = false;
  int _remaining = 0;
  String? _lastStatus;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    await _loadSets();
    await Future.wait([_refreshWord(), _refreshProgress()]);
  }

  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

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
      final uri = Uri.parse(
          '${_base()}/collect/next-word?user_id=$kUserId&set_file=$_setFile');
      final res = await http.get(uri);
      final d = jsonDecode(res.body) as Map<String, dynamic>;
      if (d['error'] != null) throw Exception(d['error']);
      if (!mounted) return;
      setState(() {
        _word = d['word']?.toString();
        _rep = _parseInt(d['rep']) ?? 1;
        _currentCount = _parseInt(d['current_count']) ?? 0;
        _idealReps = _parseInt(d['ideal_repetitions']) ?? 0;
        _totalWords = _parseInt(d['total_words']) ?? 0;
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

  Future<void> _upload() async {
    if (_busy) return;
    final word = _word?.trim() ?? '';
    if (word.isEmpty) { _snack('Hedef kelime alınamadı.'); return; }

    setState(() { _busy = true; _recording = true; _remaining = _seconds; _lastStatus = null; });

    try {
      await _ensureMic();
      for (var i = _seconds; i > 0; i--) {
        if (!mounted) return;
        setState(() => _remaining = i);
        await Future.delayed(const Duration(seconds: 1));
      }

      final file = await _recordWav(_seconds);
      if (!mounted) return;
      setState(() { _recording = false; _remaining = 0; });

      // POST /record
      final req = http.MultipartRequest('POST', Uri.parse('${_base()}/record'))
        ..fields['user_id'] = kUserId
        ..fields['word'] = word
        ..fields['rep'] = _rep.toString()
        ..files.add(await http.MultipartFile.fromPath('audio', file.path));

      final res = await req.send();
      final respBody = await res.stream.bytesToString();
      if (res.statusCode != 200) throw Exception('Sunucu: ${res.statusCode} $respBody');

      setState(() => _lastStatus = 'Kaydedildi: $word (rep $_rep)');
      _snack('Yüklendi: $word');
      await Future.wait([_refreshWord(), _refreshProgress()]);
    } catch (e) {
      _snack('Hata: $e');
    } finally {
      if (mounted) setState(() { _busy = false; _recording = false; _remaining = 0; });
    }
  }

  static int? _parseInt(dynamic v) =>
      v is int ? v : int.tryParse(v?.toString() ?? '');

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
          // İlerleme
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('İlerleme: $_completedWords / $_totalWords kelime'),
              Text('${(progress * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(fontWeight: FontWeight.w700)),
            ],
          ),
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
            value: _sets.isEmpty ? null : (_sets.contains(_setFile) ? _setFile : _sets.first),
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
                ? Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.check_circle_rounded,
                          color: accent, size: 28),
                      const SizedBox(width: 10),
                      Text('Tüm kelimeler tamamlandı!',
                          style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: accent)),
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
                        'Tekrar: $_rep  ·  Mevcut: $_currentCount / $_idealReps',
                        style: TextStyle(
                            fontSize: 13,
                            color: cs.onSurface.withOpacity(0.6)),
                      ),
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
            onChanged: _busy ? null : (v) => setState(() => _seconds = v.round()),
          ),

          // Progress bar
          if (_recording) ...[
            LinearProgressIndicator(
              value: _seconds == 0 ? null : (_seconds - _remaining) / _seconds,
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
            onPressed: (_busy || allDone) ? null : _upload,
            icon: Icon(_busy ? Icons.hourglass_top : Icons.cloud_upload_rounded),
            label: Text(
              _busy ? (_recording ? 'Kaydediliyor...' : 'Yükleniyor...') : 'Kaydet ve Yükle',
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
            ),
          ),
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

// ════════════════════════════════════════════════════════════════════════════
// 3) AYARLAR PANELİ
// ════════════════════════════════════════════════════════════════════════════
class _AyarlarPanel extends StatefulWidget {
  const _AyarlarPanel();

  @override
  State<_AyarlarPanel> createState() => _AyarlarPanelState();
}

class _AyarlarPanelState extends State<_AyarlarPanel> {
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
    setState(() { _loading = true; _error = null; });
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
          _learningRate = (d['learning_rate'] as num?)?.toDouble() ?? _learningRate;
          _epochs = d['epochs'] is int ? d['epochs'] as int : int.tryParse('${d['epochs']}') ?? _epochs;
          _batchSize = d['batch_size'] is int ? d['batch_size'] as int : int.tryParse('${d['batch_size']}') ?? _batchSize;
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
    setState(() { _training = true; _trainStatus = 'Eğitim başlatılıyor...'; });
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
            (res.statusCode == 200 ? 'Eğitim kuyruğa alındı.' : 'Hata: ${res.statusCode}');
      });
    } catch (e) {
      setState(() => _trainStatus = 'Hata: $e');
    } finally {
      if (mounted) setState(() { _training = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFFF59E0B);
    final cs = Theme.of(context).colorScheme;

    return _Sheet(
      title: 'Model Ayarları',
      icon: Icons.tune_rounded,
      iconColor: accent,
      body: _loading
          ? const Center(child: Padding(padding: EdgeInsets.all(32), child: CircularProgressIndicator()))
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
                            fontSize: 12, color: cs.onSurface.withOpacity(0.8))),
                  ),
                if (_error != null) ...[
                  const SizedBox(height: 8),
                  Text(_error!, style: const TextStyle(color: Colors.red, fontSize: 12)),
                ],
                const SizedBox(height: 20),

                // ── Model seçimi ──
                _SectionHeader(label: 'Model Tercihi'),
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
                _SectionHeader(label: 'Fine-Tune Algoritması'),
                const SizedBox(height: 10),
                SegmentedButton<String>(
                  segments: _algoOptions
                      .map((a) => ButtonSegment(value: a, label: Text(a.toUpperCase())))
                      .toList(),
                  selected: {_selectedAlgo},
                  onSelectionChanged: (s) =>
                      setState(() => _selectedAlgo = s.first),
                ),
                const SizedBox(height: 20),

                // ── Self-learning ──
                _SectionHeader(label: 'Kendi Kendine Öğrenme'),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Self-learning aktif'),
                  subtitle: Text(
                    _selfLearning
                        ? 'Yeni kayıtlar otomatik modele eklenir'
                        : 'Manuel eğitim gerektirir',
                    style: TextStyle(fontSize: 12, color: cs.onSurface.withOpacity(0.6)),
                  ),
                  value: _selfLearning,
                  activeColor: accent,
                  onChanged: (v) => setState(() => _selfLearning = v),
                ),
                const SizedBox(height: 12),

                // ── Hiperparametreler ──
                _SectionHeader(label: 'Hiperparametreler'),
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
                    Text('$_epochs', style: const TextStyle(fontWeight: FontWeight.w700)),
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
                    Text('$_batchSize', style: const TextStyle(fontWeight: FontWeight.w700)),
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
                  _ResultTile(label: 'Eğitim Durumu', value: _trainStatus, color: accent),
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
                  icon: Icon(_saving ? Icons.hourglass_top : Icons.save_rounded),
                  label: Text(
                    _saving ? 'Kaydediliyor...' : 'Ayarları Kaydet',
                    style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
                  ),
                ),
                const SizedBox(height: 10),

                // Fine-tune başlat
                OutlinedButton.icon(
                  style: OutlinedButton.styleFrom(
                    foregroundColor: accent,
                    side: BorderSide(color: accent),
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
                    style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
                  ),
                ),
              ],
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
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: color)),
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
