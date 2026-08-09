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
