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
