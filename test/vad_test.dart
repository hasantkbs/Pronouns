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
