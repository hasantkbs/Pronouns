import { describe, it, expect } from 'vitest';
import { calculateRms, pcmToWav, VadSegmenter } from './vad';

function silentChunk(length = 160): Int16Array {
  return new Int16Array(length);
}

function loudChunk(length = 160, amplitude = 10000): Int16Array {
  const arr = new Int16Array(length);
  arr.fill(amplitude);
  return arr;
}

describe('calculateRms', () => {
  it('returns 0 for silence', () => {
    expect(calculateRms(silentChunk())).toBe(0);
  });

  it('returns a value above the default onset threshold for loud audio', () => {
    expect(calculateRms(loudChunk())).toBeGreaterThan(0.012);
  });
});

describe('pcmToWav', () => {
  it('produces a valid 16-bit mono WAV header', async () => {
    const pcm = Int16Array.from([0, 100, -100, 32767]);
    const blob = pcmToWav(pcm, 16000, 1);
    expect(blob.type).toBe('audio/wav');

    const buf = await blob.arrayBuffer();
    const view = new DataView(buf);
    const tag = (offset: number) =>
      String.fromCharCode(
        view.getUint8(offset),
        view.getUint8(offset + 1),
        view.getUint8(offset + 2),
        view.getUint8(offset + 3),
      );

    expect(tag(0)).toBe('RIFF');
    expect(tag(8)).toBe('WAVE');
    expect(view.getUint32(24, true)).toBe(16000); // sample rate
    expect(view.getUint16(22, true)).toBe(1); // channels
    expect(view.getUint16(34, true)).toBe(16); // bits per sample
    expect(view.getUint32(40, true)).toBe(8); // data size = 4 samples * 2 bytes
    expect(buf.byteLength).toBe(52); // 44 byte header + 8 byte data
  });
});

describe('VadSegmenter', () => {
  it('emits started on speech onset and ended after the silence timeout, including pre-roll', () => {
    const seg = new VadSegmenter({
      sampleRate: 16000,
      silenceTimeoutMs: 100,
      preRollMs: 20,
      minSpeechDurationMs: 5,
    });
    const chunkSamples = 160; // 10ms per chunk at 16kHz

    expect(seg.addChunk(silentChunk(chunkSamples))).toBeNull();
    expect(seg.addChunk(silentChunk(chunkSamples))).toBeNull();

    const startEvent = seg.addChunk(loudChunk(chunkSamples));
    expect(startEvent?.type).toBe('started');

    expect(seg.addChunk(loudChunk(chunkSamples))).toBeNull();

    let endEvent = null;
    for (let i = 0; i < 12; i++) {
      const ev = seg.addChunk(silentChunk(chunkSamples));
      if (ev) {
        endEvent = ev;
        break;
      }
    }

    expect(endEvent?.type).toBe('ended');
    expect(endEvent?.pcmData).toBeInstanceOf(Int16Array);
    expect(endEvent!.pcmData!.length).toBeGreaterThan(0);
  });

  it('discards segments shorter than minSpeechDurationMs', () => {
    const seg = new VadSegmenter({
      sampleRate: 16000,
      silenceTimeoutMs: 20,
      minSpeechDurationMs: 100,
    });
    const chunkSamples = 160; // 10ms per chunk

    expect(seg.addChunk(loudChunk(chunkSamples))?.type).toBe('started');
    expect(seg.addChunk(silentChunk(chunkSamples))).toBeNull();
    const ev = seg.addChunk(silentChunk(chunkSamples));
    expect(ev?.type).toBe('discardedTooShort');
  });

  it('caps a segment at maxDurationMs even without silence', () => {
    const seg = new VadSegmenter({
      sampleRate: 16000,
      maxDurationMs: 50,
      minSpeechDurationMs: 5,
    });
    const chunkSamples = 160; // 10ms per chunk

    expect(seg.addChunk(loudChunk(chunkSamples))?.type).toBe('started');
    let endEvent = null;
    for (let i = 0; i < 10; i++) {
      const ev = seg.addChunk(loudChunk(chunkSamples));
      if (ev) {
        endEvent = ev;
        break;
      }
    }
    expect(endEvent?.type).toBe('ended');
  });
});
