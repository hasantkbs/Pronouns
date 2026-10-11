import { describe, it, expect } from 'vitest';
import { VadSegmenter } from './vad';

function silentChunk(length = 160): Int16Array {
  return new Int16Array(length);
}

function loudChunk(length = 160, amplitude = 10000): Int16Array {
  const arr = new Int16Array(length);
  arr.fill(amplitude);
  return arr;
}

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
