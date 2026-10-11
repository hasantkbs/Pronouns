import { describe, it, expect } from 'vitest';
import { calculateRms, pcmChunkDurationMs, pcmToWav } from './wav';

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

describe('pcmChunkDurationMs', () => {
  it('converts a sample count to milliseconds', () => {
    expect(pcmChunkDurationMs(new Int16Array(1600), 16000)).toBe(100);
  });
});
