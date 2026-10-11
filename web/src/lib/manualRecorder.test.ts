import { describe, it, expect } from 'vitest';
import { ManualRecorder, MAX_RECORDING_MS, MIN_RECORDING_MS } from './manualRecorder';

const RATE = 16000;

function chunk(ms: number, amplitude = 0): Int16Array {
  return new Int16Array((RATE * ms) / 1000).fill(amplitude);
}

describe('ManualRecorder', () => {
  it('uses 0.5 s and 30 s limits by default', () => {
    expect(MIN_RECORDING_MS).toBe(500);
    expect(MAX_RECORDING_MS).toBe(30000);
  });

  it('accumulates chunks in order into one PCM buffer', () => {
    const recorder = new ManualRecorder(RATE);
    recorder.push(chunk(300, 1));
    recorder.push(chunk(300, 2));
    const result = recorder.finish();
    expect(result.tooShort).toBe(false);
    if (result.tooShort) return;
    expect(result.pcm.length).toBe(RATE * 0.6);
    expect(result.pcm[0]).toBe(1);
    expect(result.pcm[result.pcm.length - 1]).toBe(2);
  });

  it('tracks duration in milliseconds', () => {
    const recorder = new ManualRecorder(RATE);
    recorder.push(chunk(250));
    recorder.push(chunk(250));
    expect(recorder.durationMs).toBe(500);
  });

  it('reports the level of the latest chunk', () => {
    const recorder = new ManualRecorder(RATE);
    recorder.push(chunk(100, 10000));
    expect(recorder.level).toBeGreaterThan(0.3);
    recorder.push(chunk(100, 0));
    expect(recorder.level).toBe(0);
  });

  it('marks recordings shorter than the minimum as too short', () => {
    const recorder = new ManualRecorder(RATE);
    recorder.push(chunk(400));
    expect(recorder.finish()).toEqual({ tooShort: true });
  });

  it('treats a recording with no chunks as too short', () => {
    expect(new ManualRecorder(RATE).finish()).toEqual({ tooShort: true });
  });

  it('signals auto-stop once the maximum duration is reached', () => {
    const recorder = new ManualRecorder(RATE, 500, 1000);
    expect(recorder.push(chunk(600))).toBe(false);
    expect(recorder.push(chunk(400))).toBe(true);
  });
});
