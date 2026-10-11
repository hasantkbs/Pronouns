import { calculateRms } from './wav';

export const MIN_RECORDING_MS = 500;
export const MAX_RECORDING_MS = 30000;

export type FinishResult = { tooShort: true } | { tooShort: false; pcm: Int16Array };

/**
 * Dokun-baslat / dokun-bitir kaydinin saf mantigi: PCM chunk'larini biriktirir,
 * sure ve ses seviyesini hesaplar. Mikrofona dokunmaz (test edilebilir).
 */
export class ManualRecorder {
  private chunks: Int16Array[] = [];
  private totalSamples = 0;
  private lastLevel = 0;

  constructor(
    private readonly sampleRate: number,
    private readonly minDurationMs = MIN_RECORDING_MS,
    private readonly maxDurationMs = MAX_RECORDING_MS,
  ) {}

  /** Chunk ekler; azami sureye ulasildiysa true doner (kayit otomatik bitmeli). */
  push(chunk: Int16Array): boolean {
    this.chunks.push(chunk);
    this.totalSamples += chunk.length;
    this.lastLevel = calculateRms(chunk);
    return this.durationMs >= this.maxDurationMs;
  }

  get durationMs(): number {
    return (this.totalSamples / this.sampleRate) * 1000;
  }

  /** Son chunk'in RMS seviyesi (0..1), ses seviyesi cubugu icin. */
  get level(): number {
    return this.lastLevel;
  }

  finish(): FinishResult {
    if (this.durationMs < this.minDurationMs) return { tooShort: true };
    const pcm = new Int16Array(this.totalSamples);
    let offset = 0;
    for (const chunk of this.chunks) {
      pcm.set(chunk, offset);
      offset += chunk.length;
    }
    return { tooShort: false, pcm };
  }
}
