import { calculateRms, pcmChunkDurationMs } from './wav';

export type VadEventType = 'started' | 'ended' | 'discardedTooShort';

export interface VadEvent {
  type: VadEventType;
  pcmData?: Int16Array;
}

export interface VadSegmenterOptions {
  onsetThreshold?: number;
  silenceTimeoutMs?: number;
  maxDurationMs?: number;
  minSpeechDurationMs?: number;
  preRollMs?: number;
  sampleRate?: number;
}

export class VadSegmenter {
  private onsetThreshold: number;
  private silenceTimeoutMs: number;
  private maxDurationMs: number;
  private minSpeechDurationMs: number;
  private preRollMs: number;
  private sampleRate: number;

  private preRoll: Int16Array[] = [];
  private preRollBufferedMs = 0;

  private capturing = false;
  private buffer: number[] = [];
  private capturedDurationMs = 0;
  private speechDurationMs = 0;
  private silenceElapsedMs = 0;

  constructor(options: VadSegmenterOptions = {}) {
    this.onsetThreshold = options.onsetThreshold ?? 0.012;
    this.silenceTimeoutMs = options.silenceTimeoutMs ?? 1000;
    this.maxDurationMs = options.maxDurationMs ?? 12000;
    this.minSpeechDurationMs = options.minSpeechDurationMs ?? 300;
    this.preRollMs = options.preRollMs ?? 300;
    this.sampleRate = options.sampleRate ?? 16000;
  }

  addChunk(chunk: Int16Array): VadEvent | null {
    const chunkDurationMs = pcmChunkDurationMs(chunk, this.sampleRate);
    const isSpeech = calculateRms(chunk) >= this.onsetThreshold;

    if (!this.capturing) {
      if (!isSpeech) {
        this.preRoll.push(chunk);
        this.preRollBufferedMs += chunkDurationMs;
        while (this.preRollBufferedMs > this.preRollMs && this.preRoll.length > 0) {
          const removed = this.preRoll.shift()!;
          this.preRollBufferedMs -= pcmChunkDurationMs(removed, this.sampleRate);
        }
        return null;
      }

      this.capturing = true;
      this.buffer = [];
      for (const c of this.preRoll) {
        this.buffer.push(...c);
      }
      this.buffer.push(...chunk);
      this.capturedDurationMs = this.preRollBufferedMs + chunkDurationMs;
      this.speechDurationMs = chunkDurationMs;
      this.silenceElapsedMs = 0;
      this.preRoll = [];
      this.preRollBufferedMs = 0;
      return { type: 'started' };
    }

    this.buffer.push(...chunk);
    this.capturedDurationMs += chunkDurationMs;
    if (isSpeech) {
      this.speechDurationMs += chunkDurationMs;
      this.silenceElapsedMs = 0;
    } else {
      this.silenceElapsedMs += chunkDurationMs;
    }

    const shouldEnd =
      this.silenceElapsedMs >= this.silenceTimeoutMs || this.capturedDurationMs >= this.maxDurationMs;
    if (!shouldEnd) return null;

    this.capturing = false;
    const pcm = Int16Array.from(this.buffer);
    const tooShort = this.speechDurationMs < this.minSpeechDurationMs;
    this.buffer = [];
    this.capturedDurationMs = 0;
    this.speechDurationMs = 0;
    this.silenceElapsedMs = 0;

    return tooShort ? { type: 'discardedTooShort' } : { type: 'ended', pcmData: pcm };
  }
}
