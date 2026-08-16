export function calculateRms(samples: Int16Array): number {
  if (samples.length === 0) return 0;
  let sumSquares = 0;
  for (let i = 0; i < samples.length; i++) {
    const normalized = samples[i] / 32768;
    sumSquares += normalized * normalized;
  }
  return Math.sqrt(sumSquares / samples.length);
}

export function pcmChunkDurationMs(samples: Int16Array, sampleRate: number): number {
  return (samples.length / sampleRate) * 1000;
}

export function pcmToWav(pcm: Int16Array, sampleRate: number, numChannels = 1): Blob {
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = pcm.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeString(offset: number, str: string) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < pcm.length; i++) {
    view.setInt16(offset, pcm[i], true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

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
