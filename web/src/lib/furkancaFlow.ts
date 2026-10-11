export type Phase = 'idle' | 'starting' | 'recording' | 'uploading' | 'speaking';

export interface HistoryEntry {
  text: string;
  time: Date;
}

export interface FlowState {
  phase: Phase;
  error: string | null;
  history: HistoryEntry[];
}

export type FlowEvent =
  | { type: 'START' }
  | { type: 'MIC_READY' }
  | { type: 'MIC_FAILED'; message: string }
  | { type: 'TOO_SHORT' }
  | { type: 'STOP' }
  | { type: 'RESULT'; text: string; hasAudio: boolean; time: Date }
  | { type: 'REJECTED' }
  | { type: 'FAILED'; message: string }
  | { type: 'AUDIO_DONE' };

export const TOO_SHORT_MESSAGE = 'Çok kısa, tekrar dene';
export const REJECTED_MESSAGE = 'Ses tanınmadı, tekrar dene';

export const initialFlowState: FlowState = { phase: 'idle', error: null, history: [] };

/** Dugme yalnizca bostayken (baslat) ve kayittayken (bitir) basilabilir. */
export function buttonEnabled(phase: Phase): boolean {
  return phase === 'idle' || phase === 'recording';
}

export function formatElapsed(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

/**
 * Furkanca sayfasinin durum makinesi. Gecersiz durumda gelen olay yok sayilir
 * (ornegin islenirken ikinci bir START yeni kayit baslatmaz).
 */
export function flowReducer(state: FlowState, event: FlowEvent): FlowState {
  switch (event.type) {
    case 'START':
      return state.phase === 'idle' ? { ...state, phase: 'starting', error: null } : state;
    case 'MIC_READY':
      return state.phase === 'starting' ? { ...state, phase: 'recording' } : state;
    case 'MIC_FAILED':
      return state.phase === 'starting'
        ? { ...state, phase: 'idle', error: `Mikrofon açılamadı: ${event.message}` }
        : state;
    case 'TOO_SHORT':
      return state.phase === 'recording' ? { ...state, phase: 'idle', error: TOO_SHORT_MESSAGE } : state;
    case 'STOP':
      return state.phase === 'recording' ? { ...state, phase: 'uploading' } : state;
    case 'RESULT':
      return state.phase === 'uploading'
        ? {
            ...state,
            phase: event.hasAudio ? 'speaking' : 'idle',
            history: [{ text: event.text, time: event.time }, ...state.history],
          }
        : state;
    case 'REJECTED':
      return state.phase === 'uploading' ? { ...state, phase: 'idle', error: REJECTED_MESSAGE } : state;
    case 'FAILED':
      return state.phase === 'uploading' ? { ...state, phase: 'idle', error: event.message } : state;
    case 'AUDIO_DONE':
      return state.phase === 'speaking' ? { ...state, phase: 'idle' } : state;
  }
}
