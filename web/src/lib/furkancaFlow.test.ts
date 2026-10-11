import { describe, it, expect } from 'vitest';
import {
  buttonEnabled,
  flowReducer,
  formatElapsed,
  initialFlowState,
  REJECTED_MESSAGE,
  TOO_SHORT_MESSAGE,
  type FlowEvent,
  type FlowState,
} from './furkancaFlow';

const T = new Date('2026-10-11T12:00:00Z');

function run(...events: FlowEvent[]): FlowState {
  return events.reduce(flowReducer, initialFlowState);
}

describe('flowReducer', () => {
  it('walks the happy path and returns to idle after the audio', () => {
    const states = [
      { type: 'START' },
      { type: 'MIC_READY' },
      { type: 'STOP' },
      { type: 'RESULT', text: 'Ben geldim.', hasAudio: true, time: T },
      { type: 'AUDIO_DONE' },
    ].reduce<FlowState[]>(
      (acc, event) => [...acc, flowReducer(acc[acc.length - 1], event as FlowEvent)],
      [initialFlowState],
    );
    expect(states.map((s) => s.phase)).toEqual(['idle', 'starting', 'recording', 'uploading', 'speaking', 'idle']);
    expect(states[states.length - 1].history).toEqual([{ text: 'Ben geldim.', time: T }]);
  });

  it('skips speaking when the result has no audio', () => {
    const state = run(
      { type: 'START' },
      { type: 'MIC_READY' },
      { type: 'STOP' },
      { type: 'RESULT', text: 'Ben geldim.', hasAudio: false, time: T },
    );
    expect(state.phase).toBe('idle');
    expect(state.history).toHaveLength(1);
  });

  it('puts the newest result first', () => {
    const first = run(
      { type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' },
      { type: 'RESULT', text: 'Bir.', hasAudio: false, time: T },
    );
    const second = [
      { type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' },
      { type: 'RESULT', text: 'İki.', hasAudio: false, time: T },
    ].reduce((s, e) => flowReducer(s, e as FlowEvent), first);
    expect(second.history.map((h) => h.text)).toEqual(['İki.', 'Bir.']);
  });

  it('shows a too-short message and returns to idle', () => {
    const state = run({ type: 'START' }, { type: 'MIC_READY' }, { type: 'TOO_SHORT' });
    expect(state).toMatchObject({ phase: 'idle', error: TOO_SHORT_MESSAGE });
  });

  it('shows the microphone error and returns to idle', () => {
    const state = run({ type: 'START' }, { type: 'MIC_FAILED', message: 'Permission denied' });
    expect(state).toMatchObject({ phase: 'idle', error: 'Mikrofon açılamadı: Permission denied' });
  });

  it('shows server errors instead of silently going back', () => {
    const state = run(
      { type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' },
      { type: 'FAILED', message: 'Konuşma anlaşılamadı.' },
    );
    expect(state).toMatchObject({ phase: 'idle', error: 'Konuşma anlaşılamadı.' });
  });

  it('shows a message when the server rejects the speaker', () => {
    const state = run({ type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' }, { type: 'REJECTED' });
    expect(state).toMatchObject({ phase: 'idle', error: REJECTED_MESSAGE });
  });

  it('clears the previous error on the next start', () => {
    const failed = run({ type: 'START' }, { type: 'MIC_READY' }, { type: 'TOO_SHORT' });
    expect(flowReducer(failed, { type: 'START' }).error).toBeNull();
  });

  it('ignores START while uploading or speaking', () => {
    const uploading = run({ type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' });
    expect(flowReducer(uploading, { type: 'START' })).toBe(uploading);
    const speaking = flowReducer(uploading, { type: 'RESULT', text: 'X.', hasAudio: true, time: T });
    expect(flowReducer(speaking, { type: 'START' })).toBe(speaking);
  });

  it('ignores a late STOP after the recording already ended', () => {
    const uploading = run({ type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' });
    expect(flowReducer(uploading, { type: 'STOP' })).toBe(uploading);
  });
});

describe('buttonEnabled', () => {
  it('is enabled only while idle or recording', () => {
    expect(buttonEnabled('idle')).toBe(true);
    expect(buttonEnabled('recording')).toBe(true);
    expect(buttonEnabled('starting')).toBe(false);
    expect(buttonEnabled('uploading')).toBe(false);
    expect(buttonEnabled('speaking')).toBe(false);
  });
});

describe('formatElapsed', () => {
  it('formats milliseconds as m:ss', () => {
    expect(formatElapsed(0)).toBe('0:00');
    expect(formatElapsed(3400)).toBe('0:03');
    expect(formatElapsed(65000)).toBe('1:05');
  });
});
