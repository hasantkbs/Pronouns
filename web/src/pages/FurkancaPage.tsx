import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { resolveUrl, translateSpeech } from '../lib/api';
import { startAudioCapture, type AudioCaptureHandle } from '../lib/audioCapture';
import { playAudio, PLAYBACK_TIMEOUT_MS, startUnlessCancelled } from '../lib/deadlines';
import { buttonEnabled, flowReducer, formatElapsed, initialFlowState, type Phase } from '../lib/furkancaFlow';
import { ManualRecorder } from '../lib/manualRecorder';
import { pcmToWav } from '../lib/wav';
import AppBar from '../components/AppBar';
import Icon from '../components/Icon';
import { FilledButton, ProgressBar, ResultTile } from '../components/ui';
import { formatClock } from '../lib/uiFormat';

const STATUS_TEXT: Record<Phase, string> = {
  idle: 'Hazır',
  starting: 'Mikrofon açılıyor...',
  recording: 'Kaydediliyor...',
  uploading: 'İşleniyor...',
  speaking: 'Yanıt seslendiriliyor...',
};

export default function FurkancaPage() {
  const [state, dispatch] = useReducer(flowReducer, initialFlowState);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [level, setLevel] = useState(0);

  const captureRef = useRef<AudioCaptureHandle | null>(null);
  const recorderRef = useRef<ManualRecorder | null>(null);

  const stopCapture = useCallback(() => {
    captureRef.current?.stop();
    captureRef.current = null;
  }, []);

  // Sayfadan cikilinca mikrofon kapanir (acilmakta olan mikrofon da: bkz. startUnlessCancelled)
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stopCapture();
    };
  }, [stopCapture]);

  const send = useCallback(async (pcm: Int16Array, sampleRate: number) => {
    try {
      const result = await translateSpeech(pcmToWav(pcm, sampleRate, 1));
      if (result.rejected) {
        dispatch({ type: 'REJECTED' });
        return;
      }
      const text = result.response_text?.trim() || '(boş yanıt)';
      dispatch({ type: 'RESULT', text, hasAudio: Boolean(result.audio_url), time: new Date() });
      if (result.audio_url) {
        await playAudio(new Audio(resolveUrl(result.audio_url)), PLAYBACK_TIMEOUT_MS);
        dispatch({ type: 'AUDIO_DONE' });
      }
    } catch (e) {
      dispatch({ type: 'FAILED', message: (e as Error).message });
    }
  }, []);

  const finishRecording = useCallback(() => {
    const recorder = recorderRef.current;
    const capture = captureRef.current;
    // Ikinci cagri (dugme + 30 sn siniri ayni anda) yok sayilir
    if (!recorder || !capture) return;
    recorderRef.current = null;
    const sampleRate = capture.sampleRate;
    stopCapture();

    const result = recorder.finish();
    if (result.tooShort) {
      dispatch({ type: 'TOO_SHORT' });
      return;
    }
    dispatch({ type: 'STOP' });
    void send(result.pcm, sampleRate);
  }, [send, stopCapture]);

  const startRecording = useCallback(async () => {
    dispatch({ type: 'START' });
    setElapsedMs(0);
    setLevel(0);
    try {
      const handle = await startUnlessCancelled(
        () =>
          startAudioCapture((chunk) => {
            const recorder = recorderRef.current;
            if (!recorder) return;
            const reachedMax = recorder.push(chunk);
            setElapsedMs(recorder.durationMs);
            setLevel(recorder.level);
            if (reachedMax) finishRecording();
          }),
        () => mountedRef.current,
      );
      if (!handle) return;
      captureRef.current = handle;
      recorderRef.current = new ManualRecorder(handle.sampleRate);
      // Dugme ancak mikrofon gercekten acildiktan sonra "Bitir" olur: ilk hece kacmaz
      dispatch({ type: 'MIC_READY' });
    } catch (e) {
      dispatch({ type: 'MIC_FAILED', message: (e as Error).message });
    }
  }, [finishRecording]);

  const recording = state.phase === 'recording';
  const active = state.phase !== 'idle';

  return (
    <main className="page">
      <AppBar title="Furkanca" backTo="/" />
      <div className="page-body">
        <p className="muted">
          Düğmeye dokun, konuş, bitince tekrar dokun — AI konuşmanı düzeltilmiş hâle çevirsin ve yüksek sesle
          okusun.
        </p>
        <div className="row" style={{ color: 'var(--accent)' }}>
          <Icon name={active ? 'graphic_eq' : 'mic_off'} size={18} />
          <span style={{ color: 'var(--on-surface)', fontSize: 13 }}>{STATUS_TEXT[state.phase]}</span>
        </div>
        <FilledButton
          icon={recording ? 'stop' : 'mic'}
          className={`button--tall${recording ? ' button--danger' : ''}`}
          disabled={!buttonEnabled(state.phase)}
          onClick={() => (recording ? finishRecording() : void startRecording())}
        >
          {recording ? `Bitir ${formatElapsed(elapsedMs)}` : 'Konuşmaya Başla'}
        </FilledButton>
        {recording && <ProgressBar value={(Math.min(level, 0.3) / 0.3) * 100} label="Ses seviyesi" />}
        {state.error && <ResultTile label="Hata" value={state.error} tone="error" />}
        {state.history.length === 0 ? (
          <p className="empty">Henüz çeviri yok.</p>
        ) : (
          <ul className="stack" style={{ listStyle: 'none', margin: 0, padding: 0 }}>
            {state.history.map((entry, i) => (
              <li key={i} className="history-card">
                <div className="history-time">{formatClock(entry.time)}</div>
                <div className="history-text">{entry.text}</div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </main>
  );
}
