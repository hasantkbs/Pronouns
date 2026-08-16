import { useEffect, useRef, useState } from 'react';
import { resolveUrl, translateSpeech } from '../lib/api';
import { startAudioCapture, type AudioCaptureHandle } from '../lib/audioCapture';
import { VadSegmenter, pcmToWav } from '../lib/vad';

type ListenState = 'idle' | 'listening' | 'capturing' | 'uploading' | 'speaking' | 'error' | 'paused';

interface TranslationEntry {
  text: string;
  time: Date;
}

export default function FurkancaPage() {
  const [state, setState] = useState<ListenState>('idle');
  const [paused, setPaused] = useState(false);
  const [history, setHistory] = useState<TranslationEntry[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const captureRef = useRef<AudioCaptureHandle | null>(null);
  const segmenterRef = useRef<VadSegmenter | null>(null);
  const stateRef = useRef<ListenState>('idle');
  const pausedRef = useRef(false);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    pausedRef.current = paused;
    if (paused) {
      setState('paused');
    } else if (stateRef.current === 'paused') {
      setState('listening');
    }
  }, [paused]);

  useEffect(() => {
    let cancelled = false;

    function handleChunk(chunk: Int16Array) {
      if (pausedRef.current) return;
      // Kendi çaldığı düzeltilmiş sesi mikrofonla yeniden algılayıp
      // geri besleme döngüsüne girmemesi için yükleme/konuşma sırasında
      // VAD'a chunk beslenmiyor.
      if (stateRef.current === 'uploading' || stateRef.current === 'speaking') return;

      const segmenter = segmenterRef.current;
      const capture = captureRef.current;
      if (!segmenter || !capture) return;
      const event = segmenter.addChunk(chunk);
      if (!event) return;

      if (event.type === 'started') {
        setState('capturing');
      } else if (event.type === 'discardedTooShort') {
        setState('listening');
      } else if (event.type === 'ended' && event.pcmData) {
        setState('uploading');
        void handleSegment(event.pcmData, capture.sampleRate);
      }
    }

    async function start() {
      try {
        const handle = await startAudioCapture(handleChunk);
        if (cancelled) {
          handle.stop();
          return;
        }
        captureRef.current = handle;
        segmenterRef.current = new VadSegmenter({ sampleRate: handle.sampleRate });
        setState('listening');
      } catch (e) {
        setErrorMessage(`Mikrofon erişimi reddedildi: ${(e as Error).message}`);
        setState('error');
      }
    }

    void start();
    return () => {
      cancelled = true;
      captureRef.current?.stop();
    };
  }, []);

  async function handleSegment(pcm: Int16Array, sampleRate: number) {
    try {
      const wavBlob = pcmToWav(pcm, sampleRate, 1);
      const result = await translateSpeech(wavBlob);

      if (result.rejected) {
        setState(pausedRef.current ? 'paused' : 'listening');
        return;
      }

      const corrected = result.response_text?.trim() || '(boş yanıt)';
      setHistory((prev) => [{ text: corrected, time: new Date() }, ...prev]);

      if (result.audio_url) {
        setState('speaking');
        const audio = new Audio(resolveUrl(result.audio_url));
        await new Promise<void>((resolve) => {
          audio.onended = () => resolve();
          audio.onerror = () => resolve();
          audio.play().catch(() => resolve());
        });
      }
      setState(pausedRef.current ? 'paused' : 'listening');
    } catch (e) {
      setErrorMessage(`Hata: ${(e as Error).message}`);
      setState(pausedRef.current ? 'paused' : 'listening');
    }
  }

  return (
    <main>
      <h1>Furkanca</h1>
      <p>Durum: {state}</p>
      <button onClick={() => setPaused((p) => !p)}>
        {paused ? 'Dinlemeye Devam Et' : 'Dinlemeyi Duraklat'}
      </button>
      {errorMessage && <p role="alert">{errorMessage}</p>}
      <ul>
        {history.map((entry, i) => (
          <li key={i}>
            {entry.time.toLocaleTimeString()} — {entry.text}
          </li>
        ))}
      </ul>
    </main>
  );
}
