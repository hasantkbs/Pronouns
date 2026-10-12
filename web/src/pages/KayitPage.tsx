import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { getNextWord, getProgress, getWordSets, type RecordKind } from '../lib/api';
import { startAudioCapture, type AudioCaptureHandle } from '../lib/audioCapture';
import { startUnlessCancelled } from '../lib/deadlines';
import { formatElapsed } from '../lib/furkancaFlow';
import { ManualRecorder } from '../lib/manualRecorder';
import { pcmToWav } from '../lib/wav';
import { addTake, getAllTakes, getPendingWords, getTakesForWord } from '../lib/pendingRecordingsStore';

const DEFAULT_WORD_SET = 'wordSet.txt';

type SentencePhase = 'idle' | 'starting' | 'recording';

export default function KayitPage() {
  const [kind, setKind] = useState<RecordKind>('word');
  const [sets, setSets] = useState<string[]>([]);
  const [setFile, setSetFile] = useState(DEFAULT_WORD_SET);
  const [word, setWord] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [currentCount, setCurrentCount] = useState(0);
  const [idealReps, setIdealReps] = useState(0);
  const [totalWords, setTotalWords] = useState(0);
  const [completedWords, setCompletedWords] = useState(0);
  const [seconds, setSeconds] = useState(2);
  const [remaining, setRemaining] = useState(0);
  const [recording, setRecording] = useState(false);
  const [sentencePhase, setSentencePhase] = useState<SentencePhase>('idle');
  const [elapsedMs, setElapsedMs] = useState(0);
  const [status, setStatus] = useState<string | null>(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [wordPendingCount, setWordPendingCount] = useState(0);

  // Mod/set degisince onceki istegin gec gelen yaniti yeni modun metnini ezmesin
  const wordRequestRef = useRef(0);
  const captureRef = useRef<AudioCaptureHandle | null>(null);
  const recorderRef = useRef<ManualRecorder | null>(null);

  // Sayfadan cikilinca cumle kaydinin mikrofonu kapanir
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      captureRef.current?.stop();
      captureRef.current = null;
    };
  }, []);

  const unit = kind === 'word' ? 'kelime' : 'cümle';
  const busy = recording || sentencePhase !== 'idle';

  const loadSets = useCallback(async () => {
    try {
      const { sets: s } = await getWordSets(kind);
      setSets(s);
      setSetFile((current) => (s.length > 0 && !s.includes(current) ? s[0] : current));
    } catch {
      // set listesi opsiyonel; varsayılan setFile ile devam edilir
    }
  }, [kind]);

  const refreshWord = useCallback(async () => {
    if (!setFile) return; // cümle setleri henüz yüklenmedi
    const requestId = ++wordRequestRef.current;
    try {
      const pending = Array.from(await getPendingWords(kind));
      const data = await getNextWord(setFile, [...pending, ...skipped], kind);
      if (requestId !== wordRequestRef.current) return;
      setWord(data.word ?? null);
      setDone(data.done === true);
      setCurrentCount(data.current_count ?? 0);
      setIdealReps(data.ideal_repetitions ?? 0);
      setTotalWords((prev) => data.total_words ?? prev);
    } catch (e) {
      setStatus(`${kind === 'word' ? 'Kelime' : 'Cümle'} alınamadı: ${(e as Error).message}`);
    }
  }, [setFile, kind, skipped]);

  const refreshProgress = useCallback(async () => {
    if (!setFile) return;
    try {
      const data = await getProgress(setFile, kind);
      setCompletedWords(data.completed_words ?? 0);
      setTotalWords((prev) => data.total_words ?? prev);
      setIdealReps((prev) => data.ideal_repetitions ?? prev);
    } catch {
      // ilerleme opsiyonel
    }
  }, [setFile, kind]);

  const refreshPendingCounts = useCallback(async () => {
    const all = await getAllTakes();
    setPendingCount(all.length);
    if (word) {
      const forWord = await getTakesForWord(word);
      setWordPendingCount(forWord.length);
    } else {
      setWordPendingCount(0);
    }
  }, [word]);

  useEffect(() => {
    void loadSets();
  }, [loadSets]);

  useEffect(() => {
    void refreshWord();
    void refreshProgress();
  }, [setFile, refreshWord, refreshProgress]);

  useEffect(() => {
    void refreshPendingCounts();
  }, [word, refreshPendingCounts]);

  function changeKind(next: RecordKind) {
    if (next === kind || busy) return;
    wordRequestRef.current++;
    setKind(next);
    // Cümle setleri sunucudan gelince ilki seçilir
    setSetFile(next === 'word' ? DEFAULT_WORD_SET : '');
    setSets([]);
    setSkipped([]);
    setWord(null);
    setDone(false);
    setCompletedWords(0);
    setTotalWords(0);
    setStatus(null);
  }

  async function handleRecord() {
    if (recording || !word) return;
    setRecording(true);
    setStatus(null);
    setRemaining(seconds);

    let handle: Awaited<ReturnType<typeof startAudioCapture>> | null = null;
    try {
      const chunks: Int16Array[] = [];
      handle = await startAudioCapture((chunk) => chunks.push(chunk));

      for (let i = seconds; i > 0; i--) {
        setRemaining(i);
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
      handle.stop();

      const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
      const merged = new Int16Array(totalLength);
      let offset = 0;
      for (const c of chunks) {
        merged.set(c, offset);
        offset += c.length;
      }

      const wavBlob = pcmToWav(merged, handle.sampleRate, 1);
      await addTake(word, wavBlob);
      await refreshPendingCounts();
      setStatus(`Cihazda kaydedildi: ${word}`);
    } catch (e) {
      setStatus(`Hata: ${(e as Error).message}`);
      if (handle) {
        handle.stop();
      }
    } finally {
      setRecording(false);
      setRemaining(0);
    }
  }

  const finishSentence = useCallback(async () => {
    const recorder = recorderRef.current;
    const capture = captureRef.current;
    // Ikinci cagri (dugme + 30 sn siniri ayni anda) yok sayilir
    if (!recorder || !capture || !word) return;
    recorderRef.current = null;
    captureRef.current = null;
    capture.stop();
    setSentencePhase('idle');

    const result = recorder.finish();
    if (result.tooShort) {
      setStatus('Çok kısa, tekrar dene');
      return;
    }
    try {
      await addTake(word, pcmToWav(result.pcm, capture.sampleRate, 1), 'sentence');
      await refreshPendingCounts();
      setStatus('Cihazda kaydedildi');
    } catch (e) {
      setStatus(`Hata: ${(e as Error).message}`);
    }
  }, [word, refreshPendingCounts]);

  async function startSentence() {
    if (sentencePhase !== 'idle' || !word) return;
    setSentencePhase('starting');
    setStatus(null);
    setElapsedMs(0);
    try {
      const handle = await startUnlessCancelled(
        () =>
          startAudioCapture((chunk) => {
            const recorder = recorderRef.current;
            if (!recorder) return;
            const reachedMax = recorder.push(chunk);
            setElapsedMs(recorder.durationMs);
            if (reachedMax) void finishSentence();
          }),
        () => mountedRef.current,
      );
      if (!handle) return;
      captureRef.current = handle;
      recorderRef.current = new ManualRecorder(handle.sampleRate);
      // Dugme ancak mikrofon gercekten acildiktan sonra "Bitir" olur: ilk hece kacmaz
      setSentencePhase('recording');
    } catch (e) {
      setSentencePhase('idle');
      setStatus(`Mikrofon açılamadı: ${(e as Error).message}`);
    }
  }

  function skipSentence() {
    if (!word || busy) return;
    setStatus(null);
    setSkipped((prev) => [...prev, word]);
  }

  const wordFull = word !== null && currentCount + wordPendingCount >= idealReps;
  const repsDone = currentCount + wordPendingCount;

  return (
    <main>
      <h1>Kayıt</h1>
      <p>
        <Link to="/kayit/bekleyen">Bekleyen Kayıtlar ({pendingCount})</Link>
      </p>

      <div role="group" aria-label="Kayıt türü">
        <button
          type="button"
          aria-pressed={kind === 'word'}
          disabled={busy}
          onClick={() => changeKind('word')}
        >
          Kelime
        </button>
        <button
          type="button"
          aria-pressed={kind === 'sentence'}
          disabled={busy}
          onClick={() => changeKind('sentence')}
        >
          Cümle
        </button>
      </div>

      <label>
        {kind === 'word' ? 'Kelime Seti' : 'Cümle Seti'}
        <select
          value={setFile}
          onChange={(e) => {
            setSkipped([]);
            setSetFile(e.target.value);
          }}
          disabled={busy}
        >
          {sets.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>

      <p>
        İlerleme: {completedWords} / {totalWords} {unit}
      </p>

      {word === null ? (
        done && <p>Tüm {unit}ler tamamlandı!</p>
      ) : kind === 'word' ? (
        <>
          <h2>{word}</h2>
          <p>
            {repsDone} / {idealReps} tekrar
          </p>

          <label>
            Kayıt süresi: {seconds}sn
            <input
              type="range"
              min={1}
              max={5}
              value={seconds}
              disabled={recording}
              onChange={(e) => setSeconds(Number(e.target.value))}
            />
          </label>

          <button onClick={handleRecord} disabled={recording || wordFull}>
            {recording ? `Kaydediliyor... ${remaining}` : 'Kaydet'}
          </button>

          {wordFull && (
            <button
              onClick={() => {
                void refreshWord();
                void refreshProgress();
              }}
            >
              Sıradaki Kelime
            </button>
          )}
        </>
      ) : (
        <>
          <h2 style={{ fontSize: '2rem' }}>{word}</h2>
          <p>
            Bu cümle: {repsDone} / {idealReps}
          </p>

          <button
            type="button"
            disabled={sentencePhase === 'starting' || (sentencePhase === 'idle' && wordFull)}
            onClick={() =>
              sentencePhase === 'recording' ? void finishSentence() : void startSentence()
            }
            style={{ width: '100%', minHeight: '20vh', fontSize: '1.5rem' }}
          >
            {sentencePhase === 'recording'
              ? `⏹ Bitir ${formatElapsed(elapsedMs)}`
              : sentencePhase === 'starting'
                ? 'Mikrofon açılıyor...'
                : '🎤 Okumaya Başla'}
          </button>

          {wordFull ? (
            <button
              type="button"
              onClick={() => {
                void refreshWord();
                void refreshProgress();
              }}
            >
              Sıradaki Cümle
            </button>
          ) : (
            <button type="button" disabled={busy} onClick={skipSentence}>
              Atla
            </button>
          )}
        </>
      )}

      {status && <p>{status}</p>}
    </main>
  );
}
