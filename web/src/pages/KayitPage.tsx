import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getNextWord, getProgress, getWordSets } from '../lib/api';
import { startAudioCapture } from '../lib/audioCapture';
import { pcmToWav } from '../lib/vad';
import { addTake, getAllTakes, getPendingWords, getTakesForWord } from '../lib/pendingRecordingsStore';

export default function KayitPage() {
  const [sets, setSets] = useState<string[]>([]);
  const [setFile, setSetFile] = useState('wordSet.txt');
  const [word, setWord] = useState<string | null>(null);
  const [currentCount, setCurrentCount] = useState(0);
  const [idealReps, setIdealReps] = useState(0);
  const [totalWords, setTotalWords] = useState(0);
  const [completedWords, setCompletedWords] = useState(0);
  const [seconds, setSeconds] = useState(2);
  const [remaining, setRemaining] = useState(0);
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [wordPendingCount, setWordPendingCount] = useState(0);

  const loadSets = useCallback(async () => {
    try {
      const { sets: s } = await getWordSets();
      setSets(s);
      if (s.length > 0 && !s.includes(setFile)) setSetFile(s[0]);
    } catch {
      // set listesi opsiyonel; varsayılan setFile ile devam edilir
    }
  }, [setFile]);

  const refreshWord = useCallback(async () => {
    try {
      const pendingWords = Array.from(await getPendingWords());
      const data = await getNextWord(setFile, pendingWords);
      setWord(data.word ?? null);
      setCurrentCount(data.current_count ?? 0);
      setIdealReps(data.ideal_repetitions ?? 0);
      setTotalWords((prev) => data.total_words ?? prev);
    } catch (e) {
      setStatus(`Kelime alınamadı: ${(e as Error).message}`);
    }
  }, [setFile]);

  const refreshProgress = useCallback(async () => {
    try {
      const data = await getProgress(setFile);
      setCompletedWords(data.completed_words ?? 0);
      setTotalWords((prev) => data.total_words ?? prev);
      setIdealReps((prev) => data.ideal_repetitions ?? prev);
    } catch {
      // ilerleme opsiyonel
    }
  }, [setFile]);

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

  async function handleRecord() {
    if (recording || !word) return;
    setRecording(true);
    setStatus(null);
    setRemaining(seconds);

    const chunks: Int16Array[] = [];
    const handle = await startAudioCapture((chunk) => chunks.push(chunk));

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
    setRecording(false);
    setRemaining(0);
  }

  const allDone = word === null;
  const wordFull = word !== null && currentCount + wordPendingCount >= idealReps;

  return (
    <main>
      <h1>Kayıt</h1>
      <p>
        <Link to="/kayit/bekleyen">Bekleyen Kayıtlar ({pendingCount})</Link>
      </p>

      <label>
        Kelime Seti
        <select value={setFile} onChange={(e) => setSetFile(e.target.value)} disabled={recording}>
          {sets.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>

      <p>
        İlerleme: {completedWords} / {totalWords} kelime
      </p>

      {allDone ? (
        <p>Tüm kelimeler tamamlandı!</p>
      ) : (
        <>
          <h2>{word}</h2>
          <p>
            {currentCount + wordPendingCount} / {idealReps} tekrar
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
      )}

      {status && <p>{status}</p>}
    </main>
  );
}
