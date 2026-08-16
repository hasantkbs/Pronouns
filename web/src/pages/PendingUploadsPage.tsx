import { useEffect, useState } from 'react';
import { uploadRecording } from '../lib/api';
import { getAllTakes, removeTake, type PendingTake } from '../lib/pendingRecordingsStore';

type UploadStatus = 'uploading' | 'error';

export default function PendingUploadsPage() {
  const [takes, setTakes] = useState<PendingTake[]>([]);
  const [status, setStatus] = useState<Record<number, UploadStatus>>({});
  const [errors, setErrors] = useState<Record<number, string>>({});
  const [uploadingAll, setUploadingAll] = useState(false);

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    setTakes(await getAllTakes());
  }

  async function handleUploadAll() {
    if (uploadingAll) return;
    setUploadingAll(true);
    const current = await getAllTakes();

    for (const take of current) {
      setStatus((prev) => ({ ...prev, [take.id]: 'uploading' }));
      try {
        await uploadRecording(take.word, take.audioBlob);
        await removeTake(take.id);
        setStatus((prev) => {
          const next = { ...prev };
          delete next[take.id];
          return next;
        });
        setTakes((prev) => prev.filter((t) => t.id !== take.id));
      } catch (e) {
        setStatus((prev) => ({ ...prev, [take.id]: 'error' }));
        setErrors((prev) => ({ ...prev, [take.id]: (e as Error).message }));
      }
    }

    setUploadingAll(false);
  }

  async function handleDelete(id: number) {
    await removeTake(id);
    setTakes((prev) => prev.filter((t) => t.id !== id));
  }

  const byWord = new Map<string, PendingTake[]>();
  for (const t of takes) {
    const list = byWord.get(t.word) ?? [];
    list.push(t);
    byWord.set(t.word, list);
  }

  return (
    <main>
      <h1>Bekleyen Kayıtlar</h1>
      <p>
        {takes.length} kayıt, {byWord.size} kelime bekliyor.
      </p>
      <button onClick={handleUploadAll} disabled={uploadingAll || takes.length === 0}>
        {uploadingAll ? 'Yükleniyor...' : 'Tümünü Yükle'}
      </button>

      {takes.length === 0 ? (
        <p>Bekleyen kayıt yok.</p>
      ) : (
        Array.from(byWord.entries()).map(([w, wTakes]) => (
          <section key={w}>
            <h2>{w}</h2>
            {wTakes.map((take, i) => (
              <div key={take.id}>
                <span>Tekrar {i + 1}</span>
                {status[take.id] === 'uploading' && <span>Yükleniyor...</span>}
                {status[take.id] === 'error' && <span title={errors[take.id]}>Hata</span>}
                {!status[take.id] && (
                  <>
                    <button
                      onClick={() => {
                        const audio = new Audio(URL.createObjectURL(take.audioBlob));
                        void audio.play();
                      }}
                    >
                      Oynat
                    </button>
                    <button onClick={() => handleDelete(take.id)}>Sil</button>
                  </>
                )}
              </div>
            ))}
          </section>
        ))
      )}
    </main>
  );
}
