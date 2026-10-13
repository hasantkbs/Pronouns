import { useEffect, useState } from 'react';
import { uploadRecording } from '../lib/api';
import { getAllTakes, removeTake, takeKind, type PendingTake } from '../lib/pendingRecordingsStore';
import { groupByWord } from '../lib/uiFormat';
import AppBar from '../components/AppBar';
import Icon from '../components/Icon';
import { FilledButton, IconButton } from '../components/ui';

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
        await uploadRecording(take.word, take.audioBlob, takeKind(take));
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

  const groups = groupByWord(takes);

  return (
    <main className="page">
      <AppBar title="Bekleyen Kayıtlar" backTo="/kayit" />
      <div className="page-body">
        <p className="muted">
          {takes.length} kayıt, {groups.length} kelime bekliyor.
        </p>
        <FilledButton
          icon={uploadingAll ? 'hourglass_top' : 'cloud_upload'}
          onClick={handleUploadAll}
          disabled={uploadingAll || takes.length === 0}
        >
          {uploadingAll ? 'Yükleniyor...' : 'Tümünü Yükle'}
        </FilledButton>

        {takes.length === 0 ? (
          <p className="empty">Bekleyen kayıt yok.</p>
        ) : (
          groups.map(([w, wTakes]) => (
            <section key={w}>
              <h2 className="take-group-title">{w}</h2>
              {wTakes.map((take, i) => (
                <div key={take.id} className="take-row">
                  <span className="take-label">Tekrar {i + 1}</span>
                  {status[take.id] === 'uploading' ? (
                    <span className="spinner" role="status" aria-label="Yükleniyor" />
                  ) : (
                    <>
                      {status[take.id] === 'error' && (
                        <span className="error-icon" role="img" aria-label="Hata" title={errors[take.id]}>
                          <Icon name="error_outline" size={20} />
                        </span>
                      )}
                      <IconButton
                        icon="play_arrow"
                        label="Oynat"
                        onClick={() => {
                          const audio = new Audio(URL.createObjectURL(take.audioBlob));
                          void audio.play();
                        }}
                      />
                      <IconButton icon="delete_outline" label="Sil" onClick={() => handleDelete(take.id)} />
                    </>
                  )}
                </div>
              ))}
            </section>
          ))
        )}
      </div>
    </main>
  );
}
