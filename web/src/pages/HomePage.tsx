import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { checkForUpdate, openApkDownload } from '../lib/appUpdate';

export default function HomePage() {
  const [updateAvailable, setUpdateAvailable] = useState(false);

  useEffect(() => {
    let active = true;
    void checkForUpdate().then((available) => {
      if (active) setUpdateAvailable(available);
    });
    return () => {
      active = false;
    };
  }, []);

  return (
    <main>
      {updateAvailable && (
        <div role="status" style={{ background: '#fff3cd', padding: '0.75rem', marginBottom: '1rem' }}>
          <strong>Uygulama güncellemesi var</strong>{' '}
          <button type="button" onClick={() => void openApkDownload()}>
            Güncelle
          </button>
        </div>
      )}
      <h1>Furkanca</h1>
      <p>Konuşma Asistanı</p>
      <nav>
        <ul>
          <li>
            <Link to="/furkanca">Furkanca</Link>
          </li>
          <li>
            <Link to="/kayit">Kayıt</Link>
          </li>
          <li>
            <Link to="/ayarlar">Ayarlar</Link>
          </li>
        </ul>
      </nav>
    </main>
  );
}
