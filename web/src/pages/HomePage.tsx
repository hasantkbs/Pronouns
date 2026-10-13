import { useEffect, useState } from 'react';
import Icon from '../components/Icon';
import { BigButton, FilledButton } from '../components/ui';
import { USER_ID } from '../lib/api';
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
    <main className="home">
      {updateAvailable && (
        <div role="status" className="update-banner">
          <Icon name="system_update" />
          <strong>Uygulama güncellemesi var</strong>
          <FilledButton onClick={() => void openApkDownload()}>Güncelle</FilledButton>
        </div>
      )}
      <div className="home-logo">
        <Icon name="spatial_audio_off" size={46} />
      </div>
      <h1 className="home-title">Furkanca</h1>
      <p className="home-subtitle">Konuşma Asistanı</p>
      <nav className="home-buttons">
        <BigButton
          to="/furkanca"
          label="Furkanca"
          sublabel="Konuş → Düzeltilmiş metin & ses"
          icon="record_voice_over"
          gradient="purple"
        />
        <BigButton to="/kayit" label="Kayıt" sublabel="Kelime & ses kaydı yükle" icon="mic" gradient="green" />
        <BigButton to="/ayarlar" label="Ayarlar" sublabel="Model tercihi & ince ayar" icon="tune" gradient="orange" />
      </nav>
      <p className="home-footer">Kullanıcı: {USER_ID}</p>
    </main>
  );
}
