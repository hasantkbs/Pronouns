import { useState, type FormEvent } from 'react';
import AppBar from '../components/AppBar';
import Icon from '../components/Icon';
import { FilledButton } from '../components/ui';

const ADMIN_USERNAME = 'admin';
const ADMIN_PASSWORD = 'adminpassword';

export default function AyarlarLoginPage({ onSuccess }: { onSuccess: () => void }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (username === ADMIN_USERNAME && password === ADMIN_PASSWORD) {
      onSuccess();
    } else {
      setError('Kullanıcı adı veya şifre hatalı');
    }
  }

  return (
    <main className="page">
      <AppBar title="Ayarlar Girişi" backTo="/" />
      <form className="page-body" style={{ justifyContent: 'center' }} onSubmit={handleSubmit}>
        <span className="settings-lock">
          <Icon name="lock_outline" size={48} />
        </span>
        <label className="field">
          <span className="field-label">Kullanıcı Adı</span>
          <Icon name="person_outline" />
          <input value={username} autoComplete="username" onChange={(e) => setUsername(e.target.value)} />
        </label>
        <label className="field">
          <span className="field-label">Şifre</span>
          <Icon name="key" />
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        {error && (
          <p role="alert" className="error-text">
            {error}
          </p>
        )}
        <FilledButton type="submit" icon="login" className="button--settings">
          Giriş Yap
        </FilledButton>
      </form>
    </main>
  );
}
