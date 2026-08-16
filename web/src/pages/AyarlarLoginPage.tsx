import { useState, type FormEvent } from 'react';

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
    <main>
      <h1>Ayarlar Girişi</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Kullanıcı Adı
          <input value={username} onChange={(e) => setUsername(e.target.value)} />
        </label>
        <label>
          Şifre
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        {error && <p role="alert">{error}</p>}
        <button type="submit">Giriş Yap</button>
      </form>
    </main>
  );
}
