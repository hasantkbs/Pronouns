import { Link } from 'react-router-dom';

export default function HomePage() {
  return (
    <main>
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
