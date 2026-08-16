import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import FurkancaPage from './pages/FurkancaPage';
import KayitPage from './pages/KayitPage';
import PendingUploadsPage from './pages/PendingUploadsPage';
import AyarlarLoginPage from './pages/AyarlarLoginPage';
import AyarlarPage from './pages/AyarlarPage';

export default function App() {
  const [ayarlarUnlocked, setAyarlarUnlocked] = useState(false);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/furkanca" element={<FurkancaPage />} />
        <Route path="/kayit" element={<KayitPage />} />
        <Route path="/kayit/bekleyen" element={<PendingUploadsPage />} />
        <Route
          path="/ayarlar"
          element={
            ayarlarUnlocked ? (
              <AyarlarPage />
            ) : (
              <AyarlarLoginPage onSuccess={() => setAyarlarUnlocked(true)} />
            )
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
