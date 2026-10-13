import type { ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import AyarlarPage from './AyarlarPage';

// Sunucu tarafi render: useEffect calismaz, sayfanin ilk gorunumu test edilir.
const html = (node: ReactNode) => renderToStaticMarkup(<MemoryRouter>{node}</MemoryRouter>);

describe('AyarlarPage (Flutter görünümü)', () => {
  it('settings page keeps the app bar while loading', () => {
    const out = html(<AyarlarPage />);
    expect(out).toContain('Model Ayarları');
    expect(out).toContain('aria-label="Yükleniyor"');
  });
});
