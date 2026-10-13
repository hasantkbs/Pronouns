import type { ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import FurkancaPage from './FurkancaPage';

// Sunucu tarafi render: useEffect calismaz, sayfanin ilk gorunumu test edilir.
const html = (node: ReactNode) => renderToStaticMarkup(<MemoryRouter>{node}</MemoryRouter>);

describe('FurkancaPage (Flutter görünümü)', () => {
  it('furkanca has an app bar back to home, the tall start button and the empty history text', () => {
    const out = html(<FurkancaPage />);
    expect(out).toContain('class="app-bar"');
    expect(out).toContain('href="/"');
    expect(out).toMatch(/class="button button--filled button--tall"[^>]*>.*Konuşmaya Başla/);
    expect(out).toContain('Henüz çeviri yok.');
  });
});
