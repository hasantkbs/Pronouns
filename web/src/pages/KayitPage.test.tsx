import type { ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import KayitPage from './KayitPage';

// Sunucu tarafi render: useEffect calismaz, sayfanin ilk gorunumu test edilir.
const html = (node: ReactNode) => renderToStaticMarkup(<MemoryRouter>{node}</MemoryRouter>);

describe('KayitPage (Flutter görünümü)', () => {
  it('kayit has the pending folder action and the word/sentence segments', () => {
    const out = html(<KayitPage />);
    expect(out).toContain('href="/kayit/bekleyen"');
    expect(out).toContain('aria-label="Bekleyen Kayıtlar (0)"');
    expect(out).toMatch(/aria-pressed="true"[^>]*>Kelime/);
    expect(out).toContain('Kelime Seti');
    expect(out).toContain('role="progressbar"');
  });
});
