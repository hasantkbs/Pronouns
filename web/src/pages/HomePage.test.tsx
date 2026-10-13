import type { ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import HomePage from './HomePage';

// Sunucu tarafi render: useEffect calismaz, sayfanin ilk gorunumu test edilir.
const html = (node: ReactNode) => renderToStaticMarkup(<MemoryRouter>{node}</MemoryRouter>);

describe('HomePage (Flutter görünümü)', () => {
  it('home shows the three big buttons with their Flutter sublabels and the user', () => {
    const out = html(<HomePage />);
    expect(out).toContain('class="home"');
    expect(out.match(/class="big-button /g)).toHaveLength(3);
    for (const href of ['/furkanca', '/kayit', '/ayarlar']) expect(out).toContain(`href="${href}"`);
    expect(out).toContain('Konuş → Düzeltilmiş metin &amp; ses');
    expect(out).toContain('Kelime &amp; ses kaydı yükle');
    expect(out).toContain('Model tercihi &amp; ince ayar');
    expect(out).toContain('Kullanıcı: FurkanV1');
  });
});
