import type { ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import AyarlarLoginPage from './AyarlarLoginPage';

// Sunucu tarafi render: useEffect calismaz, sayfanin ilk gorunumu test edilir.
const html = (node: ReactNode) => renderToStaticMarkup(<MemoryRouter>{node}</MemoryRouter>);

describe('AyarlarLoginPage (Flutter görünümü)', () => {
  it('settings login shows the lock, both fields and an orange submit button', () => {
    const out = html(<AyarlarLoginPage onSuccess={() => {}} />);
    expect(out).toContain('class="settings-lock"');
    expect(out).toContain('Kullanıcı Adı');
    expect(out).toContain('Şifre');
    expect(out).toMatch(/type="submit" class="button button--filled button--settings"/);
  });
});
