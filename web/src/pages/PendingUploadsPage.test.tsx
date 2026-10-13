import type { ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import PendingUploadsPage from './PendingUploadsPage';

// Sunucu tarafi render: useEffect calismaz, sayfanin ilk gorunumu test edilir.
const html = (node: ReactNode) => renderToStaticMarkup(<MemoryRouter>{node}</MemoryRouter>);

describe('PendingUploadsPage (Flutter görünümü)', () => {
  it('pending uploads goes back to kayit and shows the upload-all button', () => {
    const out = html(<PendingUploadsPage />);
    expect(out).toContain('href="/kayit"');
    expect(out).toContain('Tümünü Yükle');
    expect(out).toContain('Bekleyen kayıt yok.');
  });
});
