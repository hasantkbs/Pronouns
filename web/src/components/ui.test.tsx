import type { ReactElement, ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import AppBar from './AppBar';
import Icon from './Icon';
import { BigButton, FilledButton, OutlinedButton, ProgressBar, ResultTile, SegmentedButton } from './ui';

const html = (node: ReactNode) => renderToStaticMarkup(<MemoryRouter>{node}</MemoryRouter>);

describe('AppBar', () => {
  it('links back to a fixed page so the app never leaves its history', () => {
    const out = html(<AppBar title="Kayıt" backTo="/" />);
    expect(out).toContain('href="/"');
    expect(out).toContain('aria-label="Geri"');
    expect(out).toContain('Kayıt');
  });

  it('replaces the history entry so the phone back key on home closes the app', () => {
    // <Link> push'u Home -> Kayit -> ok -> Home sonrasi geri tusunu Kayit'a dondururdu
    const link = (AppBar({ title: 'Kayıt', backTo: '/' }) as ReactElement<{ children: ReactElement[] }>).props
      .children[0];
    expect(link.props).toMatchObject({ to: '/', replace: true });
  });

  it('renders an optional action on the right', () => {
    expect(html(<AppBar title="Kayıt" backTo="/" action={<span>eylem</span>} />)).toContain('eylem');
  });
});

describe('Icon', () => {
  it('renders a decorative svg path', () => {
    const out = html(<Icon name="mic" />);
    expect(out).toContain('<svg');
    expect(out).toContain('aria-hidden="true"');
    expect(out).toMatch(/<path d="M[^"]+"/);
  });
});

describe('buttons', () => {
  it('passes disabled through so busy states cannot be tapped', () => {
    expect(html(<FilledButton icon="mic" disabled>Kaydet</FilledButton>)).toContain('disabled=""');
    expect(html(<OutlinedButton icon="skip_next" disabled>Atla</OutlinedButton>)).toContain('disabled=""');
  });

  it('big button links to its page with label and sublabel', () => {
    const out = html(
      <BigButton to="/kayit" label="Kayıt" sublabel="Kelime & ses kaydı yükle" icon="mic" gradient="green" />,
    );
    expect(out).toContain('href="/kayit"');
    expect(out).toContain('Kayıt');
    expect(out).toContain('Kelime &amp; ses kaydı yükle');
  });
});

describe('ProgressBar', () => {
  it('clamps the fill between 0 and 100 percent', () => {
    expect(html(<ProgressBar value={150} />)).toContain('width:100%');
    expect(html(<ProgressBar value={-5} />)).toContain('width:0%');
  });
});

describe('ResultTile', () => {
  it('keeps errors announced to screen readers', () => {
    const out = html(<ResultTile label="Hata" value="Mikrofon açılamadı" tone="error" />);
    expect(out).toContain('role="alert"');
    expect(out).toContain('Mikrofon açılamadı');
  });

  it('uses a polite status role for normal messages', () => {
    expect(html(<ResultTile label="Durum" value="Cihazda kaydedildi" />)).toContain('role="status"');
  });
});

describe('SegmentedButton', () => {
  it('marks only the selected segment as pressed', () => {
    const out = html(
      <SegmentedButton
        label="Kayıt türü"
        options={[
          { value: 'word', label: 'Kelime' },
          { value: 'sentence', label: 'Cümle' },
        ]}
        value="sentence"
        onChange={() => {}}
      />,
    );
    expect(out.match(/aria-pressed="true"/g)).toHaveLength(1);
    expect(out).toMatch(/aria-pressed="true"[^>]*>Cümle/);
  });
});
