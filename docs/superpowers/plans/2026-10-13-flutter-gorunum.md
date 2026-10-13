# Flutter Görünümüne Dönüş Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** React arayüzü (Furkan'ın telefonunda canlı yüklenen) Flutter uygulamasının koyu mor temasına, ikonlu büyük düğmelerine ve kartlarına birebir getirilsin; davranış değişmesin.

**Architecture:** Tek tema dosyası (`src/theme.css`, CSS değişkenleri) + satır içi SVG Material ikonları (`components/Icon.tsx`) + Flutter widget'larının karşılığı küçük bileşenler (`components/AppBar.tsx`, `components/ui.tsx`). Altı sayfanın yalnızca JSX'i bu bileşenlerle yeniden yazılır; durum, istek ve kayıt mantığı satırı satırına aynı kalır. Görünüme ait küçük saf yardımcılar `lib/uiFormat.ts`'de.

**Tech Stack:** React 18 + TypeScript + Vite 5 + vitest 2 (`environment: 'node'`; render testleri `react-dom/server` `renderToStaticMarkup` + `MemoryRouter` ile, yeni kütüphane yok); react-router-dom 6.

**Spec:** `docs/superpowers/specs/2026-10-13-flutter-gorunum-design.md`

## Global Constraints

- İş `/home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite`, dal `worktree-react-rewrite` (başlangıç `c8ec06cf`). `npm`/`npx` komutları `web/` altından; `git apply` worktree kökünden.
- Yalnızca görünüm değişir: sayfaların `useState`/`useCallback`/`useEffect`/istek/kayıt kodu aynen kalır; diff'ler yalnızca import'ları ve `return (...)` JSX'ini değiştirir (tek istisna: `api.ts`'de `USER_ID` export edilir).
- Renkler: ana `#6C63FF`, Ayarlar `#F59E0B`; koyu zemin `#131318`. Yeni npm paketi yok; dış font/ikon yüklenmez; `color-mix()` kullanılmaz (eski Android WebView).
- **`npm run build` çalıştırılmaz ve `web/dist`'e yazılmaz** (canlı site oradan sunuluyor; dağıtım kullanıcı onayıyla ayrı). Görsel kontrol `--outDir` ile plan çalışma alanına derlenir. `deploy-web.sh` ve `deploy-android.sh` kullanılmaz.
- İstemci takımı: `cd web && npx tsc -b && npx vitest run` (başlangıçta 10 dosya / 64 test).
- Her commit'te dosyalar tek tek `git add` edilir. Commit mesajları şu satırla biter: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **Uzun cümle ve kelime metinleri** (Kayıt kartı, Bekleyen Kayıtlar başlıkları, Furkanca geçmişi) 412 px genişlikte taşmamalı: `overflow-wrap: anywhere`. Task 8'deki görsel kontrol `scrollWidth === 412` ile sabitler (uzun cümle içeren bekleyen kayıt dahil).
2. **Meşgul durumda dokunma:** kayıt/yükleme sırasında devre dışı düğmeler gerçekten `disabled` olmalı ve öyle görünmeli (opacity .38). Task 2 testi `disabled` aktarımını sabitler; sayfalar mevcut `disabled` koşullarını aynen geçirir.
3. **Geri oku APK'da uygulamadan çıkmamalı:** `history.back()` değil sabit hedef (`/`, Bekleyen → `/kayit`). Task 2 ve sayfa testleri `href`'i sabitler.
4. **Hata mesajları kaybolmamalı:** eski `role="alert"` metinleri `ResultTile tone="error"` (`role="alert"`) ya da `.error-text` `role="alert"` olarak kalır. Task 2 testi sabitler.
5. **Sunucudaki ayar değeri seçenek listesinde yoksa** (bugün `model: "whisper"`) segmentte hiçbir şey seçili görünmemeli değil, o değer seçili görünmeli; kaydedilen değer değişmemeli. Task 1 (`withCurrent`) ve Task 8 görsel kontrolü sabitler.

---

### Task 1: Görünüm yardımcıları (`lib/uiFormat.ts`)

**Files:**
- Create: `web/src/lib/uiFormat.ts`
- Test: `web/src/lib/uiFormat.test.ts`

**Interfaces:**
- Produces:
  - `progressPercent(completed: number, total: number): number` — 0–100 tam sayı; `total <= 0` → 0. (Task 5)
  - `groupByWord<T extends { word: string }>(takes: T[]): [string, T[]][]` — ilk görülme sırası. (Task 6)
  - `formatClock(date: Date): string` — `SS:DD:ss`. (Task 4)
  - `withCurrent(options: string[], current: string): string[]` — listede olmayan dolu değer sona eklenir. (Task 7)

- [ ] **Step 1: Write the failing test**

`web/src/lib/uiFormat.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { formatClock, groupByWord, progressPercent, withCurrent } from './uiFormat';

describe('progressPercent', () => {
  it('is 0 when there is nothing to do', () => {
    expect(progressPercent(0, 0)).toBe(0);
    expect(progressPercent(3, 0)).toBe(0);
  });

  it('rounds to a whole percent', () => {
    expect(progressPercent(1, 3)).toBe(33);
    expect(progressPercent(2, 3)).toBe(67);
  });

  it('stays between 0 and 100', () => {
    expect(progressPercent(12, 10)).toBe(100);
    expect(progressPercent(-1, 10)).toBe(0);
  });
});

describe('groupByWord', () => {
  it('groups takes by word in first-seen order, keeping take order', () => {
    const takes = [
      { id: 1, word: 'su' },
      { id: 2, word: 'ekmek' },
      { id: 3, word: 'su' },
    ];
    expect(groupByWord(takes)).toEqual([
      ['su', [takes[0], takes[2]]],
      ['ekmek', [takes[1]]],
    ]);
  });

  it('returns an empty list for no takes', () => {
    expect(groupByWord([])).toEqual([]);
  });
});

describe('formatClock', () => {
  it('formats HH:MM:SS with zero padding', () => {
    expect(formatClock(new Date(2026, 9, 13, 9, 5, 7))).toBe('09:05:07');
    expect(formatClock(new Date(2026, 9, 13, 23, 59, 0))).toBe('23:59:00');
  });
});

describe('withCurrent', () => {
  it('keeps the options when the current value is one of them', () => {
    expect(withCurrent(['Furkan', 'base'], 'base')).toEqual(['Furkan', 'base']);
  });

  it('adds a server value missing from the list so it shows as selected', () => {
    expect(withCurrent(['Furkan', 'base', 'large'], 'whisper')).toEqual(['Furkan', 'base', 'large', 'whisper']);
  });

  it('ignores an empty current value', () => {
    expect(withCurrent(['lora'], '')).toEqual(['lora']);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/lib/uiFormat.test.ts`
Expected: FAIL — `Failed to load url ./uiFormat` (dosya yok).

- [ ] **Step 3: Implement**

`web/src/lib/uiFormat.ts`:

```ts
/** Ilerleme cubugu icin tam sayi yuzde; toplam yoksa 0, her zaman 0-100. */
export function progressPercent(completed: number, total: number): number {
  if (total <= 0) return 0;
  return Math.min(100, Math.max(0, Math.round((completed / total) * 100)));
}

/** Kayitlari kelimeye gore gruplar; kelimeler ilk goruldukleri sirada. */
export function groupByWord<T extends { word: string }>(takes: T[]): [string, T[]][] {
  const groups = new Map<string, T[]>();
  for (const take of takes) {
    const list = groups.get(take.word) ?? [];
    list.push(take);
    groups.set(take.word, list);
  }
  return Array.from(groups.entries());
}

/** Gecmis kartlarindaki saat: SS:DD:ss */
export function formatClock(date: Date): string {
  const pad = (n: number) => n.toString().padStart(2, '0');
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

/** Segment secenekleri: sunucudaki deger listede yoksa sona eklenir ki secili gorunsun. */
export function withCurrent(options: string[], current: string): string[] {
  return current && !options.includes(current) ? [...options, current] : options;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/lib/uiFormat.test.ts`
Expected: `Tests  9 passed (9)`

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/uiFormat.ts web/src/lib/uiFormat.test.ts
git commit -m "feat: add display helpers for the Flutter-style UI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Tema, ikonlar ve ortak bileşenler

**Files:**
- Create: `web/src/theme.css`, `web/src/components/Icon.tsx`, `web/src/components/AppBar.tsx`, `web/src/components/ui.tsx`
- Modify: `web/src/main.tsx` (tema import'u), `web/index.html` (`theme-color`, koyu `body` zemini — açılışta beyaz yanıp sönme olmasın)
- Test: `web/src/components/ui.test.tsx`

**Interfaces:**
- Produces (Task 3–7 kullanır):
  - `Icon({ name: IconName, size?: number })` (default export), `IconName` = `arrow_back | arrow_forward_ios | check_circle | cloud_upload | delete_outline | error_outline | folder_open | graphic_eq | hourglass_top | key | list_alt | lock_outline | login | mic | mic_off | person_outline | play_arrow | record_voice_over | save | skip_next | spatial_audio_off | stop | system_update | timer | tune`
  - `AppBar({ title: string, backTo: string, action?: ReactNode })` (default export)
  - `ui.tsx` named exports:
    - `BigButton({ to, label, sublabel, icon: IconName, gradient: 'purple' | 'green' | 'orange' })`
    - `FilledButton` / `OutlinedButton` (`ButtonHTMLAttributes` + `icon?: IconName`; `type` varsayılan `"button"`; renk `className`: `button--danger`, `button--settings`, yükseklik `button--tall`)
    - `IconButton({ icon, label, ...button })`
    - `ResultTile({ label, value: ReactNode, tone?: 'accent' | 'settings' | 'error' })` — `error` → `role="alert"`, diğerleri `role="status"`
    - `SectionHeader({ children })`, `ProgressBar({ value: number /*0-100*/, label?: string })`
    - `SegmentedButton<T extends string>({ label, options: {value: T, label: string}[], value: T, onChange(v: T), disabled?, className? })`
  - CSS sınıfları (sayfalar kullanır): `page`, `page-body`, `home*`, `update-banner`, `muted`, `faint`, `small`, `row`, `row--between`, `stack`, `badge`, `icon-button`, `field`, `field-label`, `target-card`, `target-card--done`, `target-word`, `target-word--sentence`, `history-card`, `history-time`, `history-text`, `empty`, `switch-row`, `take-group-title`, `take-row`, `spinner`, `spinner--large`, `error-icon`, `error-text`, `settings`, `settings-lock`, `segmented--settings`, `value`.

- [ ] **Step 1: Write the failing test**

`web/src/components/ui.test.tsx`:

```tsx
import type { ReactNode } from 'react';
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/components`
Expected: FAIL — `Failed to load url ./AppBar`.

- [ ] **Step 3: Write the icons**

`web/src/components/Icon.tsx`:

```tsx
// Flutter uygulamasinda kullanilan Material ikonlarinin SVG yollari (24x24).
const PATHS = {
  arrow_back: 'M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z',
  arrow_forward_ios: 'M6.23 20.23 8 22l10-10L8 2 6.23 3.77 14.46 12z',
  check_circle:
    'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z',
  cloud_upload:
    'M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z',
  delete_outline:
    'M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM8 9h8v10H8V9zm7.5-5-1-1h-5l-1 1H5v2h14V4z',
  error_outline:
    'M11 15h2v2h-2zm0-8h2v6h-2zm.99-5C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8z',
  folder_open:
    'M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z',
  graphic_eq: 'M7 18h2V6H7v12zm4 4h2V2h-2v20zm-8-8h2v-4H3v4zm12 4h2V6h-2v12zm4-8v4h2v-4h-2z',
  hourglass_top: 'M6 2v6l4 4-4 4v6h12v-6l-4-4 4-4V2H6zm10 14.5V20H8v-3.5l4-4 4 4z',
  key: 'M12.65 10C11.83 7.67 9.61 6 7 6c-3.31 0-6 2.69-6 6s2.69 6 6 6c2.61 0 4.83-1.67 5.65-4H17v4h4v-4h2v-4H12.65zM7 14c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2z',
  list_alt:
    'M19 5v14H5V5h14m1.1-2H3.9c-.5 0-.9.4-.9.9v16.2c0 .4.4.9.9.9h16.2c.4 0 .9-.5.9-.9V3.9c0-.5-.5-.9-.9-.9zM11 7h6v2h-6V7zm0 4h6v2h-6v-2zm0 4h6v2h-6zM7 7h2v2H7zm0 4h2v2H7zm0 4h2v2H7z',
  lock_outline:
    'M12 17c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm6-9h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zM8.9 6c0-1.71 1.39-3.1 3.1-3.1s3.1 1.39 3.1 3.1v2H8.9V6zM18 20H6V10h12v10z',
  login:
    'M11 7 9.6 8.4l2.6 2.6H2v2h10.2l-2.6 2.6L11 17l5-5-5-5zm9 12h-8v2h8c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2h-8v2h8v14z',
  mic: 'M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z',
  mic_off:
    'M19 11h-1.7c0 .74-.16 1.43-.43 2.05l1.23 1.23c.56-.98.9-2.09.9-3.28zm-4.02.17c0-.06.02-.11.02-.17V5c0-1.66-1.34-3-3-3S9 3.34 9 5v.18l5.98 5.99zM4.27 3 3 4.27l6.01 6.01V11c0 1.66 1.33 3 2.99 3 .22 0 .44-.03.65-.08l1.66 1.66c-.71.33-1.5.52-2.31.52-2.76 0-5.3-2.1-5.3-5.1H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c.91-.13 1.77-.45 2.54-.9L19.73 21 21 19.73 4.27 3z',
  person_outline:
    'M12 5.9c1.16 0 2.1.94 2.1 2.1s-.94 2.1-2.1 2.1S9.9 9.16 9.9 8s.94-2.1 2.1-2.1m0 9c2.97 0 6.1 1.46 6.1 2.1v1.1H5.9V17c0-.64 3.13-2.1 6.1-2.1M12 4C9.79 4 8 5.79 8 8s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4zm0 9c-2.67 0-8 1.34-8 4v3h16v-3c0-2.66-5.33-4-8-4z',
  play_arrow: 'M8 5v14l11-7z',
  record_voice_over:
    'M9 13c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4zm7.76-9.64-1.68 1.69c.84 1.18.84 2.71 0 3.89l1.68 1.69c2.02-2.02 2.02-5.07 0-7.27zM20.07 2l-1.63 1.63c2.77 3.02 2.77 7.56 0 10.74L20.07 16c3.9-3.89 3.91-9.95 0-14z',
  save: 'M17 3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V7l-4-4zm-5 16c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3zm3-10H5V5h10v4z',
  skip_next: 'M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z',
  spatial_audio_off:
    'M10 13c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm6.39 2.56C14.71 14.7 12.53 14 10 14s-4.71.7-6.39 1.56C2.61 16.07 2 17.1 2 18.22V21h16v-2.78c0-1.12-.61-2.15-1.61-2.66zM20.36 1l-1.41 1.41c2.73 2.73 2.73 7.17 0 9.9l1.41 1.41c3.52-3.51 3.52-9.21 0-12.72zM17.54 10.9c1.95-1.95 1.95-5.12 0-7.07l-1.41 1.41c1.17 1.17 1.17 3.07 0 4.24l1.41 1.42z',
  stop: 'M6 6h12v12H6z',
  system_update:
    'M17 1.01 7 1c-1.1 0-1.99.9-1.99 2v18c0 1.1.89 2 1.99 2h10c1.1 0 2-.9 2-2V3c0-1.1-.9-1.99-2-1.99zM17 19H7V5h10v14zm-1-6h-3V8h-2v5H8l4 4 4-4z',
  timer:
    'M15 1H9v2h6V1zm-4 13h2V8h-2v6zm8.03-6.61 1.42-1.42c-.43-.51-.9-.99-1.41-1.41l-1.42 1.42C16.07 4.74 14.12 4 12 4c-4.97 0-9 4.03-9 9s4.02 9 9 9 9-4.03 9-9c0-2.12-.74-4.07-1.97-5.61zM12 20c-3.87 0-7-3.13-7-7s3.13-7 7-7 7 3.13 7 7-3.13 7-7 7z',
  tune: 'M3 17v2h6v-2H3zM3 5v2h10V5H3zm10 16v-2h8v-2h-8v-2h-2v6h2zM7 9v2H3v2h4v2h2V9H7zm14 4v-2H11v2h10zm-6-4h2V7h4V5h-4V3h-2v6z',
} as const;

export type IconName = keyof typeof PATHS;

export default function Icon({ name, size = 24 }: { name: IconName; size?: number }) {
  return (
    <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}
```

- [ ] **Step 4: Write the app bar and shared components**

`web/src/components/AppBar.tsx`:

```tsx
import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import Icon from './Icon';

/** Flutter AppBar'i: geri oku sabit bir sayfaya gider (gecmis bos olsa da calisir). */
export default function AppBar({ title, backTo, action }: { title: string; backTo: string; action?: ReactNode }) {
  return (
    <header className="app-bar">
      <Link to={backTo} className="icon-button" aria-label="Geri">
        <Icon name="arrow_back" />
      </Link>
      <h1 className="app-bar-title">{title}</h1>
      {action && <div className="app-bar-action">{action}</div>}
    </header>
  );
}
```

`web/src/components/ui.tsx`:

```tsx
import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { Link } from 'react-router-dom';
import Icon, { type IconName } from './Icon';

type Gradient = 'purple' | 'green' | 'orange';

/** Ana sayfadaki renk gecisli buyuk dugme (Flutter _BigButton). */
export function BigButton(props: { to: string; label: string; sublabel: string; icon: IconName; gradient: Gradient }) {
  return (
    <Link to={props.to} className={`big-button big-button--${props.gradient}`}>
      <span className="big-button-icon">
        <Icon name={props.icon} size={28} />
      </span>
      <span className="big-button-text">
        <span className="big-button-label">{props.label}</span>
        <span className="big-button-sublabel">{props.sublabel}</span>
      </span>
      <Icon name="arrow_forward_ios" size={16} />
    </Link>
  );
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & { icon?: IconName; children: ReactNode };

function ButtonContent({ icon, children }: { icon?: IconName; children: ReactNode }) {
  return (
    <>
      {icon && <Icon name={icon} />}
      <span>{children}</span>
    </>
  );
}

/** Dolu, ikonlu dugme (Flutter FilledButton.icon). Renk icin className: 'button--danger', 'button--settings'. */
export function FilledButton({ icon, children, className = '', type = 'button', ...rest }: ButtonProps) {
  return (
    <button type={type} className={`button button--filled ${className}`} {...rest}>
      <ButtonContent icon={icon}>{children}</ButtonContent>
    </button>
  );
}

/** Cerceveli, ikonlu dugme (Flutter OutlinedButton.icon). */
export function OutlinedButton({ icon, children, className = '', type = 'button', ...rest }: ButtonProps) {
  return (
    <button type={type} className={`button button--outlined ${className}`} {...rest}>
      <ButtonContent icon={icon}>{children}</ButtonContent>
    </button>
  );
}

/** Kucuk yuvarlak ikon dugmesi (oynat, sil). */
export function IconButton({ icon, label, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & { icon: IconName; label: string }) {
  return (
    <button type="button" className="icon-button" aria-label={label} title={label} {...rest}>
      <Icon name={icon} size={20} />
    </button>
  );
}

/** Renkli cerceveli bilgi kutusu (Flutter _ResultTile). Hata kutusu ekran okuyucuya hemen bildirilir. */
export function ResultTile({ label, value, tone = 'accent' }: { label: string; value: ReactNode; tone?: 'accent' | 'settings' | 'error' }) {
  return (
    <div className={`result-tile result-tile--${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <div className="result-tile-label">{label}</div>
      <div className="result-tile-value">{value}</div>
    </div>
  );
}

export function SectionHeader({ children }: { children: ReactNode }) {
  return <h2 className="section-header">{children}</h2>;
}

/** Yuzde (0-100) ile dolan cubuk (Flutter LinearProgressIndicator). */
export function ProgressBar({ value, label }: { value: number; label?: string }) {
  const width = Math.min(100, Math.max(0, value));
  return (
    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(width)} aria-label={label}>
      <div className="progress-fill" style={{ width: `${width}%` }} />
    </div>
  );
}

/** Secili parcasi dolu segment dugmesi (Flutter SegmentedButton). */
export function SegmentedButton<T extends string>(props: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <div className={`segmented ${props.className ?? ''}`} role="group" aria-label={props.label}>
      {props.options.map((o) => (
        <button
          key={o.value}
          type="button"
          className="segmented-option"
          aria-pressed={o.value === props.value}
          disabled={props.disabled}
          onClick={() => props.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd web && npx vitest run src/components`
Expected: `Tests  9 passed (9)`

- [ ] **Step 6: Write the theme and load it**

`web/src/theme.css`:

```css
/* Flutter uygulamasinin gorunumu: Material 3, koyu tema, tohum renk #6C63FF */
:root {
  color-scheme: dark;
  --accent: #6c63ff;
  --settings-accent: #f59e0b;
  --danger: #ff5252;
  --error: #f44336;
  --primary: #c5c0ff;
  --surface: #131318;
  --surface-high: #35343a;
  --on-surface: #e5e1e9;
  --on-surface-muted: rgba(229, 225, 233, 0.6);
  --on-surface-faint: rgba(229, 225, 233, 0.4);
  --outline: #918f9a;
  font-family: Roboto, system-ui, -apple-system, 'Segoe UI', sans-serif;
  font-size: 14px;
  line-height: 1.4;
  -webkit-tap-highlight-color: transparent;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--surface);
  color: var(--on-surface);
}

h1,
h2,
p {
  margin: 0;
}

/* Uzun cumleler telefon genisliginde tasmasin */
body,
button,
input,
select {
  overflow-wrap: anywhere;
}

button,
input,
select {
  font: inherit;
  color: inherit;
}

.page {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.page-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 24px;
}

.muted {
  color: var(--on-surface-muted);
  font-size: 13px;
}

.faint {
  color: var(--on-surface-faint);
  font-size: 13px;
}

.small {
  font-size: 12px;
}

.center {
  text-align: center;
}

.row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.row--between {
  justify-content: space-between;
}

.stack {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.icon {
  flex-shrink: 0;
}

/* ── AppBar ── */
.app-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 64px;
  padding: 8px 8px 8px 4px;
  background: var(--surface);
  position: sticky;
  top: 0;
  z-index: 1;
}

.app-bar-title {
  flex: 1;
  font-size: 22px;
  font-weight: 400;
}

.app-bar-action {
  display: flex;
  align-items: center;
}

/* ── Ikon dugmesi ── */
.icon-button {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  border: none;
  border-radius: 50%;
  background: transparent;
  color: var(--on-surface);
  text-decoration: none;
  cursor: pointer;
}

.icon-button:active {
  background: rgba(255, 255, 255, 0.08);
}

.badge {
  position: absolute;
  top: 6px;
  right: 4px;
  min-width: 16px;
  height: 16px;
  padding: 0 4px;
  border-radius: 8px;
  background: #ffb4ab;
  color: #690005;
  font-size: 11px;
  font-weight: 500;
  line-height: 16px;
  text-align: center;
}

/* ── Ana sayfa ── */
.home {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px 24px;
  gap: 18px;
}

.home-logo {
  margin-top: 16px;
  width: 90px;
  height: 90px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--accent);
  animation: pulse 2s ease-in-out infinite alternate;
}

@keyframes pulse {
  from {
    background: radial-gradient(circle, rgba(108, 99, 255, 0.4), rgba(108, 99, 255, 0.05));
  }
  to {
    background: radial-gradient(circle, rgba(108, 99, 255, 0.6), rgba(108, 99, 255, 0.05));
  }
}

.home-title {
  font-size: 32px;
  font-weight: 800;
  letter-spacing: 1.2px;
  color: var(--primary);
  margin-top: -4px;
}

.home-subtitle {
  margin-top: -14px;
  color: rgba(229, 225, 233, 0.55);
}

.home-buttons {
  flex: 1;
  width: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 18px;
  margin-top: 18px;
}

.home-footer {
  font-size: 10px;
  color: rgba(229, 225, 233, 0.35);
}

.update-banner {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 12px;
  border: 1px solid rgba(108, 99, 255, 0.5);
  background: rgba(108, 99, 255, 0.15);
}

.update-banner strong {
  flex: 1;
}

.update-banner .button {
  width: auto;
  padding: 10px 16px;
}

/* ── Buyuk dugme ── */
.big-button {
  display: flex;
  align-items: center;
  gap: 18px;
  width: 100%;
  padding: 20px 24px;
  border-radius: 20px;
  color: #fff;
  text-decoration: none;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
}

.big-button:active {
  filter: brightness(0.92);
}

.big-button--purple {
  background: linear-gradient(90deg, #6c63ff, #3b82f6);
}

.big-button--green {
  background: linear-gradient(90deg, #10b981, #059669);
}

.big-button--orange {
  background: linear-gradient(90deg, #f59e0b, #ef4444);
}

.big-button-icon {
  width: 52px;
  height: 52px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.big-button-text {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.big-button-label {
  font-size: 20px;
  font-weight: 800;
}

.big-button-sublabel {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.82);
}

.big-button > .icon:last-child {
  color: rgba(255, 255, 255, 0.54);
}

/* ── Dugmeler ── */
.button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  padding: 16px;
  border-radius: 14px;
  font-size: 16px;
  font-weight: 700;
  cursor: pointer;
}

.button--filled {
  border: none;
  background: var(--accent);
  color: #fff;
}

.button--outlined {
  border: 1px solid var(--outline);
  background: transparent;
  color: var(--primary);
  padding: 12px 16px;
  font-size: 14px;
  font-weight: 500;
}

.button--filled.button--danger {
  background: var(--danger);
}

.button--filled.button--settings {
  background: var(--settings-accent);
}

.button--outlined.button--settings {
  border-color: var(--settings-accent);
  color: var(--settings-accent);
  padding: 14px 16px;
  font-size: 15px;
  font-weight: 700;
}

.button--tall {
  min-height: 30vh;
  font-size: 22px;
  flex-direction: column;
}

.button--tall .icon {
  width: 48px;
  height: 48px;
}

.button:disabled,
.segmented-option:disabled {
  opacity: 0.38;
  cursor: default;
}

.button:not(:disabled):active {
  filter: brightness(0.9);
}

/* ── Bilgi kutusu ── */
.result-tile {
  --tone: var(--accent);
  padding: 14px;
  border-radius: 12px;
  border: 1px solid rgba(108, 99, 255, 0.3);
  background: rgba(108, 99, 255, 0.08);
}

.result-tile--settings {
  --tone: var(--settings-accent);
  border-color: rgba(245, 158, 11, 0.3);
  background: rgba(245, 158, 11, 0.1);
}

.result-tile--error {
  --tone: var(--error);
  border-color: rgba(244, 67, 54, 0.3);
  background: rgba(244, 67, 54, 0.08);
}

.result-tile-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--tone);
  margin-bottom: 6px;
}

.result-tile-value {
  font-size: 14px;
  white-space: pre-wrap;
}

.section-header {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: var(--primary);
}

/* ── Ilerleme cubugu ── */
.progress {
  height: 8px;
  border-radius: 6px;
  background: rgba(108, 99, 255, 0.15);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--accent);
  transition: width 0.2s;
}

/* ── Segment dugmesi ── */
.segmented {
  display: flex;
  border: 1px solid var(--outline);
  border-radius: 20px;
  overflow: hidden;
}

.segmented-option {
  flex: 1;
  min-height: 40px;
  border: none;
  background: transparent;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
}

.segmented-option + .segmented-option {
  border-left: 1px solid var(--outline);
}

.segmented-option[aria-pressed='true'] {
  background: #474364;
  color: #e4dfff;
}

.segmented--settings .segmented-option[aria-pressed='true'] {
  background: rgba(245, 158, 11, 0.3);
  color: #ffddb3;
}

/* ── Form alanlari (Flutter OutlineInputBorder) ── */
.field {
  position: relative;
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 56px;
  padding: 0 12px;
  border: 1px solid var(--outline);
  border-radius: 4px;
}

.field:focus-within {
  border: 2px solid var(--primary);
  padding: 0 11px;
}

.field-label {
  position: absolute;
  top: -9px;
  left: 10px;
  padding: 0 4px;
  background: var(--surface);
  font-size: 12px;
  color: var(--on-surface-muted);
}

.field input,
.field select {
  flex: 1;
  min-width: 0;
  height: 54px;
  border: none;
  outline: none;
  background: transparent;
  font-size: 16px;
}

.field select option {
  background: var(--surface-high);
}

/* ── Kartlar ── */
.target-card {
  padding: 20px;
  border-radius: 16px;
  border: 1px solid rgba(108, 99, 255, 0.4);
  background: var(--surface-high);
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.target-card--done {
  background: rgba(108, 99, 255, 0.12);
  color: var(--accent);
}

.target-word {
  font-size: 30px;
  font-weight: 900;
}

.target-word--sentence {
  font-size: 24px;
}

.history-card {
  padding: 14px;
  border-radius: 12px;
  border: 1px solid rgba(108, 99, 255, 0.3);
  background: rgba(108, 99, 255, 0.08);
}

.history-time {
  font-size: 11px;
  font-weight: 700;
  color: var(--accent);
  margin-bottom: 6px;
}

.history-text {
  font-size: 14px;
}

.empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px 0;
  color: var(--on-surface-faint);
  font-size: 13px;
}

/* ── Kaydirici ── */
input[type='range'] {
  width: 100%;
  accent-color: var(--accent);
}

.settings input[type='range'] {
  accent-color: var(--settings-accent);
}

/* ── Anahtar (Flutter Switch) ── */
.switch-row {
  display: flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
}

.switch-row input {
  appearance: none;
  width: 52px;
  height: 32px;
  border-radius: 16px;
  border: 2px solid var(--outline);
  background: var(--surface-high);
  position: relative;
  flex-shrink: 0;
  cursor: pointer;
}

.switch-row input::after {
  content: '';
  position: absolute;
  top: 6px;
  left: 6px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--outline);
  transition: all 0.15s;
}

.switch-row input:checked {
  background: var(--settings-accent);
  border-color: var(--settings-accent);
}

.switch-row input:checked::after {
  left: 22px;
  top: 2px;
  width: 24px;
  height: 24px;
  background: #fff;
}

/* ── Bekleyen kayitlar ── */
.take-group-title {
  font-size: 18px;
  font-weight: 800;
  margin-bottom: 6px;
}

.take-row {
  display: flex;
  align-items: center;
  min-height: 40px;
}

.take-row > span {
  flex: 1;
}

.spinner {
  width: 16px;
  height: 16px;
  margin: 0 14px;
  border: 2px solid rgba(197, 192, 255, 0.3);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.spinner--large {
  width: 36px;
  height: 36px;
  border-width: 4px;
  margin: 32px auto;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.error-icon {
  display: inline-flex;
  color: var(--error);
  margin: 0 12px;
}

.error-text {
  color: var(--error);
  font-size: 13px;
}

/* ── Ayarlar ── */
.settings-lock {
  align-self: center;
  color: var(--settings-accent);
}

.settings .section-header {
  margin-top: 4px;
}

.value {
  font-weight: 700;
}
```

Worktree kökünde `git apply` (`main.tsx` tema import'u, `index.html` koyu zemin):

```diff
diff --git a/web/index.html b/web/index.html
index f796617d..0caffd6f 100644
--- a/web/index.html
+++ b/web/index.html
@@ -3,9 +3,10 @@
   <head>
     <meta charset="UTF-8" />
     <meta name="viewport" content="width=device-width, initial-scale=1.0" />
+    <meta name="theme-color" content="#131318" />
     <title>Furkanca</title>
   </head>
-  <body>
+  <body style="background: #131318">
     <div id="root"></div>
     <script type="module" src="/src/main.tsx"></script>
   </body>
diff --git a/web/src/main.tsx b/web/src/main.tsx
index d9736adc..f6e13ab6 100644
--- a/web/src/main.tsx
+++ b/web/src/main.tsx
@@ -1,6 +1,7 @@
 import { StrictMode } from 'react';
 import { createRoot } from 'react-dom/client';
 import App from './App';
+import './theme.css';
 
 createRoot(document.getElementById('root')!).render(
   <StrictMode>
```

- [ ] **Step 7: Type-check and run the suite**

Run: `cd web && npx tsc -b && npx vitest run`
Expected: tsc çıktısız; `Test Files  12 passed (12)`, `Tests  82 passed (82)`.

- [ ] **Step 8: Commit**

```bash
git add web/src/theme.css web/src/components/Icon.tsx web/src/components/AppBar.tsx web/src/components/ui.tsx web/src/components/ui.test.tsx web/src/main.tsx web/index.html
git commit -m "feat: add the Flutter dark theme, Material icons and shared UI components

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Ana sayfa

**Files:**
- Modify: `web/src/pages/HomePage.tsx` (yalnızca import'lar ve JSX), `web/src/lib/api.ts` (`USER_ID` export)
- Test: `web/src/pages/HomePage.test.tsx`

**Interfaces:**
- Consumes: `Icon`, `BigButton`, `FilledButton` (Task 2); `checkForUpdate`, `openApkDownload` (mevcut, değişmez).
- Produces: `export const USER_ID = 'FurkanV1'` (`lib/api.ts`).

- [ ] **Step 1: Write the failing test**

`web/src/pages/HomePage.test.tsx`:

```tsx
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/pages/HomePage.test.tsx`
Expected: FAIL — `expected '<main><h1>Furkanca</h1>…' to contain 'class="home"'`.

- [ ] **Step 3: Implement**

Worktree kökünde `git apply`:

```diff
diff --git a/web/src/lib/api.ts b/web/src/lib/api.ts
index ff42295f..8677435a 100644
--- a/web/src/lib/api.ts
+++ b/web/src/lib/api.ts
@@ -2,7 +2,7 @@ import { fetchWithTimeout, TRANSLATE_TIMEOUT_MS } from './deadlines';
 
 const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
 const API_KEY = import.meta.env.VITE_API_KEY;
-const USER_ID = 'FurkanV1';
+export const USER_ID = 'FurkanV1';
 
 function authHeaders(extra?: Record<string, string>): Record<string, string> {
   return { 'X-API-Key': API_KEY, ...extra };
```

```diff
diff --git a/web/src/pages/HomePage.tsx b/web/src/pages/HomePage.tsx
index 18625ba4..f49e93ef 100644
--- a/web/src/pages/HomePage.tsx
+++ b/web/src/pages/HomePage.tsx
@@ -1,5 +1,7 @@
 import { useEffect, useState } from 'react';
-import { Link } from 'react-router-dom';
+import Icon from '../components/Icon';
+import { BigButton, FilledButton } from '../components/ui';
+import { USER_ID } from '../lib/api';
 import { checkForUpdate, openApkDownload } from '../lib/appUpdate';
 
 export default function HomePage() {
@@ -16,30 +18,31 @@ export default function HomePage() {
   }, []);
 
   return (
-    <main>
+    <main className="home">
       {updateAvailable && (
-        <div role="status" style={{ background: '#fff3cd', padding: '0.75rem', marginBottom: '1rem' }}>
-          <strong>Uygulama güncellemesi var</strong>{' '}
-          <button type="button" onClick={() => void openApkDownload()}>
-            Güncelle
-          </button>
+        <div role="status" className="update-banner">
+          <Icon name="system_update" />
+          <strong>Uygulama güncellemesi var</strong>
+          <FilledButton onClick={() => void openApkDownload()}>Güncelle</FilledButton>
         </div>
       )}
-      <h1>Furkanca</h1>
-      <p>Konuşma Asistanı</p>
-      <nav>
-        <ul>
-          <li>
-            <Link to="/furkanca">Furkanca</Link>
-          </li>
-          <li>
-            <Link to="/kayit">Kayıt</Link>
-          </li>
-          <li>
-            <Link to="/ayarlar">Ayarlar</Link>
-          </li>
-        </ul>
+      <div className="home-logo">
+        <Icon name="spatial_audio_off" size={46} />
+      </div>
+      <h1 className="home-title">Furkanca</h1>
+      <p className="home-subtitle">Konuşma Asistanı</p>
+      <nav className="home-buttons">
+        <BigButton
+          to="/furkanca"
+          label="Furkanca"
+          sublabel="Konuş → Düzeltilmiş metin & ses"
+          icon="record_voice_over"
+          gradient="purple"
+        />
+        <BigButton to="/kayit" label="Kayıt" sublabel="Kelime & ses kaydı yükle" icon="mic" gradient="green" />
+        <BigButton to="/ayarlar" label="Ayarlar" sublabel="Model tercihi & ince ayar" icon="tune" gradient="orange" />
       </nav>
+      <p className="home-footer">Kullanıcı: {USER_ID}</p>
     </main>
   );
 }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx tsc -b && npx vitest run src/pages/HomePage.test.tsx`
Expected: tsc çıktısız; `Tests  1 passed (1)`.

- [ ] **Step 5: Commit**

```bash
git add web/src/pages/HomePage.tsx web/src/pages/HomePage.test.tsx web/src/lib/api.ts
git commit -m "feat: restore the Flutter home screen with big gradient buttons

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Furkanca sayfası

**Files:**
- Modify: `web/src/pages/FurkancaPage.tsx` (yalnızca import'lar ve JSX; `send`, `finishRecording`, `startRecording` aynen)
- Test: `web/src/pages/FurkancaPage.test.tsx`

**Interfaces:**
- Consumes: `AppBar`, `Icon`, `FilledButton`, `ProgressBar`, `ResultTile` (Task 2); `formatClock` (Task 1); `formatElapsed`, `buttonEnabled` (mevcut).

- [ ] **Step 1: Write the failing test**

`web/src/pages/FurkancaPage.test.tsx`:

```tsx
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/pages/FurkancaPage.test.tsx`
Expected: FAIL — `to contain 'class="app-bar"'`.

- [ ] **Step 3: Implement**

Worktree kökünde `git apply`:

```diff
diff --git a/web/src/pages/FurkancaPage.tsx b/web/src/pages/FurkancaPage.tsx
index 2c95806d..ccfdf472 100644
--- a/web/src/pages/FurkancaPage.tsx
+++ b/web/src/pages/FurkancaPage.tsx
@@ -5,6 +5,10 @@ import { playAudio, PLAYBACK_TIMEOUT_MS, startUnlessCancelled } from '../lib/dea
 import { buttonEnabled, flowReducer, formatElapsed, initialFlowState, type Phase } from '../lib/furkancaFlow';
 import { ManualRecorder } from '../lib/manualRecorder';
 import { pcmToWav } from '../lib/wav';
+import AppBar from '../components/AppBar';
+import Icon from '../components/Icon';
+import { FilledButton, ProgressBar, ResultTile } from '../components/ui';
+import { formatClock } from '../lib/uiFormat';
 
 const STATUS_TEXT: Record<Phase, string> = {
   idle: 'Hazır',
@@ -101,33 +105,43 @@ export default function FurkancaPage() {
   }, [finishRecording]);
 
   const recording = state.phase === 'recording';
+  const active = state.phase !== 'idle';
 
   return (
-    <main>
-      <h1>Furkanca</h1>
-      <button
-        type="button"
-        disabled={!buttonEnabled(state.phase)}
-        onClick={() => (recording ? finishRecording() : void startRecording())}
-        style={{ width: '100%', minHeight: '30vh', fontSize: '2rem' }}
-      >
-        {recording ? '⏹ Bitir' : '🎤 Konuşmaya Başla'}
-      </button>
-      <p>Durum: {STATUS_TEXT[state.phase]}</p>
-      {recording && (
-        <p>
-          {formatElapsed(elapsedMs)}{' '}
-          <meter min={0} max={0.3} value={Math.min(level, 0.3)} aria-label="Ses seviyesi" />
+    <main className="page">
+      <AppBar title="Furkanca" backTo="/" />
+      <div className="page-body">
+        <p className="muted">
+          Düğmeye dokun, konuş, bitince tekrar dokun — AI konuşmanı düzeltilmiş hâle çevirsin ve yüksek sesle
+          okusun.
         </p>
-      )}
-      {state.error && <p role="alert">{state.error}</p>}
-      <ul>
-        {state.history.map((entry, i) => (
-          <li key={i}>
-            {entry.time.toLocaleTimeString()} — {entry.text}
-          </li>
-        ))}
-      </ul>
+        <div className="row" style={{ color: 'var(--accent)' }}>
+          <Icon name={active ? 'graphic_eq' : 'mic_off'} size={18} />
+          <span style={{ color: 'var(--on-surface)', fontSize: 13 }}>{STATUS_TEXT[state.phase]}</span>
+        </div>
+        <FilledButton
+          icon={recording ? 'stop' : 'mic'}
+          className={`button--tall${recording ? ' button--danger' : ''}`}
+          disabled={!buttonEnabled(state.phase)}
+          onClick={() => (recording ? finishRecording() : void startRecording())}
+        >
+          {recording ? `Bitir ${formatElapsed(elapsedMs)}` : 'Konuşmaya Başla'}
+        </FilledButton>
+        {recording && <ProgressBar value={(Math.min(level, 0.3) / 0.3) * 100} label="Ses seviyesi" />}
+        {state.error && <ResultTile label="Hata" value={state.error} tone="error" />}
+        {state.history.length === 0 ? (
+          <p className="empty">Henüz çeviri yok.</p>
+        ) : (
+          <ul className="stack" style={{ listStyle: 'none', margin: 0, padding: 0 }}>
+            {state.history.map((entry, i) => (
+              <li key={i} className="history-card">
+                <div className="history-time">{formatClock(entry.time)}</div>
+                <div className="history-text">{entry.text}</div>
+              </li>
+            ))}
+          </ul>
+        )}
+      </div>
     </main>
   );
 }
```

Not: ses seviyesi `meter` yerine `ProgressBar`: `level` 0–0,3 aralığı (eskiden `meter max=0.3`) yüzdeye çevrilir. Hata `ResultTile tone="error"` (`role="alert"` korunur).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx tsc -b && npx vitest run src/pages/FurkancaPage.test.tsx`
Expected: tsc çıktısız; `Tests  1 passed (1)`.

- [ ] **Step 5: Commit**

```bash
git add web/src/pages/FurkancaPage.tsx web/src/pages/FurkancaPage.test.tsx
git commit -m "feat: give the Furkanca page the Flutter layout

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Kayıt sayfası

**Files:**
- Modify: `web/src/pages/KayitPage.tsx` (yalnızca import'lar ve JSX; tüm durum, `RequestGate`, `handleRecord`, `finishSentence`, `startSentence`, `skipSentence`, `changeKind` aynen)
- Test: `web/src/pages/KayitPage.test.tsx`

**Interfaces:**
- Consumes: `AppBar`, `Icon`, `FilledButton`, `OutlinedButton`, `ProgressBar`, `ResultTile`, `SegmentedButton` (Task 2); `progressPercent` (Task 1).

- [ ] **Step 1: Write the failing test**

`web/src/pages/KayitPage.test.tsx`:

```tsx
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/pages/KayitPage.test.tsx`
Expected: FAIL — `to contain 'aria-label="Bekleyen Kayıtlar (0)"'`.

- [ ] **Step 3: Implement**

Worktree kökünde `git apply`:

```diff
diff --git a/web/src/pages/KayitPage.tsx b/web/src/pages/KayitPage.tsx
index ffccf78a..2e246b63 100644
--- a/web/src/pages/KayitPage.tsx
+++ b/web/src/pages/KayitPage.tsx
@@ -7,6 +7,10 @@ import { formatElapsed } from '../lib/furkancaFlow';
 import { ManualRecorder } from '../lib/manualRecorder';
 import { RequestGate } from '../lib/requestGate';
 import { pcmToWav } from '../lib/wav';
+import AppBar from '../components/AppBar';
+import Icon from '../components/Icon';
+import { FilledButton, OutlinedButton, ProgressBar, ResultTile, SegmentedButton } from '../components/ui';
+import { progressPercent } from '../lib/uiFormat';
 import { addTake, getAllTakes, getPendingWords, getTakesForWord } from '../lib/pendingRecordingsStore';
 
 const DEFAULT_WORD_SET = 'wordSet.txt';
@@ -260,133 +264,154 @@ export default function KayitPage() {
   const wordFull = word !== null && currentCount + wordPendingCount >= idealReps;
   const repsDone = currentCount + wordPendingCount;
 
-  return (
-    <main>
-      <h1>Kayıt</h1>
-      <p>
-        <Link to="/kayit/bekleyen">Bekleyen Kayıtlar ({pendingCount})</Link>
-      </p>
-
-      <div role="group" aria-label="Kayıt türü">
-        <button
-          type="button"
-          aria-pressed={kind === 'word'}
-          disabled={busy}
-          onClick={() => changeKind('word')}
-        >
-          Kelime
-        </button>
-        <button
-          type="button"
-          aria-pressed={kind === 'sentence'}
-          disabled={busy}
-          onClick={() => changeKind('sentence')}
-        >
-          Cümle
-        </button>
-      </div>
+  const nextLabel = kind === 'word' ? 'Sıradaki Kelime' : 'Sıradaki Cümle';
+  const goNext = () => {
+    void refreshWord();
+    void refreshProgress();
+  };
 
-      <label>
-        {kind === 'word' ? 'Kelime Seti' : 'Cümle Seti'}
-        <select
-          value={setFile}
-          onChange={(e) => {
-            setSkipped([]);
-            setSetFile(e.target.value);
-          }}
+  return (
+    <main className="page">
+      <AppBar
+        title="Kayıt"
+        backTo="/"
+        action={
+          <Link to="/kayit/bekleyen" className="icon-button" aria-label={`Bekleyen Kayıtlar (${pendingCount})`}>
+            <Icon name="folder_open" />
+            {pendingCount > 0 && <span className="badge">{pendingCount}</span>}
+          </Link>
+        }
+      />
+      <div className="page-body">
+        <div>
+          <div className="row row--between">
+            <span>
+              İlerleme: {completedWords} / {totalWords} {unit}
+            </span>
+            <span className="value">{progressPercent(completedWords, totalWords)}%</span>
+          </div>
+          {pendingCount > 0 && <p className="faint small">+{pendingCount} kayıt cihazda bekliyor</p>}
+          <div style={{ marginTop: 6 }}>
+            <ProgressBar value={progressPercent(completedWords, totalWords)} label="İlerleme" />
+          </div>
+        </div>
+
+        <SegmentedButton
+          label="Kayıt türü"
+          options={[
+            { value: 'word', label: 'Kelime' },
+            { value: 'sentence', label: 'Cümle' },
+          ]}
+          value={kind}
           disabled={busy}
-        >
-          {sets.map((s) => (
-            <option key={s} value={s}>
-              {s}
-            </option>
-          ))}
-        </select>
-      </label>
-
-      <p>
-        İlerleme: {completedWords} / {totalWords} {unit}
-      </p>
-
-      {word === null ? (
-        done && <p>Tüm {unit}ler tamamlandı!</p>
-      ) : kind === 'word' ? (
-        <>
-          <h2>{word}</h2>
-          <p>
-            {repsDone} / {idealReps} tekrar
-          </p>
-
-          <label>
-            Kayıt süresi: {seconds}sn
-            <input
-              type="range"
-              min={1}
-              max={5}
-              value={seconds}
-              disabled={recording}
-              onChange={(e) => setSeconds(Number(e.target.value))}
-            />
-          </label>
-
-          <button onClick={handleRecord} disabled={recording || wordFull}>
-            {recording ? `Kaydediliyor... ${remaining}` : 'Kaydet'}
-          </button>
-
-          {wordFull && (
-            <button
-              onClick={() => {
-                void refreshWord();
-                void refreshProgress();
-              }}
-            >
-              Sıradaki Kelime
-            </button>
-          )}
-        </>
-      ) : (
-        <>
-          <h2 style={{ fontSize: '2rem' }}>{word}</h2>
-          <p>
-            Bu cümle: {repsDone} / {idealReps}
-          </p>
-
-          <button
-            type="button"
-            disabled={
-              sentencePhase === 'starting' || (sentencePhase === 'idle' && (wordFull || wordLoading))
-            }
-            onClick={() =>
-              sentencePhase === 'recording' ? void finishSentence() : void startSentence()
-            }
-            style={{ width: '100%', minHeight: '20vh', fontSize: '1.5rem' }}
+          onChange={changeKind}
+        />
+
+        <label className="field">
+          <span className="field-label">{kind === 'word' ? 'Kelime Seti' : 'Cümle Seti'}</span>
+          <Icon name="list_alt" />
+          <select
+            value={setFile}
+            onChange={(e) => {
+              setSkipped([]);
+              setSetFile(e.target.value);
+            }}
+            disabled={busy}
           >
-            {sentencePhase === 'recording'
-              ? `⏹ Bitir ${formatElapsed(elapsedMs)}`
-              : sentencePhase === 'starting'
-                ? 'Mikrofon açılıyor...'
-                : '🎤 Okumaya Başla'}
-          </button>
-
-          {wordFull ? (
-            <button
-              type="button"
-              onClick={() => {
-                void refreshWord();
-                void refreshProgress();
-              }}
+            {sets.map((s) => (
+              <option key={s} value={s}>
+                {s}
+              </option>
+            ))}
+          </select>
+        </label>
+
+        {word === null ? (
+          done && (
+            <div className="target-card target-card--done">
+              <Icon name="check_circle" size={28} />
+              <strong style={{ fontSize: 16 }}>Tüm {unit}ler tamamlandı!</strong>
+            </div>
+          )
+        ) : (
+          <div className="target-card">
+            <h2 className={kind === 'word' ? 'target-word' : 'target-word target-word--sentence'}>{word}</h2>
+            <p className="muted">
+              {wordFull ? `Bu ${unit} tamamlandı` : `Sıradaki Tekrar: ${repsDone + 1}`} · Mevcut: {repsDone} / {idealReps}
+            </p>
+          </div>
+        )}
+
+        {word !== null && kind === 'word' && (
+          <>
+            <div>
+              <div className="row">
+                <Icon name="timer" size={18} />
+                <span>Kayıt süresi: {seconds} sn</span>
+              </div>
+              <input
+                type="range"
+                min={1}
+                max={5}
+                value={seconds}
+                disabled={recording}
+                aria-label="Kayıt süresi"
+                onChange={(e) => setSeconds(Number(e.target.value))}
+              />
+            </div>
+
+            {recording && (
+              <div className="stack">
+                <ProgressBar value={((seconds - remaining) / seconds) * 100} label="Kayıt süresi" />
+                <p className="small">Kayıt alınıyor... {remaining} sn kaldı</p>
+              </div>
+            )}
+
+            <FilledButton
+              icon={recording ? 'hourglass_top' : 'mic'}
+              onClick={handleRecord}
+              disabled={recording || wordFull}
             >
-              Sıradaki Cümle
-            </button>
-          ) : (
-            <button type="button" disabled={busy || wordLoading} onClick={skipSentence}>
-              Atla
-            </button>
-          )}
-        </>
-      )}
-
-      {status && <p>{status}</p>}
+              {recording ? 'Kaydediliyor...' : 'Kaydet (Cihazda Sakla)'}
+            </FilledButton>
+
+            {wordFull && (
+              <OutlinedButton icon="skip_next" onClick={goNext}>
+                {nextLabel}
+              </OutlinedButton>
+            )}
+          </>
+        )}
+
+        {word !== null && kind === 'sentence' && (
+          <>
+            <FilledButton
+              icon={sentencePhase === 'recording' ? 'stop' : sentencePhase === 'starting' ? 'hourglass_top' : 'mic'}
+              className={`button--tall${sentencePhase === 'recording' ? ' button--danger' : ''}`}
+              disabled={sentencePhase === 'starting' || (sentencePhase === 'idle' && (wordFull || wordLoading))}
+              onClick={() => (sentencePhase === 'recording' ? void finishSentence() : void startSentence())}
+            >
+              {sentencePhase === 'recording'
+                ? `Bitir ${formatElapsed(elapsedMs)}`
+                : sentencePhase === 'starting'
+                  ? 'Mikrofon açılıyor...'
+                  : 'Okumaya Başla'}
+            </FilledButton>
+
+            {wordFull ? (
+              <OutlinedButton icon="skip_next" onClick={goNext}>
+                {nextLabel}
+              </OutlinedButton>
+            ) : (
+              <OutlinedButton icon="skip_next" disabled={busy || wordLoading} onClick={skipSentence}>
+                Atla
+              </OutlinedButton>
+            )}
+          </>
+        )}
+
+        {status && <ResultTile label="Durum" value={status} />}
+      </div>
     </main>
   );
 }
```

Davranış eşlemesi (inceleme için): "Kelime | Cümle" `changeKind`'i `busy` iken devre dışı çağırır (eskisiyle aynı); set seçimi aynı `onChange`; kelime modunda "Kaydet (Cihazda Sakla)" `disabled={recording || wordFull}`; "Sıradaki Kelime/Cümle" yalnızca `wordFull` iken; cümle düğmesi ve "Atla" koşulları aynen. Hedef kartı alt satırı: `Sıradaki Tekrar: repsDone+1 · Mevcut: repsDone / idealReps`, tamamlanınca `Bu kelime|cümle tamamlandı · Mevcut: …`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx tsc -b && npx vitest run src/pages/KayitPage.test.tsx`
Expected: tsc çıktısız; `Tests  1 passed (1)`.

- [ ] **Step 5: Commit**

```bash
git add web/src/pages/KayitPage.tsx web/src/pages/KayitPage.test.tsx
git commit -m "feat: give the Kayit page the Flutter layout

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Bekleyen Kayıtlar ve Ayarlar Girişi

**Files:**
- Modify: `web/src/pages/PendingUploadsPage.tsx` (satır içi gruplama → `groupByWord`; `handleUploadAll`, `handleDelete` aynen), `web/src/pages/AyarlarLoginPage.tsx` (yalnızca JSX; `<form onSubmit>` korunur)
- Test: `web/src/pages/PendingUploadsPage.test.tsx`, `web/src/pages/AyarlarLoginPage.test.tsx`

**Interfaces:**
- Consumes: `AppBar`, `Icon`, `FilledButton`, `IconButton` (Task 2); `groupByWord` (Task 1).

- [ ] **Step 1: Write the failing tests**

`web/src/pages/PendingUploadsPage.test.tsx`:

```tsx
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
```

`web/src/pages/AyarlarLoginPage.test.tsx`:

```tsx
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd web && npx vitest run src/pages/PendingUploadsPage.test.tsx src/pages/AyarlarLoginPage.test.tsx`
Expected: FAIL ×2 — `to contain 'href="/kayit"'` ve `to contain 'class="settings-lock"'`.

- [ ] **Step 3: Implement**

Worktree kökünde `git apply`:

```diff
diff --git a/web/src/pages/PendingUploadsPage.tsx b/web/src/pages/PendingUploadsPage.tsx
index fa705066..86dad82c 100644
--- a/web/src/pages/PendingUploadsPage.tsx
+++ b/web/src/pages/PendingUploadsPage.tsx
@@ -1,6 +1,10 @@
 import { useEffect, useState } from 'react';
 import { uploadRecording } from '../lib/api';
 import { getAllTakes, removeTake, takeKind, type PendingTake } from '../lib/pendingRecordingsStore';
+import { groupByWord } from '../lib/uiFormat';
+import AppBar from '../components/AppBar';
+import Icon from '../components/Icon';
+import { FilledButton, IconButton } from '../components/ui';
 
 type UploadStatus = 'uploading' | 'error';
 
@@ -48,52 +52,58 @@ export default function PendingUploadsPage() {
     setTakes((prev) => prev.filter((t) => t.id !== id));
   }
 
-  const byWord = new Map<string, PendingTake[]>();
-  for (const t of takes) {
-    const list = byWord.get(t.word) ?? [];
-    list.push(t);
-    byWord.set(t.word, list);
-  }
+  const groups = groupByWord(takes);
 
   return (
-    <main>
-      <h1>Bekleyen Kayıtlar</h1>
-      <p>
-        {takes.length} kayıt, {byWord.size} kelime bekliyor.
-      </p>
-      <button onClick={handleUploadAll} disabled={uploadingAll || takes.length === 0}>
-        {uploadingAll ? 'Yükleniyor...' : 'Tümünü Yükle'}
-      </button>
+    <main className="page">
+      <AppBar title="Bekleyen Kayıtlar" backTo="/kayit" />
+      <div className="page-body">
+        <p className="muted">
+          {takes.length} kayıt, {groups.length} kelime bekliyor.
+        </p>
+        <FilledButton
+          icon={uploadingAll ? 'hourglass_top' : 'cloud_upload'}
+          onClick={handleUploadAll}
+          disabled={uploadingAll || takes.length === 0}
+        >
+          {uploadingAll ? 'Yükleniyor...' : 'Tümünü Yükle'}
+        </FilledButton>
 
-      {takes.length === 0 ? (
-        <p>Bekleyen kayıt yok.</p>
-      ) : (
-        Array.from(byWord.entries()).map(([w, wTakes]) => (
-          <section key={w}>
-            <h2>{w}</h2>
-            {wTakes.map((take, i) => (
-              <div key={take.id}>
-                <span>Tekrar {i + 1}</span>
-                {status[take.id] === 'uploading' && <span>Yükleniyor...</span>}
-                {status[take.id] === 'error' && <span title={errors[take.id]}>Hata</span>}
-                {status[take.id] !== 'uploading' && (
-                  <>
-                    <button
-                      onClick={() => {
-                        const audio = new Audio(URL.createObjectURL(take.audioBlob));
-                        void audio.play();
-                      }}
-                    >
-                      Oynat
-                    </button>
-                    <button onClick={() => handleDelete(take.id)}>Sil</button>
-                  </>
-                )}
-              </div>
-            ))}
-          </section>
-        ))
-      )}
+        {takes.length === 0 ? (
+          <p className="empty">Bekleyen kayıt yok.</p>
+        ) : (
+          groups.map(([w, wTakes]) => (
+            <section key={w}>
+              <h2 className="take-group-title">{w}</h2>
+              {wTakes.map((take, i) => (
+                <div key={take.id} className="take-row">
+                  <span>Tekrar {i + 1}</span>
+                  {status[take.id] === 'uploading' ? (
+                    <span className="spinner" role="status" aria-label="Yükleniyor" />
+                  ) : (
+                    <>
+                      {status[take.id] === 'error' && (
+                        <span className="error-icon" role="img" aria-label="Hata" title={errors[take.id]}>
+                          <Icon name="error_outline" size={20} />
+                        </span>
+                      )}
+                      <IconButton
+                        icon="play_arrow"
+                        label="Oynat"
+                        onClick={() => {
+                          const audio = new Audio(URL.createObjectURL(take.audioBlob));
+                          void audio.play();
+                        }}
+                      />
+                      <IconButton icon="delete_outline" label="Sil" onClick={() => handleDelete(take.id)} />
+                    </>
+                  )}
+                </div>
+              ))}
+            </section>
+          ))
+        )}
+      </div>
     </main>
   );
 }
```

```diff
diff --git a/web/src/pages/AyarlarLoginPage.tsx b/web/src/pages/AyarlarLoginPage.tsx
index f7e80362..f3574405 100644
--- a/web/src/pages/AyarlarLoginPage.tsx
+++ b/web/src/pages/AyarlarLoginPage.tsx
@@ -1,4 +1,7 @@
 import { useState, type FormEvent } from 'react';
+import AppBar from '../components/AppBar';
+import Icon from '../components/Icon';
+import { FilledButton } from '../components/ui';
 
 const ADMIN_USERNAME = 'admin';
 const ADMIN_PASSWORD = 'adminpassword';
@@ -18,23 +21,35 @@ export default function AyarlarLoginPage({ onSuccess }: { onSuccess: () => void
   }
 
   return (
-    <main>
-      <h1>Ayarlar Girişi</h1>
-      <form onSubmit={handleSubmit}>
-        <label>
-          Kullanıcı Adı
-          <input value={username} onChange={(e) => setUsername(e.target.value)} />
+    <main className="page">
+      <AppBar title="Ayarlar Girişi" backTo="/" />
+      <form className="page-body" style={{ justifyContent: 'center' }} onSubmit={handleSubmit}>
+        <span className="settings-lock">
+          <Icon name="lock_outline" size={48} />
+        </span>
+        <label className="field">
+          <span className="field-label">Kullanıcı Adı</span>
+          <Icon name="person_outline" />
+          <input value={username} autoComplete="username" onChange={(e) => setUsername(e.target.value)} />
         </label>
-        <label>
-          Şifre
+        <label className="field">
+          <span className="field-label">Şifre</span>
+          <Icon name="key" />
           <input
             type="password"
+            autoComplete="current-password"
             value={password}
             onChange={(e) => setPassword(e.target.value)}
           />
         </label>
-        {error && <p role="alert">{error}</p>}
-        <button type="submit">Giriş Yap</button>
+        {error && (
+          <p role="alert" className="error-text">
+            {error}
+          </p>
+        )}
+        <FilledButton type="submit" icon="login" className="button--settings">
+          Giriş Yap
+        </FilledButton>
       </form>
     </main>
   );
```

Not: hata alan kayıtta kırmızı `error_outline` ikonu (hata metni `title`'da) + oynat/sil düğmeleri birlikte görünür (eski sayfada "Hata" yazısı + düğmeler; davranış aynı).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd web && npx tsc -b && npx vitest run src/pages/PendingUploadsPage.test.tsx src/pages/AyarlarLoginPage.test.tsx`
Expected: tsc çıktısız; `Tests  2 passed (2)`.

- [ ] **Step 5: Commit**

```bash
git add web/src/pages/PendingUploadsPage.tsx web/src/pages/PendingUploadsPage.test.tsx web/src/pages/AyarlarLoginPage.tsx web/src/pages/AyarlarLoginPage.test.tsx
git commit -m "feat: give pending uploads and settings login the Flutter layout

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Ayarlar sayfası

**Files:**
- Modify: `web/src/pages/AyarlarPage.tsx` (import'lar, `isErrorMessage`, JSX; `loadAll`, `handleSave`, `handleRebuildVoiceProfile` aynen)
- Test: `web/src/pages/AyarlarPage.test.tsx`

**Interfaces:**
- Consumes: `AppBar`, `FilledButton`, `OutlinedButton`, `ResultTile`, `SectionHeader`, `SegmentedButton` (Task 2); `withCurrent` (Task 1).

- [ ] **Step 1: Write the failing test**

`web/src/pages/AyarlarPage.test.tsx`:

```tsx
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/pages/AyarlarPage.test.tsx`
Expected: FAIL — `expected '<p>Yükleniyor...</p>' to contain 'Model Ayarları'`.

- [ ] **Step 3: Implement**

Worktree kökünde `git apply`:

```diff
diff --git a/web/src/pages/AyarlarPage.tsx b/web/src/pages/AyarlarPage.tsx
index 6c891764..7829eb94 100644
--- a/web/src/pages/AyarlarPage.tsx
+++ b/web/src/pages/AyarlarPage.tsx
@@ -9,10 +9,16 @@ import {
   type SettingsResponse,
   type VoiceProfileStatus,
 } from '../lib/api';
+import { withCurrent } from '../lib/uiFormat';
+import AppBar from '../components/AppBar';
+import { FilledButton, OutlinedButton, ResultTile, SectionHeader, SegmentedButton } from '../components/ui';
 
 const MODEL_OPTIONS = ['Furkan', 'base', 'large'];
 const ALGO_OPTIONS = ['lora', 'qlora', 'full'];
 
+// handleSave / handleRebuildVoiceProfile hata mesajlari bu kaliplarla baslar
+const isErrorMessage = (m: string) => m.startsWith('Hata:') || m.startsWith('Ses profili güncellenemedi');
+
 export default function AyarlarPage() {
   const [loading, setLoading] = useState(true);
   const [saving, setSaving] = useState(false);
@@ -81,103 +87,120 @@ export default function AyarlarPage() {
     }
   }
 
-  if (loading) return <p>Yükleniyor...</p>;
-  if (error) return <p role="alert">{error}</p>;
-  if (!settings) return null;
-
   return (
-    <main>
-      <h1>Ayarlar</h1>
-      {modelInfo && (
-        <p>
-          Model: {modelInfo.model} • WER: {modelInfo.wer} • Veri: {modelInfo.sample_count} ses
-        </p>
-      )}
-
-      <label>
-        Model
-        <select
-          value={settings.model}
-          onChange={(e) => setSettings({ ...settings, model: e.target.value })}
-        >
-          {MODEL_OPTIONS.map((m) => (
-            <option key={m} value={m}>
-              {m}
-            </option>
-          ))}
-        </select>
-      </label>
-
-      <label>
-        Algoritma
-        <select
-          value={settings.algorithm}
-          onChange={(e) => setSettings({ ...settings, algorithm: e.target.value })}
-        >
-          {ALGO_OPTIONS.map((a) => (
-            <option key={a} value={a}>
-              {a}
-            </option>
-          ))}
-        </select>
-      </label>
-
-      <label>
-        <input
-          type="checkbox"
-          checked={settings.self_learning}
-          onChange={(e) => setSettings({ ...settings, self_learning: e.target.checked })}
-        />
-        Kendi kendine öğrenme
-      </label>
-
-      <label>
-        Öğrenme oranı
-        <input
-          type="number"
-          step="0.00001"
-          value={settings.learning_rate}
-          onChange={(e) => setSettings({ ...settings, learning_rate: Number(e.target.value) })}
-        />
-      </label>
-
-      <label>
-        Epoch
-        <input
-          type="number"
-          value={settings.epochs}
-          onChange={(e) => setSettings({ ...settings, epochs: Number(e.target.value) })}
-        />
-      </label>
-
-      <label>
-        Batch size
-        <input
-          type="number"
-          value={settings.batch_size}
-          onChange={(e) => setSettings({ ...settings, batch_size: Number(e.target.value) })}
-        />
-      </label>
-
-      <button onClick={handleSave} disabled={saving}>
-        {saving ? 'Kaydediliyor...' : 'Kaydet'}
-      </button>
-
-      <section>
-        <h2>Ses Profili</h2>
-        <p>
-          {voiceProfile
-            ? voiceProfile.exists
-              ? `${voiceProfile.sample_count} kayıttan oluşturuldu${voiceProfile.created_at ? ` • ${voiceProfile.created_at}` : ''}`
-              : 'Henüz ses profili oluşturulmadı.'
-            : 'Durum yükleniyor...'}
-        </p>
-        <button onClick={handleRebuildVoiceProfile} disabled={voiceProfileBusy}>
-          {voiceProfileBusy ? 'Güncelleniyor...' : 'Ses Profilini Güncelle'}
-        </button>
-      </section>
-
-      {message && <p>{message}</p>}
+    <main className="page settings">
+      <AppBar title="Model Ayarları" backTo="/" />
+      <div className="page-body">
+        {loading ? (
+          <div className="spinner spinner--large" role="status" aria-label="Yükleniyor" />
+        ) : error ? (
+          <ResultTile label="Hata" value={error} tone="error" />
+        ) : settings ? (
+          <>
+            {modelInfo && (
+              <ResultTile
+                label="Model"
+                tone="settings"
+                value={`Model: ${modelInfo.model} • WER: ${modelInfo.wer} • Veri: ${modelInfo.sample_count} ses`}
+              />
+            )}
+
+            <SectionHeader>Model Tercihi</SectionHeader>
+            <SegmentedButton
+              label="Model Tercihi"
+              className="segmented--settings"
+              options={withCurrent(MODEL_OPTIONS, settings.model).map((m) => ({ value: m, label: m }))}
+              value={settings.model}
+              onChange={(model) => setSettings({ ...settings, model })}
+            />
+
+            <SectionHeader>Fine-Tune Algoritması</SectionHeader>
+            <SegmentedButton
+              label="Fine-Tune Algoritması"
+              className="segmented--settings"
+              options={withCurrent(ALGO_OPTIONS, settings.algorithm).map((a) => ({ value: a, label: a.toUpperCase() }))}
+              value={settings.algorithm}
+              onChange={(algorithm) => setSettings({ ...settings, algorithm })}
+            />
+
+            <SectionHeader>Kendi Kendine Öğrenme</SectionHeader>
+            <label className="switch-row">
+              <span style={{ flex: 1 }}>
+                Self-learning aktif
+                <br />
+                <span className="muted small">
+                  {settings.self_learning ? 'Yeni kayıtlar otomatik modele eklenir' : 'Manuel eğitim gerektirir'}
+                </span>
+              </span>
+              <input
+                type="checkbox"
+                role="switch"
+                checked={settings.self_learning}
+                onChange={(e) => setSettings({ ...settings, self_learning: e.target.checked })}
+              />
+            </label>
+
+            <SectionHeader>Hiperparametreler</SectionHeader>
+            <label className="field">
+              <span className="field-label">Learning Rate</span>
+              <input
+                type="number"
+                step="0.00001"
+                value={settings.learning_rate}
+                onChange={(e) => setSettings({ ...settings, learning_rate: Number(e.target.value) })}
+              />
+            </label>
+            <label className="field">
+              <span className="field-label">Epoch Sayısı</span>
+              <input
+                type="number"
+                value={settings.epochs}
+                onChange={(e) => setSettings({ ...settings, epochs: Number(e.target.value) })}
+              />
+            </label>
+            <label className="field">
+              <span className="field-label">Batch Size</span>
+              <input
+                type="number"
+                value={settings.batch_size}
+                onChange={(e) => setSettings({ ...settings, batch_size: Number(e.target.value) })}
+              />
+            </label>
+
+            <FilledButton
+              icon={saving ? 'hourglass_top' : 'save'}
+              className="button--settings"
+              onClick={handleSave}
+              disabled={saving}
+            >
+              {saving ? 'Kaydediliyor...' : 'Ayarları Kaydet'}
+            </FilledButton>
+
+            <SectionHeader>Ses Profili</SectionHeader>
+            <ResultTile
+              label="Durum"
+              tone="settings"
+              value={
+                voiceProfile
+                  ? voiceProfile.exists
+                    ? `${voiceProfile.sample_count} kayıttan oluşturuldu${voiceProfile.created_at ? ` • ${voiceProfile.created_at}` : ''}`
+                    : 'Henüz ses profili oluşturulmadı.'
+                  : 'Durum yükleniyor...'
+              }
+            />
+            <OutlinedButton
+              icon={voiceProfileBusy ? 'hourglass_top' : 'record_voice_over'}
+              className="button--settings"
+              onClick={handleRebuildVoiceProfile}
+              disabled={voiceProfileBusy}
+            >
+              {voiceProfileBusy ? 'Güncelleniyor...' : 'Ses Profilini Güncelle'}
+            </OutlinedButton>
+
+            {message && <ResultTile label="Durum" value={message} tone={isErrorMessage(message) ? 'error' : 'settings'} />}
+          </>
+        ) : null}
+      </div>
     </main>
   );
 }
```

Notlar: yükleme ve yükleme hatası artık `AppBar`'lı sayfa içinde (eskiden çıplak `<p>`); model ve algoritma `<select>` → `SegmentedButton` (`withCurrent` ile sunucudaki liste dışı değer seçili görünür); hiperparametreler bugünkü sayı alanları olarak kalır; mesaj `isErrorMessage` ise kırmızı `ResultTile`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx tsc -b && npx vitest run`
Expected: tsc çıktısız; `Test Files  18 passed (18)`, `Tests  88 passed (88)`.

- [ ] **Step 5: Commit**

```bash
git add web/src/pages/AyarlarPage.tsx web/src/pages/AyarlarPage.test.tsx
git commit -m "feat: give the settings page the Flutter layout

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Çevrimdışı sayfa ve telefon boyutunda görsel kontrol

**Files:**
- Modify: `web/public/offline.html` (aynı koyu tema, satır içi stil + `wifi_off` SVG; dış kaynak yok)
- Test: mevcut `web/src/shell.test.ts` (çevrimdışı sayfa dış kaynak kullanmıyor, canlı adresi deniyor) + görsel kontrol (commit edilmeyen betik)

- [ ] **Step 1: Write the offline page**

`web/public/offline.html`:

```html
<!doctype html>
<html lang="tr">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#131318" />
    <title>Furkanca</title>
    <style>
      body {
        margin: 0;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 16px;
        padding: 24px;
        box-sizing: border-box;
        background: #131318;
        color: #e5e1e9;
        font-family: Roboto, system-ui, sans-serif;
        text-align: center;
      }
      h1 { margin: 0; font-size: 24px; font-weight: 800; color: #c5c0ff; }
      p { margin: 0; font-size: 14px; color: rgba(229, 225, 233, 0.6); }
      svg { color: #6c63ff; }
      button {
        width: 100%;
        max-width: 360px;
        padding: 16px;
        border: none;
        border-radius: 14px;
        background: #6c63ff;
        color: #fff;
        font: inherit;
        font-size: 16px;
        font-weight: 700;
      }
    </style>
  </head>
  <body>
    <svg width="64" height="64" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M23.64 7c-.45-.34-4.93-4-11.64-4-1.5 0-2.89.19-4.15.48L18.18 13.8 23.64 7zm-6.6 8.22L3.27 1.44 2 2.72l2.05 2.06C1.91 5.76.59 6.82.36 7l11.63 14.49.01.01.01-.01 3.9-4.86 3.32 3.32 1.27-1.27-3.46-3.46z" />
    </svg>
    <h1>İnternet bağlantısı yok</h1>
    <p>Furkanca açılmak için internete bağlanmalı.</p>
    <button type="button" onclick="location.href = 'https://app.furkanca.algorixstudio.com'">
      Tekrar dene
    </button>
  </body>
</html>
```

- [ ] **Step 2: Run the shell test**

Run: `cd web && npx vitest run src/shell.test.ts`
Expected: `Tests  3 passed (3)` (metin, canlı adres ve "dış kaynak yok" kuralı korunur).

- [ ] **Step 3: Build to the plan workspace and serve it (NOT `web/dist`)**

`W` = bu planın çalışma alanı (`sdd-workspace` çıktısı).

```bash
cd web && npx vite build --outDir "$W/preview-dist" --emptyOutDir
```

Arka planda: `cd web && npx vite preview --outDir "$W/preview-dist" --port 4179 --strictPort`
Expected: `curl -s -o /dev/null -w "%{http_code}" http://localhost:4179/kayit/bekleyen` → `200`.

- [ ] **Step 4: Save the screenshot driver**

`$W/cdp.mjs` (commit edilmez; Playwright'ın indirdiği Chromium'u DevTools protokolüyle sürer):

```js
// Kullanim: node cdp.mjs <url> <out.png> [js-adimlari...]  (her adim sayfada calisir, 1.5 sn arayla)
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
const [url, out, ...steps] = process.argv.slice(2);
const chrome = spawn(process.env.CHROME, ['--headless', '--no-sandbox', '--disable-gpu', '--hide-scrollbars',
  '--remote-debugging-port=9333', '--window-size=412,915', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch('http://127.0.0.1:9333/json')).json()).find((t) => t.type === 'page'); } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (pending.has(d.id)) { pending.get(d.id)(d.result); pending.delete(d.id); } };
const send = (method, params = {}) => new Promise((r) => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
await send('Emulation.setDeviceMetricsOverride', { width: 412, height: 915, deviceScaleFactor: 1, mobile: true });
await send('Page.navigate', { url });
await sleep(3000);
for (const s of steps) {
  const r = await send('Runtime.evaluate', { expression: s, awaitPromise: true, returnByValue: true });
  if (r?.result?.value !== undefined) console.log('=>', JSON.stringify(r.result.value));
  if (r?.exceptionDetails) console.log('!!', r.exceptionDetails.exception?.description);
  await sleep(1500);
}
const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
writeFileSync(out, Buffer.from(shot.data, 'base64'));
chrome.kill();
process.exit(0);
```

- [ ] **Step 5: Take the screenshots and check the layout**

```bash
export CHROME=$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome | tail -1)
N="node --experimental-websocket $W/cdp.mjs"
SET='window.setv=(el,v)=>{const p=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,"value");p.set.call(el,v);el.dispatchEvent(new Event("input",{bubbles:true}));};1'
ADD='new Promise(r=>{const q=indexedDB.open("furkanca-pending-recordings",1);q.onsuccess=()=>{const tx=q.result.transaction("takes","readwrite");const st=tx.objectStore("takes");const b=new Blob(["x"]);for(const w of ["havuç","havuç","Bugün kendimi iyi hissediyorum ve dışarı çıkmak istiyorum.","su"])st.add({word:w,audioBlob:b,recordedAt:"2026-10-13"});tx.oncomplete=()=>r(1)}})'
for u in / /furkanca /kayit /ayarlar; do echo "$u"; $N "http://localhost:4179$u" "$W/shot$(echo $u | tr / _).png" 'document.documentElement.scrollWidth'; done
$N http://localhost:4179/kayit "$W/shot_cumle.png" '[...document.querySelectorAll(".segmented-option")][1].click();1' 'new Promise(r=>setTimeout(()=>r(document.documentElement.scrollWidth),3000))'
$N http://localhost:4179/kayit/bekleyen "$W/shot_bekleyen.png" "$ADD" 'location.reload();1' 'new Promise(r=>setTimeout(()=>r([document.documentElement.scrollWidth,document.querySelectorAll(".take-row").length]),2500))'
$N http://localhost:4179/ayarlar "$W/shot_ayarlar_ic.png" "$SET" 'const i=document.querySelectorAll("input");setv(i[0],"admin");setv(i[1],"adminpassword");1' 'document.querySelector("button[type=submit]").click();1' 'new Promise(r=>setTimeout(()=>r([document.documentElement.scrollWidth,[...document.querySelectorAll("[aria-pressed=true]")].map(b=>b.textContent)]),3000))'
```

Expected: her `scrollWidth` `412`; bekleyen satır sayısı `4`; Ayarlar'da seçili `["whisper","LORA"]` (canlı sunucu ayarı). Ekran görüntülerini aç ve Flutter düzeniyle karşılaştır: ana sayfada mor logo + üç renkli düğme; Kayıt'ta klasör ikonu, ilerleme çubuğu, Kelime|Cümle segmenti, kelime kartı, mor "Kaydet (Cihazda Sakla)"; cümle modunda büyük "Okumaya Başla" + "Atla"; Bekleyen'de uzun cümle satır kırarak; Ayarlar'da turuncu kartlar ve düğmeler. Önizleme sunucusunu kapat.

- [ ] **Step 6: Run the full suite and commit**

Run: `cd web && npx tsc -b && npx vitest run`
Expected: tsc çıktısız; `Test Files  18 passed (18)`, `Tests  88 passed (88)`.

```bash
git add web/public/offline.html
git commit -m "feat: match the offline page to the dark Flutter theme

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

## Dağıtım (plan dışında, kullanıcı onayıyla)

1. `worktree-react-rewrite` push (merge yok).
2. `cd .claude/worktrees/react-rewrite/web && npm run build` — canlı site; web servisi yeniden başlatılmaz (`serve.json` değişmedi). `deploy-web.sh` kullanılmaz.
3. Furkan uygulamayı kapatıp açar; APK kurulmaz. `offline.html` değişikliği bir sonraki native derlemeyle telefona gider.

