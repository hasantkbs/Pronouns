import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import config from '../capacitor.config';

const LIVE_URL = 'https://app.furkanca.algorixstudio.com';
const publicFile = (name: string) => new URL(`../public/${name}`, import.meta.url);

describe('kabuk APK', () => {
  it('loads the UI from the live site with a local offline page', () => {
    expect(config.server?.url).toBe(LIVE_URL);
    expect(config.server?.errorPath).toBe('offline.html');
  });

  it('ships an offline page that retries the live site without external resources', () => {
    expect(existsSync(publicFile('offline.html'))).toBe(true);
    const html = readFileSync(publicFile('offline.html'), 'utf-8');
    expect(html).toContain('İnternet bağlantısı yok');
    expect(html).toContain(LIVE_URL);
    expect(html).not.toMatch(/(src|href)="https?:/);
  });

  it('asks the server to revalidate html on every open', () => {
    const serve = JSON.parse(readFileSync(publicFile('serve.json'), 'utf-8'));
    expect(serve.headers).toContainEqual({
      source: '**/*.html',
      headers: [{ key: 'Cache-Control', value: 'no-cache' }],
    });
  });
});
