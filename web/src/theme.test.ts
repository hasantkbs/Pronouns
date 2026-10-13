import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const css = readFileSync(new URL('./theme.css', import.meta.url), 'utf-8');

describe('theme.css', () => {
  it('stretches only the take label in pending rows, not the spinner or error icon', () => {
    // flex: 1 on every span made the 16px spinner a 170px rotating bar
    expect(css).not.toMatch(/\.take-row\s*>\s*span\s*\{/);
    expect(css).toMatch(/\.take-label\s*\{\s*flex:\s*1;/);
  });
});
