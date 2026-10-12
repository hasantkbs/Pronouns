import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { nextVersion, readVersionCode, writeVersion } from './apk-version.mjs';

const tempFile = () => join(mkdtempSync(join(tmpdir(), 'apk-version-')), 'version.json');

describe('APK sürüm numarası', () => {
  it('starts from 1 when there is no version file, so the first build is 2', () => {
    const file = tempFile();
    expect(readVersionCode(file)).toBe(1);
    expect(nextVersion(readVersionCode(file)).versionCode).toBe(2);
  });

  it('treats a broken version file as version 1', () => {
    const file = tempFile();
    writeFileSync(file, '{bozuk');
    expect(readVersionCode(file)).toBe(1);
  });

  it('increments the stored version and derives the name', () => {
    const now = new Date('2026-10-12T12:00:00Z');
    expect(nextVersion(3, now)).toEqual({
      versionCode: 4,
      versionName: '1.3',
      built_at: '2026-10-12T12:00:00.000Z',
    });
  });

  it('reads back what it writes', () => {
    const file = tempFile();
    writeVersion(file, nextVersion(1));
    expect(readVersionCode(file)).toBe(2);
  });
});
