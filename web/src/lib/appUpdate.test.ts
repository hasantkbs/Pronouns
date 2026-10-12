import { describe, expect, it } from 'vitest';
import { checkForUpdate, needsUpdate, type UpdateDeps } from './appUpdate';

function deps(overrides: Partial<UpdateDeps> = {}): UpdateDeps {
  return {
    isNative: () => true,
    installedVersion: async () => 2,
    serverVersion: async () => 2,
    ...overrides,
  };
}

describe('needsUpdate', () => {
  it('is true only when the server has a newer build', () => {
    expect(needsUpdate(2, 3)).toBe(true);
    expect(needsUpdate(3, 3)).toBe(false);
    expect(needsUpdate(4, 3)).toBe(false);
  });

  it('is false when the installed build is unknown', () => {
    expect(needsUpdate(Number.NaN, 3)).toBe(false);
  });
});

describe('checkForUpdate', () => {
  it('offers an update when the server build is newer', async () => {
    expect(await checkForUpdate(deps({ serverVersion: async () => 3 }))).toBe(true);
  });

  it('does not offer an update for the same build', async () => {
    expect(await checkForUpdate(deps())).toBe(false);
  });

  it('never checks in a web browser', async () => {
    let asked = false;
    const result = await checkForUpdate(
      deps({
        isNative: () => false,
        serverVersion: async () => {
          asked = true;
          return 9;
        },
      }),
    );
    expect(result).toBe(false);
    expect(asked).toBe(false);
  });

  it('stays quiet when the version request fails', async () => {
    const failing = deps({
      serverVersion: async () => {
        throw new Error('ağ yok');
      },
    });
    expect(await checkForUpdate(failing)).toBe(false);
  });
});
