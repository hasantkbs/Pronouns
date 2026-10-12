import { describe, expect, it } from 'vitest';
import { RequestGate } from './requestGate';

describe('RequestGate', () => {
  it('makes an older request stale when a newer one begins', () => {
    const gate = new RequestGate();
    const first = gate.begin();
    const second = gate.begin();
    expect(gate.isCurrent(first)).toBe(false);
    expect(gate.isCurrent(second)).toBe(true);
  });

  it('makes the in-flight request stale on invalidate', () => {
    const gate = new RequestGate();
    const token = gate.begin();
    gate.invalidate();
    expect(gate.isCurrent(token)).toBe(false);
    expect(gate.pending).toBe(false);
  });

  it('stays pending until the current request finishes', () => {
    const gate = new RequestGate();
    expect(gate.pending).toBe(false);
    const token = gate.begin();
    expect(gate.pending).toBe(true);
    gate.finish(token);
    expect(gate.pending).toBe(false);
  });

  it('ignores a stale request finishing while a newer one is pending', () => {
    const gate = new RequestGate();
    const old = gate.begin();
    gate.begin();
    gate.finish(old);
    expect(gate.pending).toBe(true);
  });
});
