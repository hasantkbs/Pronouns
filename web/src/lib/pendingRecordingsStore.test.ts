import 'fake-indexeddb/auto';
import { openDB } from 'idb';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  _resetForTests,
  addTake,
  getAllTakes,
  getPendingWords,
  getTakesForWord,
  removeTake,
  takeKind,
} from './pendingRecordingsStore';

beforeEach(async () => {
  await _resetForTests();
});

describe('pendingRecordingsStore', () => {
  it('adds and retrieves a take', async () => {
    const blob = new Blob([new Uint8Array([1, 2, 3])], { type: 'audio/wav' });
    const take = await addTake('elma', blob);
    expect(take.word).toBe('elma');
    expect(typeof take.id).toBe('number');

    const all = await getAllTakes();
    expect(all).toHaveLength(1);
    expect(all[0].id).toBe(take.id);
  });

  it('filters takes by word', async () => {
    const blob = new Blob([new Uint8Array([1])], { type: 'audio/wav' });
    await addTake('armut', blob);
    await addTake('armut', blob);
    await addTake('muz', blob);

    const armutTakes = await getTakesForWord('armut');
    expect(armutTakes).toHaveLength(2);
    expect(armutTakes.every((t) => t.word === 'armut')).toBe(true);
  });

  it('removes a take', async () => {
    const blob = new Blob([new Uint8Array([1])], { type: 'audio/wav' });
    const take = await addTake('kiraz', blob);
    await removeTake(take.id);
    expect(await getAllTakes()).toHaveLength(0);
  });

  it('collects pending words as a set', async () => {
    const blob = new Blob([new Uint8Array([1])], { type: 'audio/wav' });
    await addTake('nar', blob);
    await addTake('nar', blob);
    await addTake('vişne', blob);
    expect(await getPendingWords()).toEqual(new Set(['nar', 'vişne']));
  });

  it('stores the kind of a take', async () => {
    const blob = new Blob([new Uint8Array([1])], { type: 'audio/wav' });
    await addTake('Su yok.', blob, 'sentence');
    const [take] = await getAllTakes();
    expect(take.kind).toBe('sentence');
    expect(takeKind(take)).toBe('sentence');
  });

  it('treats takes saved before kind existed as words', async () => {
    await getAllTakes(); // veritabanini olusturur
    const db = await openDB('furkanca-pending-recordings', 1);
    await db.add('takes', { word: 'elma', audioBlob: new Blob([]), recordedAt: '2026-10-01T00:00:00Z' });
    db.close();

    const [take] = await getAllTakes();
    expect(takeKind(take)).toBe('word');
    expect(await getPendingWords('word')).toEqual(new Set(['elma']));
    expect(await getPendingWords('sentence')).toEqual(new Set());
  });

  it('filters pending texts by kind', async () => {
    const blob = new Blob([new Uint8Array([1])], { type: 'audio/wav' });
    await addTake('nar', blob);
    await addTake('Su yok.', blob, 'sentence');
    expect(await getPendingWords('word')).toEqual(new Set(['nar']));
    expect(await getPendingWords('sentence')).toEqual(new Set(['Su yok.']));
  });
});
