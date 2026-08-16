import 'fake-indexeddb/auto';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  _resetForTests,
  addTake,
  getAllTakes,
  getPendingWords,
  getTakesForWord,
  removeTake,
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
});
