import { afterEach, describe, expect, it, vi } from 'vitest';
import { getNextWord, getProgress, getWordSets, uploadRecording } from './api';

function stubFetch(body: unknown) {
  const fetchMock = vi.fn(
    async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify(body), { status: 200 }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function sentQuery(fetchMock: ReturnType<typeof stubFetch>): URLSearchParams {
  const url = String(fetchMock.mock.calls[0][0]);
  return new URLSearchParams(url.split('?')[1] ?? '');
}

function sentForm(fetchMock: ReturnType<typeof stubFetch>): FormData {
  return fetchMock.mock.calls[0][1]?.body as FormData;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('kayıt türü (kind)', () => {
  it('getWordSets asks for sentence sets', async () => {
    const fetchMock = stubFetch({ sets: [] });
    await getWordSets('sentence');
    expect(sentQuery(fetchMock).get('kind')).toBe('sentence');
  });

  it('getWordSets defaults to word', async () => {
    const fetchMock = stubFetch({ sets: [] });
    await getWordSets();
    expect(sentQuery(fetchMock).get('kind')).toBe('word');
  });

  it('getNextWord joins sentence exclusions with newlines', async () => {
    const fetchMock = stubFetch({ done: true });
    await getNextWord('olcum_a.txt', ['Uykum geldi, yatmak istiyorum.', 'Su yok.'], 'sentence');
    const query = sentQuery(fetchMock);
    expect(query.get('kind')).toBe('sentence');
    expect(query.get('exclude')).toBe('Uykum geldi, yatmak istiyorum.\nSu yok.');
  });

  it('getNextWord keeps comma-separated word exclusions', async () => {
    const fetchMock = stubFetch({ done: true });
    await getNextWord('wordSet.txt', ['elma', 'armut']);
    const query = sentQuery(fetchMock);
    expect(query.get('kind')).toBe('word');
    expect(query.get('exclude')).toBe('elma,armut');
  });

  it('getProgress sends kind', async () => {
    const fetchMock = stubFetch({ completed_words: 0, total_words: 30, ideal_repetitions: 2 });
    await getProgress('olcum_a.txt', 'sentence');
    expect(sentQuery(fetchMock).get('kind')).toBe('sentence');
  });

  it('uploadRecording sends kind in the form', async () => {
    const fetchMock = stubFetch({});
    await uploadRecording('Su yok.', new Blob([new Uint8Array([1])]), 'sentence');
    const form = sentForm(fetchMock);
    expect(form.get('kind')).toBe('sentence');
    expect(form.get('word')).toBe('Su yok.');
  });

  it('uploadRecording defaults to word', async () => {
    const fetchMock = stubFetch({});
    await uploadRecording('elma', new Blob([new Uint8Array([1])]));
    expect(sentForm(fetchMock).get('kind')).toBe('word');
  });
});
