import { afterEach, describe, it, expect, vi } from 'vitest';
import {
  fetchWithTimeout,
  playAudio,
  startUnlessCancelled,
  TIMEOUT_MESSAGE,
  type PlayableAudio,
} from './deadlines';

afterEach(() => {
  vi.useRealTimers();
});

function hangingFetch(): typeof fetch {
  // Hic cevap vermeyen sunucu: yalnizca iptal sinyaliyle sonlanir
  return ((_input: RequestInfo | URL, init?: RequestInit) =>
    new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
    })) as typeof fetch;
}

describe('fetchWithTimeout', () => {
  it('fails with a visible message when the server never answers', async () => {
    vi.useFakeTimers();
    const pending = fetchWithTimeout('http://x/translate', {}, 60000, hangingFetch());
    const assertion = expect(pending).rejects.toThrow(TIMEOUT_MESSAGE);
    await vi.advanceTimersByTimeAsync(60000);
    await assertion;
  });

  it('returns the response when the server answers in time', async () => {
    const ok = new Response('{}', { status: 200 });
    const res = await fetchWithTimeout('http://x', {}, 60000, (async () => ok) as typeof fetch);
    expect(res).toBe(ok);
  });

  it('passes through other network errors unchanged', async () => {
    const failing = (async () => {
      throw new TypeError('Failed to fetch');
    }) as typeof fetch;
    await expect(fetchWithTimeout('http://x', {}, 60000, failing)).rejects.toThrow('Failed to fetch');
  });
});

class FakeAudio implements PlayableAudio {
  onended: (() => void) | null = null;
  onerror: (() => void) | null = null;
  paused = false;
  constructor(private readonly playResult: Promise<void> = Promise.resolve()) {}
  play(): Promise<void> {
    return this.playResult;
  }
  pause(): void {
    this.paused = true;
  }
}

describe('playAudio', () => {
  it('resolves when playback ends', async () => {
    const audio = new FakeAudio();
    const done = playAudio(audio, 60000);
    audio.onended?.();
    await expect(done).resolves.toBeUndefined();
  });

  it('resolves when playback errors or play() is rejected', async () => {
    const erroring = new FakeAudio();
    const first = playAudio(erroring, 60000);
    erroring.onerror?.();
    await expect(first).resolves.toBeUndefined();

    const blocked = new FakeAudio(Promise.reject(new Error('NotAllowedError')));
    await expect(playAudio(blocked, 60000)).resolves.toBeUndefined();
  });

  it('gives up and pauses when playback stalls past the deadline', async () => {
    vi.useFakeTimers();
    const audio = new FakeAudio(new Promise<void>(() => {}));
    const done = playAudio(audio, 60000);
    await vi.advanceTimersByTimeAsync(60000);
    await expect(done).resolves.toBeUndefined();
    expect(audio.paused).toBe(true);
  });
});

describe('startUnlessCancelled', () => {
  it('returns the handle while still active', async () => {
    const handle = { stop: vi.fn() };
    await expect(startUnlessCancelled(async () => handle, () => true)).resolves.toBe(handle);
    expect(handle.stop).not.toHaveBeenCalled();
  });

  it('stops the microphone if the page was left while it was opening', async () => {
    const handle = { stop: vi.fn() };
    await expect(startUnlessCancelled(async () => handle, () => false)).resolves.toBeNull();
    expect(handle.stop).toHaveBeenCalledOnce();
  });
});
