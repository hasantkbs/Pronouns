// Furkanca sayfasinda hicbir bekleme sonsuz olmamali: sunucu ya da ses
// takilirsa dugme kilitli kalir ve Furkan tekrar deneyemez.

export const TRANSLATE_TIMEOUT_MS = 60000;
export const PLAYBACK_TIMEOUT_MS = 60000;
export const TIMEOUT_MESSAGE = 'Sunucu yanıt vermedi, tekrar dene';

export async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs: number,
  fetchImpl: typeof fetch = fetch,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetchImpl(input, { ...init, signal: controller.signal });
  } catch (e) {
    if (controller.signal.aborted) throw new Error(TIMEOUT_MESSAGE);
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

// HTMLAudioElement'in olay isleyicileri (this, ev: Event) alir; burada parametreler kullanilmaz.
type Handler = ((...args: never[]) => unknown) | null;

export interface PlayableAudio {
  onended: Handler;
  onerror: Handler;
  play(): Promise<void>;
  pause(): void;
}

/** Ses bitince, hata verince ya da sure dolunca (durdurup) cozulur; asla reddedilmez. */
export function playAudio(audio: PlayableAudio, timeoutMs: number): Promise<void> {
  return new Promise<void>((resolve) => {
    const timer = setTimeout(() => {
      audio.pause();
      resolve();
    }, timeoutMs);
    const done = () => {
      clearTimeout(timer);
      resolve();
    };
    audio.onended = done;
    audio.onerror = done;
    audio.play().catch(done);
  });
}

/** Mikrofon acilirken sayfadan cikildiysa acilan mikrofonu hemen kapatir. */
export async function startUnlessCancelled<T extends { stop: () => void }>(
  start: () => Promise<T>,
  isActive: () => boolean,
): Promise<T | null> {
  const handle = await start();
  if (!isActive()) {
    handle.stop();
    return null;
  }
  return handle;
}
