import { fetchWithTimeout, TRANSLATE_TIMEOUT_MS } from './deadlines';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
const API_KEY = import.meta.env.VITE_API_KEY;
const USER_ID = 'FurkanV1';

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  return { 'X-API-Key': API_KEY, ...extra };
}

export interface SettingsResponse {
  user_id: string;
  model: string;
  algorithm: string;
  self_learning: boolean;
  learning_rate: number;
  epochs: number;
  batch_size: number;
}

export async function getSettings(): Promise<SettingsResponse> {
  const res = await fetch(`${API_BASE_URL}/settings?user_id=${USER_ID}`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Ayarlar alınamadı: ${res.status}`);
  return res.json();
}

export async function saveSettings(
  payload: Omit<SettingsResponse, 'user_id'>,
): Promise<SettingsResponse> {
  const res = await fetch(`${API_BASE_URL}/settings`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ user_id: USER_ID, ...payload }),
  });
  if (!res.ok) throw new Error(`Ayarlar kaydedilemedi: ${res.status}`);
  return res.json();
}

export interface ModelInfoResponse {
  user_id: string;
  model: string;
  sample_count: number;
  wer: number;
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE_URL}/model/info?user_id=${USER_ID}`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Model bilgisi alınamadı: ${res.status}`);
  return res.json();
}

export interface VoiceProfileStatus {
  exists: boolean;
  sample_count: number;
  created_at: string | null;
}

export async function getVoiceProfileStatus(): Promise<VoiceProfileStatus> {
  const res = await fetch(`${API_BASE_URL}/voice-profile/status?user_id=${USER_ID}`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Ses profili durumu alınamadı: ${res.status}`);
  return res.json();
}

export async function rebuildVoiceProfile(): Promise<{ sample_count: number }> {
  const res = await fetch(`${API_BASE_URL}/voice-profile/rebuild?user_id=${USER_ID}`, {
    method: 'POST',
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Ses profili güncellenemedi: ${res.status}`);
  return res.json();
}

export interface WordSetsResponse {
  sets: string[];
}

export async function getWordSets(): Promise<WordSetsResponse> {
  const res = await fetch(`${API_BASE_URL}/collect/word-sets`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Kelime setleri alınamadı: ${res.status}`);
  return res.json();
}

export interface NextWordResponse {
  done?: boolean;
  word?: string;
  rep?: number;
  current_count?: number;
  ideal_repetitions?: number;
  total_words?: number;
  set_file?: string;
  error?: string;
}

export async function getNextWord(setFile: string, exclude: string[]): Promise<NextWordResponse> {
  const params = new URLSearchParams({
    user_id: USER_ID,
    set_file: setFile,
    exclude: exclude.join(','),
  });
  const res = await fetch(`${API_BASE_URL}/collect/next-word?${params.toString()}`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Sonraki kelime alınamadı: ${res.status}`);
  const data = (await res.json()) as NextWordResponse;
  if (data.error) throw new Error(data.error);
  return data;
}

export interface ProgressResponse {
  completed_words: number;
  total_words: number;
  ideal_repetitions: number;
}

export async function getProgress(setFile: string): Promise<ProgressResponse> {
  const params = new URLSearchParams({ user_id: USER_ID, set_file: setFile });
  const res = await fetch(`${API_BASE_URL}/collect/progress?${params.toString()}`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`İlerleme alınamadı: ${res.status}`);
  return res.json();
}

export async function uploadRecording(word: string, audioBlob: Blob): Promise<void> {
  const form = new FormData();
  form.append('user_id', USER_ID);
  form.append('word', word);
  form.append('audio', audioBlob, `${word}.wav`);
  const res = await fetch(`${API_BASE_URL}/record`, {
    method: 'POST',
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Sunucu hatası (${res.status}): ${body}`);
  }
}

export interface TranslateResponse {
  rejected?: boolean;
  reason?: string;
  response_text?: string;
  audio_url?: string | null;
  error?: string;
}

export async function translateSpeech(audioBlob: Blob): Promise<TranslateResponse> {
  const form = new FormData();
  form.append('user_id', USER_ID);
  form.append('audio', audioBlob, 'live.wav');
  // Sunucu takilirsa Furkanca dugmesi kilitli kalmasin
  const res = await fetchWithTimeout(
    `${API_BASE_URL}/translate`,
    { method: 'POST', headers: authHeaders(), body: form },
    TRANSLATE_TIMEOUT_MS,
  );
  const body = await res.text();
  if (!res.ok) throw new Error(`Sunucu hatası (${res.status}): ${body}`);
  const data = JSON.parse(body) as TranslateResponse;
  if (data.error) throw new Error(data.error);
  return data;
}

export function resolveUrl(pathOrUrl: string): string {
  return pathOrUrl.startsWith('http') ? pathOrUrl : `${API_BASE_URL}${pathOrUrl}`;
}
