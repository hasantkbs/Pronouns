# Furkanca React Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Flutter Furkanca app with a single React + TypeScript codebase (wrapped by Capacitor for Android) that serves both `https://app.furkanca.algorixstudio.com` (web) and an Android APK from the existing `/apk` endpoint, preserving all three panels (Furkanca, Kayıt, Ayarlar) and their exact backend contract.

**Architecture:** Vite + React + TypeScript app in `web/` (new directory at the Pronouns repo root, sibling to `lib/`). Pure-logic modules (VAD segmenter, WAV encoding, IndexedDB pending-recordings store) are ported 1:1 from the Dart originals and unit-tested with Vitest. UI pages call the existing FastAPI backend unchanged, over HTTPS via `https://furkanca.algorixstudio.com` (already live). Android packaging via Capacitor wraps the same web build; no native plugins needed because Capacitor's WebView supports `getUserMedia`/Web Audio directly.

**Tech Stack:** React 18, TypeScript, Vite, React Router, Vitest, `idb` (IndexedDB wrapper), Capacitor 6.

## Global Constraints

- Backend API (`api.py`) is **not modified** — every task consumes existing endpoints as-is.
- Production API base URL: `https://furkanca.algorixstudio.com` (already live, valid Let's Encrypt cert, verified reachable).
- Production web app base URL: `https://app.furkanca.algorixstudio.com` (new subdomain, created in Task 12).
- `X-API-Key` header required on every call except `/apk` and `/download/*`; current value: `c2a6dd1afc678eb93f0c46c8c1528e2b1c07676fbde0951f` (must match `config.API_SECRET_KEY` on the server).
- Single hardcoded user: `user_id=FurkanV1` on every relevant call (matches Flutter's `kUserId`).
- Ayarlar login is a plaintext client-side check: username `admin`, password `adminpassword` — no backend call, matches existing Flutter behavior exactly (not a real auth system).
- All new code lives under `web/` at the Pronouns repo root (`/home/mayasoft/app/Pronouns/web/`), never touching `lib/`, `android/`, or any backend file.
- Audio capture uses raw PCM16 via the Web Audio API only (`getUserMedia` + `ScriptProcessorNode`) — **not** `MediaRecorder`/WebM — because the VAD segmenter needs raw sample chunks and the backend expects real WAV bytes with no server-side transcoding.
- Visual styling is explicitly out of scope for this plan (deferred to a follow-up frontend-design pass per the spec) — pages use plain semantic HTML elements, no CSS framework, no visual polish.
- Target server: Node 20.20.2 / npm 10.8.2 already installed; nginx/certbot commands run passwordless via `/etc/sudoers.d/algorix-automation`.

---

## Task 1: Project Scaffold

**Files:**
- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/tsconfig.node.json`
- Create: `web/vite.config.ts`
- Create: `web/index.html`
- Create: `web/src/main.tsx`
- Create: `web/src/App.tsx`
- Create: `web/.env.production`
- Create: `web/.gitignore`

**Interfaces:**
- Produces: a working `npm run dev` / `npm run build` / `npm run test` toolchain that every later task builds on.

- [ ] **Step 1: Create the directory and package.json**

```bash
mkdir -p /home/mayasoft/app/Pronouns/web/src
cd /home/mayasoft/app/Pronouns/web
```

`web/package.json`:
```json
{
  "name": "furkanca-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "idb": "^8.0.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "fake-indexeddb": "^6.0.0",
    "typescript": "^5.5.4",
    "vite": "^5.4.0",
    "vitest": "^2.0.5"
  }
}
```

- [ ] **Step 2: Add TypeScript config**

`web/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

`web/tsconfig.node.json`:
```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 3: Add Vite + Vitest config**

`web/vite.config.ts`:
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'node',
  },
});
```

- [ ] **Step 4: Add HTML entry point and app bootstrap**

`web/index.html`:
```html
<!doctype html>
<html lang="tr">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Furkanca</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`web/src/main.tsx`:
```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

`web/src/App.tsx` (placeholder shell — routing added in Task 6):
```tsx
export default function App() {
  return <main>Furkanca</main>;
}
```

- [ ] **Step 5: Add production env file and gitignore**

`web/.env.production`:
```
VITE_API_BASE_URL=https://furkanca.algorixstudio.com
VITE_API_KEY=c2a6dd1afc678eb93f0c46c8c1528e2b1c07676fbde0951f
```

`web/.gitignore`:
```
node_modules
dist
.env.local
```

- [ ] **Step 6: Install dependencies and verify the toolchain**

```bash
cd /home/mayasoft/app/Pronouns/web
npm install
npm run build
```

Expected: `npm install` completes without errors; `npm run build` produces `web/dist/` with `index.html` and bundled assets, no TypeScript errors.

- [ ] **Step 7: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/
git commit -m "Scaffold React+Vite+TypeScript project for Furkanca web/Android rewrite"
```

---

## Task 2: VAD Engine Port (`lib/vad.ts`)

**Files:**
- Create: `web/src/lib/vad.ts`
- Test: `web/src/lib/vad.test.ts`

**Interfaces:**
- Produces: `calculateRms(samples: Int16Array): number`, `pcmChunkDurationMs(samples: Int16Array, sampleRate: number): number`, `pcmToWav(pcm: Int16Array, sampleRate: number, numChannels?: number): Blob`, `class VadSegmenter` with `addChunk(chunk: Int16Array): VadEvent | null`, `type VadEvent = { type: 'started' | 'ended' | 'discardedTooShort'; pcmData?: Int16Array }`. Used by Task 5 (audio capture), Task 9 (Kayıt), Task 10 (Furkanca).

- [ ] **Step 1: Write the failing tests**

`web/src/lib/vad.test.ts`:
```typescript
import { describe, it, expect } from 'vitest';
import { calculateRms, pcmToWav, VadSegmenter } from './vad';

function silentChunk(length = 160): Int16Array {
  return new Int16Array(length);
}

function loudChunk(length = 160, amplitude = 10000): Int16Array {
  const arr = new Int16Array(length);
  arr.fill(amplitude);
  return arr;
}

describe('calculateRms', () => {
  it('returns 0 for silence', () => {
    expect(calculateRms(silentChunk())).toBe(0);
  });

  it('returns a value above the default onset threshold for loud audio', () => {
    expect(calculateRms(loudChunk())).toBeGreaterThan(0.012);
  });
});

describe('pcmToWav', () => {
  it('produces a valid 16-bit mono WAV header', async () => {
    const pcm = Int16Array.from([0, 100, -100, 32767]);
    const blob = pcmToWav(pcm, 16000, 1);
    expect(blob.type).toBe('audio/wav');

    const buf = await blob.arrayBuffer();
    const view = new DataView(buf);
    const tag = (offset: number) =>
      String.fromCharCode(
        view.getUint8(offset),
        view.getUint8(offset + 1),
        view.getUint8(offset + 2),
        view.getUint8(offset + 3),
      );

    expect(tag(0)).toBe('RIFF');
    expect(tag(8)).toBe('WAVE');
    expect(view.getUint32(24, true)).toBe(16000); // sample rate
    expect(view.getUint16(22, true)).toBe(1); // channels
    expect(view.getUint16(34, true)).toBe(16); // bits per sample
    expect(view.getUint32(40, true)).toBe(8); // data size = 4 samples * 2 bytes
    expect(buf.byteLength).toBe(52); // 44 byte header + 8 byte data
  });
});

describe('VadSegmenter', () => {
  it('emits started on speech onset and ended after the silence timeout, including pre-roll', () => {
    const seg = new VadSegmenter({
      sampleRate: 16000,
      silenceTimeoutMs: 100,
      preRollMs: 20,
      minSpeechDurationMs: 5,
    });
    const chunkSamples = 160; // 10ms per chunk at 16kHz

    expect(seg.addChunk(silentChunk(chunkSamples))).toBeNull();
    expect(seg.addChunk(silentChunk(chunkSamples))).toBeNull();

    const startEvent = seg.addChunk(loudChunk(chunkSamples));
    expect(startEvent?.type).toBe('started');

    expect(seg.addChunk(loudChunk(chunkSamples))).toBeNull();

    let endEvent = null;
    for (let i = 0; i < 12; i++) {
      const ev = seg.addChunk(silentChunk(chunkSamples));
      if (ev) {
        endEvent = ev;
        break;
      }
    }

    expect(endEvent?.type).toBe('ended');
    expect(endEvent?.pcmData).toBeInstanceOf(Int16Array);
    expect(endEvent!.pcmData!.length).toBeGreaterThan(0);
  });

  it('discards segments shorter than minSpeechDurationMs', () => {
    const seg = new VadSegmenter({
      sampleRate: 16000,
      silenceTimeoutMs: 20,
      minSpeechDurationMs: 100,
    });
    const chunkSamples = 160; // 10ms per chunk

    expect(seg.addChunk(loudChunk(chunkSamples))?.type).toBe('started');
    expect(seg.addChunk(silentChunk(chunkSamples))).toBeNull();
    const ev = seg.addChunk(silentChunk(chunkSamples));
    expect(ev?.type).toBe('discardedTooShort');
  });

  it('caps a segment at maxDurationMs even without silence', () => {
    const seg = new VadSegmenter({
      sampleRate: 16000,
      maxDurationMs: 50,
      minSpeechDurationMs: 5,
    });
    const chunkSamples = 160; // 10ms per chunk

    expect(seg.addChunk(loudChunk(chunkSamples))?.type).toBe('started');
    let endEvent = null;
    for (let i = 0; i < 10; i++) {
      const ev = seg.addChunk(loudChunk(chunkSamples));
      if (ev) {
        endEvent = ev;
        break;
      }
    }
    expect(endEvent?.type).toBe('ended');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run test -- src/lib/vad.test.ts
```

Expected: FAIL — `Cannot find module './vad'`.

- [ ] **Step 3: Implement `vad.ts`**

`web/src/lib/vad.ts`:
```typescript
export function calculateRms(samples: Int16Array): number {
  if (samples.length === 0) return 0;
  let sumSquares = 0;
  for (let i = 0; i < samples.length; i++) {
    const normalized = samples[i] / 32768;
    sumSquares += normalized * normalized;
  }
  return Math.sqrt(sumSquares / samples.length);
}

export function pcmChunkDurationMs(samples: Int16Array, sampleRate: number): number {
  return (samples.length / sampleRate) * 1000;
}

export function pcmToWav(pcm: Int16Array, sampleRate: number, numChannels = 1): Blob {
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = pcm.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeString(offset: number, str: string) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < pcm.length; i++) {
    view.setInt16(offset, pcm[i], true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

export type VadEventType = 'started' | 'ended' | 'discardedTooShort';

export interface VadEvent {
  type: VadEventType;
  pcmData?: Int16Array;
}

export interface VadSegmenterOptions {
  onsetThreshold?: number;
  silenceTimeoutMs?: number;
  maxDurationMs?: number;
  minSpeechDurationMs?: number;
  preRollMs?: number;
  sampleRate?: number;
}

export class VadSegmenter {
  private onsetThreshold: number;
  private silenceTimeoutMs: number;
  private maxDurationMs: number;
  private minSpeechDurationMs: number;
  private preRollMs: number;
  private sampleRate: number;

  private preRoll: Int16Array[] = [];
  private preRollBufferedMs = 0;

  private capturing = false;
  private buffer: number[] = [];
  private capturedDurationMs = 0;
  private speechDurationMs = 0;
  private silenceElapsedMs = 0;

  constructor(options: VadSegmenterOptions = {}) {
    this.onsetThreshold = options.onsetThreshold ?? 0.012;
    this.silenceTimeoutMs = options.silenceTimeoutMs ?? 1000;
    this.maxDurationMs = options.maxDurationMs ?? 12000;
    this.minSpeechDurationMs = options.minSpeechDurationMs ?? 300;
    this.preRollMs = options.preRollMs ?? 300;
    this.sampleRate = options.sampleRate ?? 16000;
  }

  addChunk(chunk: Int16Array): VadEvent | null {
    const chunkDurationMs = pcmChunkDurationMs(chunk, this.sampleRate);
    const isSpeech = calculateRms(chunk) >= this.onsetThreshold;

    if (!this.capturing) {
      if (!isSpeech) {
        this.preRoll.push(chunk);
        this.preRollBufferedMs += chunkDurationMs;
        while (this.preRollBufferedMs > this.preRollMs && this.preRoll.length > 0) {
          const removed = this.preRoll.shift()!;
          this.preRollBufferedMs -= pcmChunkDurationMs(removed, this.sampleRate);
        }
        return null;
      }

      this.capturing = true;
      this.buffer = [];
      for (const c of this.preRoll) {
        this.buffer.push(...c);
      }
      this.buffer.push(...chunk);
      this.capturedDurationMs = this.preRollBufferedMs + chunkDurationMs;
      this.speechDurationMs = chunkDurationMs;
      this.silenceElapsedMs = 0;
      this.preRoll = [];
      this.preRollBufferedMs = 0;
      return { type: 'started' };
    }

    this.buffer.push(...chunk);
    this.capturedDurationMs += chunkDurationMs;
    if (isSpeech) {
      this.speechDurationMs += chunkDurationMs;
      this.silenceElapsedMs = 0;
    } else {
      this.silenceElapsedMs += chunkDurationMs;
    }

    const shouldEnd =
      this.silenceElapsedMs >= this.silenceTimeoutMs || this.capturedDurationMs >= this.maxDurationMs;
    if (!shouldEnd) return null;

    this.capturing = false;
    const pcm = Int16Array.from(this.buffer);
    const tooShort = this.speechDurationMs < this.minSpeechDurationMs;
    this.buffer = [];
    this.capturedDurationMs = 0;
    this.speechDurationMs = 0;
    this.silenceElapsedMs = 0;

    return tooShort ? { type: 'discardedTooShort' } : { type: 'ended', pcmData: pcm };
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run test -- src/lib/vad.test.ts
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/src/lib/vad.ts web/src/lib/vad.test.ts
git commit -m "Port VAD segmenter and WAV encoder from vad.dart to TypeScript"
```

---

## Task 3: Pending Recordings Store (IndexedDB)

**Files:**
- Create: `web/src/lib/pendingRecordingsStore.ts`
- Test: `web/src/lib/pendingRecordingsStore.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `interface PendingTake { id: number; word: string; audioBlob: Blob; recordedAt: string }`, `addTake(word: string, audioBlob: Blob, now?: Date): Promise<PendingTake>`, `removeTake(id: number): Promise<void>`, `getAllTakes(): Promise<PendingTake[]>`, `getTakesForWord(word: string): Promise<PendingTake[]>`, `getPendingWords(): Promise<Set<string>>`, `_resetForTests(): Promise<void>` (test-only). Used by Task 8 (Kayıt) and Task 9 (Pending Uploads).

- [ ] **Step 1: Write the failing tests**

`web/src/lib/pendingRecordingsStore.test.ts`:
```typescript
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run test -- src/lib/pendingRecordingsStore.test.ts
```

Expected: FAIL — `Cannot find module './pendingRecordingsStore'`.

- [ ] **Step 3: Implement the store**

`web/src/lib/pendingRecordingsStore.ts`:
```typescript
import { openDB, type IDBPDatabase } from 'idb';

export interface PendingTake {
  id: number;
  word: string;
  audioBlob: Blob;
  recordedAt: string;
}

const DB_NAME = 'furkanca-pending-recordings';
const STORE_NAME = 'takes';
const WORD_INDEX = 'by-word';

let dbPromise: Promise<IDBPDatabase> | null = null;

function getDb(): Promise<IDBPDatabase> {
  if (!dbPromise) {
    dbPromise = openDB(DB_NAME, 1, {
      upgrade(db) {
        const store = db.createObjectStore(STORE_NAME, {
          keyPath: 'id',
          autoIncrement: true,
        });
        store.createIndex(WORD_INDEX, 'word');
      },
    });
  }
  return dbPromise;
}

export async function addTake(
  word: string,
  audioBlob: Blob,
  now: Date = new Date(),
): Promise<PendingTake> {
  const db = await getDb();
  const record = { word, audioBlob, recordedAt: now.toISOString() };
  const id = (await db.add(STORE_NAME, record)) as number;
  return { id, ...record };
}

export async function removeTake(id: number): Promise<void> {
  const db = await getDb();
  await db.delete(STORE_NAME, id);
}

export async function getAllTakes(): Promise<PendingTake[]> {
  const db = await getDb();
  return db.getAll(STORE_NAME);
}

export async function getTakesForWord(word: string): Promise<PendingTake[]> {
  const db = await getDb();
  return db.getAllFromIndex(STORE_NAME, WORD_INDEX, word);
}

export async function getPendingWords(): Promise<Set<string>> {
  const takes = await getAllTakes();
  return new Set(takes.map((t) => t.word));
}

export async function _resetForTests(): Promise<void> {
  if (dbPromise) {
    const db = await dbPromise;
    db.close();
  }
  dbPromise = null;
  await new Promise<void>((resolve, reject) => {
    const req = indexedDB.deleteDatabase(DB_NAME);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
    req.onblocked = () => resolve();
  });
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run test -- src/lib/pendingRecordingsStore.test.ts
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/src/lib/pendingRecordingsStore.ts web/src/lib/pendingRecordingsStore.test.ts
git commit -m "Port offline pending-recordings queue from pending_recordings.dart to IndexedDB"
```

---

## Task 4: API Client

**Files:**
- Create: `web/src/lib/api.ts`
- Create: `web/src/vite-env.d.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `getSettings`, `saveSettings`, `getModelInfo`, `getVoiceProfileStatus`, `rebuildVoiceProfile`, `getWordSets`, `getNextWord`, `getProgress`, `uploadRecording`, `translateSpeech`, `resolveUrl`, plus the response types `SettingsResponse`, `ModelInfoResponse`, `VoiceProfileStatus`, `WordSetsResponse`, `NextWordResponse`, `ProgressResponse`, `TranslateResponse`. Used by Task 7 (Ayarlar), Task 8 (Kayıt), Task 9 (Pending Uploads), Task 10 (Furkanca).

- [ ] **Step 1: Declare Vite env types**

`web/src/vite-env.d.ts`:
```typescript
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_API_KEY: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

- [ ] **Step 2: Implement the API client**

`web/src/lib/api.ts`:
```typescript
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
  created_at: string;
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
  audio_url?: string;
}

export async function translateSpeech(audioBlob: Blob): Promise<TranslateResponse> {
  const form = new FormData();
  form.append('user_id', USER_ID);
  form.append('audio', audioBlob, 'live.wav');
  const res = await fetch(`${API_BASE_URL}/translate`, {
    method: 'POST',
    headers: authHeaders(),
    body: form,
  });
  const body = await res.text();
  if (!res.ok) throw new Error(`Sunucu hatası (${res.status}): ${body}`);
  return JSON.parse(body) as TranslateResponse;
}

export function resolveUrl(pathOrUrl: string): string {
  return pathOrUrl.startsWith('http') ? pathOrUrl : `${API_BASE_URL}${pathOrUrl}`;
}
```

Note: no automated test for this file — every function is a thin wrapper around `fetch` against the live backend, so it is covered by the manual end-to-end test pass in Task 13 instead of a mocked unit test.

- [ ] **Step 3: Verify it compiles**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run build
```

Expected: build succeeds with no TypeScript errors.

- [ ] **Step 4: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/src/lib/api.ts web/src/vite-env.d.ts
git commit -m "Add typed API client matching the existing FastAPI backend contract"
```

---

## Task 5: Audio Capture

**Files:**
- Create: `web/src/lib/audioCapture.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks (used alongside Task 2's `VadSegmenter`/`pcmToWav` by callers).
- Produces: `interface AudioCaptureHandle { sampleRate: number; stop: () => void }`, `startAudioCapture(onChunk: (chunk: Int16Array) => void, bufferSize?: number): Promise<AudioCaptureHandle>`. Used by Task 8 (Kayıt) and Task 10 (Furkanca).

- [ ] **Step 1: Implement audio capture**

`web/src/lib/audioCapture.ts`:
```typescript
export interface AudioCaptureHandle {
  sampleRate: number;
  stop: () => void;
}

export async function startAudioCapture(
  onChunk: (chunk: Int16Array) => void,
  bufferSize: 1024 | 2048 | 4096 = 4096,
): Promise<AudioCaptureHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });

  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

  processor.onaudioprocess = (event) => {
    const float32 = event.inputBuffer.getChannelData(0);
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const sample = Math.max(-1, Math.min(1, float32[i]));
      int16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    onChunk(int16);
  };

  // ScriptProcessorNode only fires onaudioprocess while connected to a
  // destination; we discard the (silent) output, we only need the input.
  source.connect(processor);
  processor.connect(audioContext.destination);

  const stop = () => {
    processor.disconnect();
    source.disconnect();
    stream.getTracks().forEach((track) => track.stop());
    void audioContext.close();
  };

  return { sampleRate: audioContext.sampleRate, stop };
}
```

No automated test: this module only wraps browser-only APIs (`getUserMedia`, `AudioContext`) that don't exist in the Vitest/Node test environment. It is covered by the manual microphone test in Task 13.

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run build
```

Expected: build succeeds with no TypeScript errors.

- [ ] **Step 3: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/src/lib/audioCapture.ts
git commit -m "Add raw PCM16 microphone capture via Web Audio API"
```

---

## Task 6: Routing + Home Page

**Files:**
- Create: `web/src/pages/HomePage.tsx`
- Modify: `web/src/App.tsx`

**Interfaces:**
- Consumes: nothing (page components created in later tasks are wired in as they land, see Step 3's TODO note resolved by Tasks 7–10).
- Produces: the top-level route table other tasks' pages plug into.

- [ ] **Step 1: Add `react-router-dom` routing and the home page**

`web/src/pages/HomePage.tsx`:
```tsx
import { Link } from 'react-router-dom';

export default function HomePage() {
  return (
    <main>
      <h1>Furkanca</h1>
      <p>Konuşma Asistanı</p>
      <nav>
        <ul>
          <li>
            <Link to="/furkanca">Furkanca</Link>
          </li>
          <li>
            <Link to="/kayit">Kayıt</Link>
          </li>
          <li>
            <Link to="/ayarlar">Ayarlar</Link>
          </li>
        </ul>
      </nav>
    </main>
  );
}
```

- [ ] **Step 2: Wire up the router in `App.tsx`**

`web/src/App.tsx` (final version — pages imported here are created in Tasks 7–10; until those tasks land, comment out the corresponding `<Route>` lines and leave only `HomePage`):

```tsx
import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import FurkancaPage from './pages/FurkancaPage';
import KayitPage from './pages/KayitPage';
import PendingUploadsPage from './pages/PendingUploadsPage';
import AyarlarLoginPage from './pages/AyarlarLoginPage';
import AyarlarPage from './pages/AyarlarPage';

export default function App() {
  const [ayarlarUnlocked, setAyarlarUnlocked] = useState(false);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/furkanca" element={<FurkancaPage />} />
        <Route path="/kayit" element={<KayitPage />} />
        <Route path="/kayit/bekleyen" element={<PendingUploadsPage />} />
        <Route
          path="/ayarlar"
          element={
            ayarlarUnlocked ? (
              <AyarlarPage />
            ) : (
              <AyarlarLoginPage onSuccess={() => setAyarlarUnlocked(true)} />
            )
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
```

Since the imported page modules don't exist yet, this task creates **stub** versions of the four not-yet-built pages so the build stays green; Tasks 7–10 replace each stub with the real implementation.

`web/src/pages/FurkancaPage.tsx` (stub, replaced in Task 10):
```tsx
export default function FurkancaPage() {
  return <main>Furkanca (yakında)</main>;
}
```

`web/src/pages/KayitPage.tsx` (stub, replaced in Task 8):
```tsx
export default function KayitPage() {
  return <main>Kayıt (yakında)</main>;
}
```

`web/src/pages/PendingUploadsPage.tsx` (stub, replaced in Task 9):
```tsx
export default function PendingUploadsPage() {
  return <main>Bekleyen Kayıtlar (yakında)</main>;
}
```

`web/src/pages/AyarlarLoginPage.tsx` (stub, replaced in Task 7):
```tsx
export default function AyarlarLoginPage({ onSuccess }: { onSuccess: () => void }) {
  return (
    <main>
      <button onClick={onSuccess}>Giriş (yakında)</button>
    </main>
  );
}
```

`web/src/pages/AyarlarPage.tsx` (stub, replaced in Task 7):
```tsx
export default function AyarlarPage() {
  return <main>Ayarlar (yakında)</main>;
}
```

- [ ] **Step 3: Add the router dependency and verify the build**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run build
npm run dev -- --port 5173 &
sleep 2
curl -sS -o /dev/null -w "HTTP %{http_code}\n" http://localhost:5173/
kill %1
```

Expected: build succeeds; dev server responds `HTTP 200` at `/`.

- [ ] **Step 4: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/src/App.tsx web/src/pages/
git commit -m "Add routing shell, home page, and page stubs for Furkanca web app"
```

---

## Task 7: Ayarlar (Login + Settings Page)

**Files:**
- Modify: `web/src/pages/AyarlarLoginPage.tsx`
- Modify: `web/src/pages/AyarlarPage.tsx`

**Interfaces:**
- Consumes: `getSettings`, `saveSettings`, `getModelInfo`, `getVoiceProfileStatus`, `rebuildVoiceProfile` from `../lib/api` (Task 4).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Implement the login gate**

`web/src/pages/AyarlarLoginPage.tsx`:
```tsx
import { useState, type FormEvent } from 'react';

const ADMIN_USERNAME = 'admin';
const ADMIN_PASSWORD = 'adminpassword';

export default function AyarlarLoginPage({ onSuccess }: { onSuccess: () => void }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (username === ADMIN_USERNAME && password === ADMIN_PASSWORD) {
      onSuccess();
    } else {
      setError('Kullanıcı adı veya şifre hatalı');
    }
  }

  return (
    <main>
      <h1>Ayarlar Girişi</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Kullanıcı Adı
          <input value={username} onChange={(e) => setUsername(e.target.value)} />
        </label>
        <label>
          Şifre
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        {error && <p role="alert">{error}</p>}
        <button type="submit">Giriş Yap</button>
      </form>
    </main>
  );
}
```

- [ ] **Step 2: Implement the settings page**

`web/src/pages/AyarlarPage.tsx`:
```tsx
import { useEffect, useState } from 'react';
import {
  getModelInfo,
  getSettings,
  getVoiceProfileStatus,
  rebuildVoiceProfile,
  saveSettings,
  type ModelInfoResponse,
  type SettingsResponse,
  type VoiceProfileStatus,
} from '../lib/api';

const MODEL_OPTIONS = ['Furkan', 'base', 'large'];
const ALGO_OPTIONS = ['lora', 'qlora', 'full'];

export default function AyarlarPage() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [voiceProfile, setVoiceProfile] = useState<VoiceProfileStatus | null>(null);
  const [voiceProfileBusy, setVoiceProfileBusy] = useState(false);

  useEffect(() => {
    void loadAll();
  }, []);

  async function loadAll() {
    setLoading(true);
    setError(null);
    try {
      const [s, info] = await Promise.all([getSettings(), getModelInfo()]);
      setSettings(s);
      setModelInfo(info);
    } catch (e) {
      setError(`Ayarlar yüklenemedi: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
    try {
      setVoiceProfile(await getVoiceProfileStatus());
    } catch {
      // ses profili durumu opsiyonel
    }
  }

  async function handleSave() {
    if (!settings) return;
    setSaving(true);
    try {
      const updated = await saveSettings({
        model: settings.model,
        algorithm: settings.algorithm,
        self_learning: settings.self_learning,
        learning_rate: settings.learning_rate,
        epochs: settings.epochs,
        batch_size: settings.batch_size,
      });
      setSettings(updated);
      setMessage('Ayarlar kaydedildi.');
    } catch (e) {
      setMessage(`Hata: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  }

  async function handleRebuildVoiceProfile() {
    setVoiceProfileBusy(true);
    try {
      const result = await rebuildVoiceProfile();
      setVoiceProfile(await getVoiceProfileStatus());
      setMessage(`Ses profili güncellendi (${result.sample_count} kayıt).`);
    } catch (e) {
      setMessage(`Ses profili güncellenemedi: ${(e as Error).message}`);
    } finally {
      setVoiceProfileBusy(false);
    }
  }

  if (loading) return <p>Yükleniyor...</p>;
  if (error) return <p role="alert">{error}</p>;
  if (!settings) return null;

  return (
    <main>
      <h1>Ayarlar</h1>
      {modelInfo && (
        <p>
          Model: {modelInfo.model} • WER: {modelInfo.wer} • Veri: {modelInfo.sample_count} ses
        </p>
      )}

      <label>
        Model
        <select
          value={settings.model}
          onChange={(e) => setSettings({ ...settings, model: e.target.value })}
        >
          {MODEL_OPTIONS.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </label>

      <label>
        Algoritma
        <select
          value={settings.algorithm}
          onChange={(e) => setSettings({ ...settings, algorithm: e.target.value })}
        >
          {ALGO_OPTIONS.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </select>
      </label>

      <label>
        <input
          type="checkbox"
          checked={settings.self_learning}
          onChange={(e) => setSettings({ ...settings, self_learning: e.target.checked })}
        />
        Kendi kendine öğrenme
      </label>

      <label>
        Öğrenme oranı
        <input
          type="number"
          step="0.00001"
          value={settings.learning_rate}
          onChange={(e) => setSettings({ ...settings, learning_rate: Number(e.target.value) })}
        />
      </label>

      <label>
        Epoch
        <input
          type="number"
          value={settings.epochs}
          onChange={(e) => setSettings({ ...settings, epochs: Number(e.target.value) })}
        />
      </label>

      <label>
        Batch size
        <input
          type="number"
          value={settings.batch_size}
          onChange={(e) => setSettings({ ...settings, batch_size: Number(e.target.value) })}
        />
      </label>

      <button onClick={handleSave} disabled={saving}>
        {saving ? 'Kaydediliyor...' : 'Kaydet'}
      </button>

      <section>
        <h2>Ses Profili</h2>
        <p>
          {voiceProfile
            ? voiceProfile.exists
              ? `${voiceProfile.sample_count} kayıttan oluşturuldu • ${voiceProfile.created_at}`
              : 'Henüz ses profili oluşturulmadı.'
            : 'Durum yükleniyor...'}
        </p>
        <button onClick={handleRebuildVoiceProfile} disabled={voiceProfileBusy}>
          {voiceProfileBusy ? 'Güncelleniyor...' : 'Ses Profilini Güncelle'}
        </button>
      </section>

      {message && <p>{message}</p>}
    </main>
  );
}
```

- [ ] **Step 3: Verify it compiles**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run build
```

Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/src/pages/AyarlarLoginPage.tsx web/src/pages/AyarlarPage.tsx
git commit -m "Implement Ayarlar login gate and settings/model-info/voice-profile page"
```

---

## Task 8: Kayıt Page

**Files:**
- Modify: `web/src/pages/KayitPage.tsx`

**Interfaces:**
- Consumes: `getWordSets`, `getNextWord`, `getProgress` from `../lib/api` (Task 4); `startAudioCapture` from `../lib/audioCapture` (Task 5); `pcmToWav` from `../lib/vad` (Task 2); `addTake`, `getAllTakes`, `getTakesForWord`, `getPendingWords` from `../lib/pendingRecordingsStore` (Task 3).
- Produces: nothing consumed by later tasks (Task 9 reads the same `pendingRecordingsStore` independently).

- [ ] **Step 1: Implement the Kayıt page**

`web/src/pages/KayitPage.tsx`:
```tsx
import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getNextWord, getProgress, getWordSets } from '../lib/api';
import { startAudioCapture } from '../lib/audioCapture';
import { pcmToWav } from '../lib/vad';
import { addTake, getAllTakes, getPendingWords, getTakesForWord } from '../lib/pendingRecordingsStore';

export default function KayitPage() {
  const [sets, setSets] = useState<string[]>([]);
  const [setFile, setSetFile] = useState('wordSet.txt');
  const [word, setWord] = useState<string | null>(null);
  const [currentCount, setCurrentCount] = useState(0);
  const [idealReps, setIdealReps] = useState(0);
  const [totalWords, setTotalWords] = useState(0);
  const [completedWords, setCompletedWords] = useState(0);
  const [seconds, setSeconds] = useState(2);
  const [remaining, setRemaining] = useState(0);
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [wordPendingCount, setWordPendingCount] = useState(0);

  const loadSets = useCallback(async () => {
    try {
      const { sets: s } = await getWordSets();
      setSets(s);
      if (s.length > 0 && !s.includes(setFile)) setSetFile(s[0]);
    } catch {
      // set listesi opsiyonel; varsayılan setFile ile devam edilir
    }
  }, [setFile]);

  const refreshWord = useCallback(async () => {
    try {
      const pendingWords = Array.from(await getPendingWords());
      const data = await getNextWord(setFile, pendingWords);
      setWord(data.word ?? null);
      setCurrentCount(data.current_count ?? 0);
      setIdealReps(data.ideal_repetitions ?? 0);
      setTotalWords((prev) => data.total_words ?? prev);
    } catch (e) {
      setStatus(`Kelime alınamadı: ${(e as Error).message}`);
    }
  }, [setFile]);

  const refreshProgress = useCallback(async () => {
    try {
      const data = await getProgress(setFile);
      setCompletedWords(data.completed_words ?? 0);
      setTotalWords((prev) => data.total_words ?? prev);
      setIdealReps((prev) => data.ideal_repetitions ?? prev);
    } catch {
      // ilerleme opsiyonel
    }
  }, [setFile]);

  const refreshPendingCounts = useCallback(async () => {
    const all = await getAllTakes();
    setPendingCount(all.length);
    if (word) {
      const forWord = await getTakesForWord(word);
      setWordPendingCount(forWord.length);
    } else {
      setWordPendingCount(0);
    }
  }, [word]);

  useEffect(() => {
    void loadSets();
  }, [loadSets]);

  useEffect(() => {
    void refreshWord();
    void refreshProgress();
  }, [setFile, refreshWord, refreshProgress]);

  useEffect(() => {
    void refreshPendingCounts();
  }, [word, refreshPendingCounts]);

  async function handleRecord() {
    if (recording || !word) return;
    setRecording(true);
    setStatus(null);
    setRemaining(seconds);

    const chunks: Int16Array[] = [];
    const handle = await startAudioCapture((chunk) => chunks.push(chunk));

    for (let i = seconds; i > 0; i--) {
      setRemaining(i);
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    handle.stop();

    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
    const merged = new Int16Array(totalLength);
    let offset = 0;
    for (const c of chunks) {
      merged.set(c, offset);
      offset += c.length;
    }

    const wavBlob = pcmToWav(merged, handle.sampleRate, 1);
    await addTake(word, wavBlob);
    await refreshPendingCounts();
    setStatus(`Cihazda kaydedildi: ${word}`);
    setRecording(false);
    setRemaining(0);
  }

  const allDone = word === null;
  const wordFull = word !== null && currentCount + wordPendingCount >= idealReps;

  return (
    <main>
      <h1>Kayıt</h1>
      <p>
        <Link to="/kayit/bekleyen">Bekleyen Kayıtlar ({pendingCount})</Link>
      </p>

      <label>
        Kelime Seti
        <select value={setFile} onChange={(e) => setSetFile(e.target.value)} disabled={recording}>
          {sets.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>

      <p>
        İlerleme: {completedWords} / {totalWords} kelime
      </p>

      {allDone ? (
        <p>Tüm kelimeler tamamlandı!</p>
      ) : (
        <>
          <h2>{word}</h2>
          <p>
            {currentCount + wordPendingCount} / {idealReps} tekrar
          </p>

          <label>
            Kayıt süresi: {seconds}sn
            <input
              type="range"
              min={1}
              max={5}
              value={seconds}
              disabled={recording}
              onChange={(e) => setSeconds(Number(e.target.value))}
            />
          </label>

          <button onClick={handleRecord} disabled={recording || wordFull}>
            {recording ? `Kaydediliyor... ${remaining}` : 'Kaydet'}
          </button>

          {wordFull && (
            <button
              onClick={() => {
                void refreshWord();
                void refreshProgress();
              }}
            >
              Sıradaki Kelime
            </button>
          )}
        </>
      )}

      {status && <p>{status}</p>}
    </main>
  );
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run build
```

Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/src/pages/KayitPage.tsx
git commit -m "Implement Kayıt page: word sets, timed recording, offline queue"
```

---

## Task 9: Pending Uploads Page

**Files:**
- Modify: `web/src/pages/PendingUploadsPage.tsx`

**Interfaces:**
- Consumes: `uploadRecording` from `../lib/api` (Task 4); `getAllTakes`, `removeTake`, `PendingTake` from `../lib/pendingRecordingsStore` (Task 3).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Implement the pending uploads page**

`web/src/pages/PendingUploadsPage.tsx`:
```tsx
import { useEffect, useState } from 'react';
import { uploadRecording } from '../lib/api';
import { getAllTakes, removeTake, type PendingTake } from '../lib/pendingRecordingsStore';

type UploadStatus = 'uploading' | 'error';

export default function PendingUploadsPage() {
  const [takes, setTakes] = useState<PendingTake[]>([]);
  const [status, setStatus] = useState<Record<number, UploadStatus>>({});
  const [errors, setErrors] = useState<Record<number, string>>({});
  const [uploadingAll, setUploadingAll] = useState(false);

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    setTakes(await getAllTakes());
  }

  async function handleUploadAll() {
    if (uploadingAll) return;
    setUploadingAll(true);
    const current = await getAllTakes();

    for (const take of current) {
      setStatus((prev) => ({ ...prev, [take.id]: 'uploading' }));
      try {
        await uploadRecording(take.word, take.audioBlob);
        await removeTake(take.id);
        setStatus((prev) => {
          const next = { ...prev };
          delete next[take.id];
          return next;
        });
        setTakes((prev) => prev.filter((t) => t.id !== take.id));
      } catch (e) {
        setStatus((prev) => ({ ...prev, [take.id]: 'error' }));
        setErrors((prev) => ({ ...prev, [take.id]: (e as Error).message }));
      }
    }

    setUploadingAll(false);
  }

  async function handleDelete(id: number) {
    await removeTake(id);
    setTakes((prev) => prev.filter((t) => t.id !== id));
  }

  const byWord = new Map<string, PendingTake[]>();
  for (const t of takes) {
    const list = byWord.get(t.word) ?? [];
    list.push(t);
    byWord.set(t.word, list);
  }

  return (
    <main>
      <h1>Bekleyen Kayıtlar</h1>
      <p>
        {takes.length} kayıt, {byWord.size} kelime bekliyor.
      </p>
      <button onClick={handleUploadAll} disabled={uploadingAll || takes.length === 0}>
        {uploadingAll ? 'Yükleniyor...' : 'Tümünü Yükle'}
      </button>

      {takes.length === 0 ? (
        <p>Bekleyen kayıt yok.</p>
      ) : (
        Array.from(byWord.entries()).map(([w, wTakes]) => (
          <section key={w}>
            <h2>{w}</h2>
            {wTakes.map((take, i) => (
              <div key={take.id}>
                <span>Tekrar {i + 1}</span>
                {status[take.id] === 'uploading' && <span>Yükleniyor...</span>}
                {status[take.id] === 'error' && <span title={errors[take.id]}>Hata</span>}
                {!status[take.id] && (
                  <>
                    <button
                      onClick={() => {
                        const audio = new Audio(URL.createObjectURL(take.audioBlob));
                        void audio.play();
                      }}
                    >
                      Oynat
                    </button>
                    <button onClick={() => handleDelete(take.id)}>Sil</button>
                  </>
                )}
              </div>
            ))}
          </section>
        ))
      )}
    </main>
  );
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run build
```

Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/src/pages/PendingUploadsPage.tsx
git commit -m "Implement pending uploads page: list, upload all, play, delete"
```

---

## Task 10: Furkanca Page (VAD Listening)

**Files:**
- Modify: `web/src/pages/FurkancaPage.tsx`

**Interfaces:**
- Consumes: `VadSegmenter`, `pcmToWav` from `../lib/vad` (Task 2); `startAudioCapture`, `AudioCaptureHandle` from `../lib/audioCapture` (Task 5); `translateSpeech`, `resolveUrl` from `../lib/api` (Task 4).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Implement the Furkanca page**

`web/src/pages/FurkancaPage.tsx`:
```tsx
import { useEffect, useRef, useState } from 'react';
import { resolveUrl, translateSpeech } from '../lib/api';
import { startAudioCapture, type AudioCaptureHandle } from '../lib/audioCapture';
import { VadSegmenter, pcmToWav } from '../lib/vad';

type ListenState = 'idle' | 'listening' | 'capturing' | 'uploading' | 'speaking' | 'error';

interface TranslationEntry {
  text: string;
  time: Date;
}

export default function FurkancaPage() {
  const [state, setState] = useState<ListenState>('idle');
  const [history, setHistory] = useState<TranslationEntry[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const captureRef = useRef<AudioCaptureHandle | null>(null);
  const segmenterRef = useRef<VadSegmenter | null>(null);

  useEffect(() => {
    let cancelled = false;

    function handleChunk(chunk: Int16Array) {
      const segmenter = segmenterRef.current;
      const capture = captureRef.current;
      if (!segmenter || !capture) return;
      const event = segmenter.addChunk(chunk);
      if (!event) return;

      if (event.type === 'started') {
        setState('capturing');
      } else if (event.type === 'discardedTooShort') {
        setState('listening');
      } else if (event.type === 'ended' && event.pcmData) {
        setState('uploading');
        void handleSegment(event.pcmData, capture.sampleRate);
      }
    }

    async function start() {
      try {
        const handle = await startAudioCapture(handleChunk);
        if (cancelled) {
          handle.stop();
          return;
        }
        captureRef.current = handle;
        segmenterRef.current = new VadSegmenter({ sampleRate: handle.sampleRate });
        setState('listening');
      } catch (e) {
        setErrorMessage(`Mikrofon erişimi reddedildi: ${(e as Error).message}`);
        setState('error');
      }
    }

    void start();
    return () => {
      cancelled = true;
      captureRef.current?.stop();
    };
  }, []);

  async function handleSegment(pcm: Int16Array, sampleRate: number) {
    try {
      const wavBlob = pcmToWav(pcm, sampleRate, 1);
      const result = await translateSpeech(wavBlob);

      if (result.rejected) {
        setState('listening');
        return;
      }

      const corrected = result.response_text?.trim() || '(boş yanıt)';
      setHistory((prev) => [{ text: corrected, time: new Date() }, ...prev]);

      if (result.audio_url) {
        setState('speaking');
        const audio = new Audio(resolveUrl(result.audio_url));
        await new Promise<void>((resolve) => {
          audio.onended = () => resolve();
          audio.onerror = () => resolve();
          void audio.play();
        });
      }
      setState('listening');
    } catch (e) {
      setErrorMessage(`Hata: ${(e as Error).message}`);
      setState('listening');
    }
  }

  return (
    <main>
      <h1>Furkanca</h1>
      <p>Durum: {state}</p>
      {errorMessage && <p role="alert">{errorMessage}</p>}
      <ul>
        {history.map((entry, i) => (
          <li key={i}>
            {entry.time.toLocaleTimeString()} — {entry.text}
          </li>
        ))}
      </ul>
    </main>
  );
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run build
```

Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/src/pages/FurkancaPage.tsx
git commit -m "Implement Furkanca page: continuous VAD listening, translate, playback"
```

---

## Task 11: Capacitor Android Integration

**Files:**
- Create: `web/capacitor.config.ts`
- Create: `web/android/` (generated by Capacitor CLI)
- Modify: `web/package.json`

**Interfaces:**
- Consumes: the built `web/dist/` output from Task 1's build script.
- Produces: an Android project buildable into `app-release.apk`, used by Task 13's deploy step.

- [ ] **Step 1: Install Capacitor and add the Android platform**

```bash
cd /home/mayasoft/app/Pronouns/web
npm install @capacitor/core @capacitor/cli @capacitor/android
```

`web/capacitor.config.ts`:
```typescript
import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.algorixstudio.furkanca',
  appName: 'Furkanca',
  webDir: 'dist',
};

export default config;
```

```bash
npm run build
npx cap add android
```

Expected: `web/android/` is created with a standard Capacitor Android project.

- [ ] **Step 2: Add the microphone permission**

Modify `web/android/app/src/main/AndroidManifest.xml` — add inside `<manifest>`, before `<application>`:

```xml
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    <uses-permission android:name="android.permission.INTERNET" />
```

No `android:usesCleartextTraffic="true"` is needed this time (unlike the Flutter app) because the production API base URL is HTTPS (`https://furkanca.algorixstudio.com`).

- [ ] **Step 3: Add build scripts**

Modify `web/package.json` — add to `"scripts"`:
```json
    "cap:sync": "vite build && npx cap sync android",
    "cap:build:apk": "vite build && npx cap sync android && cd android && ./gradlew assembleRelease"
```

- [ ] **Step 4: Build and verify the APK is produced**

```bash
cd /home/mayasoft/app/Pronouns/web
npm run cap:build:apk
ls -la android/app/build/outputs/apk/release/app-release.apk
```

Expected: `app-release.apk` exists. (It will be signed with Capacitor's default debug-equivalent signing unless a release keystore is configured — matches the existing Flutter app's current signing setup, see Global Constraints in the design spec; no change in security posture.)

- [ ] **Step 5: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/capacitor.config.ts web/android/ web/package.json
git commit -m "Add Capacitor Android wrapper for the React Furkanca app"
```

---

## Task 12: `app.furkanca.algorixstudio.com` — DNS, nginx, TLS

**Files:**
- Create: `/etc/nginx/sites-available/app-furkanca` (server config)
- Modify: `/etc/nginx/sites-enabled/` (symlink)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a live HTTPS endpoint that Task 13 deploys the web build to.

- [ ] **Step 1: Request the DNS record**

This step cannot be automated from this machine (no `gcloud` CLI is configured here, and algorixstudio.com's DNS is on Google Cloud DNS nameservers). Ask the user directly:

> "`app.furkanca.algorixstudio.com` için Google Cloud DNS'e (algorixstudio.com'un mevcut bölgesi) şu A kaydını ekler misiniz: `app.furkanca` → `88.255.236.143` (aynı IP, `furkanca.algorixstudio.com` ile aynı)? Eklendiğinde haber verin, devam edeyim."

Wait for confirmation before proceeding to Step 2.

- [ ] **Step 2: Verify DNS resolves**

```bash
dig +short app.furkanca.algorixstudio.com
```

Expected: `88.255.236.143`. If empty, DNS hasn't propagated yet — wait and retry before continuing.

- [ ] **Step 3: Create the nginx site config**

`/etc/nginx/sites-available/app-furkanca`:
```nginx
server {
    listen 80;
    listen [::]:80;
    server_name app.furkanca.algorixstudio.com;

    root /home/mayasoft/app/Pronouns/web/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    access_log /opt/algorix/logs/app-furkanca.access.log;
    error_log  /opt/algorix/logs/app-furkanca.error.log;
}
```

```bash
sudo ln -s /etc/nginx/sites-available/app-furkanca /etc/nginx/sites-enabled/app-furkanca
sudo nginx -t
sudo systemctl reload nginx
```

Expected: `nginx -t` reports syntax OK; reload succeeds.

- [ ] **Step 4: Issue the TLS certificate**

```bash
sudo certbot --nginx -d app.furkanca.algorixstudio.com --non-interactive --agree-tos -m algorixinfo@gmail.com
```

Expected: certbot obtains a certificate and rewrites `/etc/nginx/sites-enabled/app-furkanca` to add the `listen 443 ssl` block + HTTP→HTTPS redirect (same pattern as the existing `furkanca` site).

- [ ] **Step 5: Verify HTTPS serves a 404 (no build deployed yet)**

```bash
curl -sS -o /dev/null -w "HTTP %{http_code}\n" https://app.furkanca.algorixstudio.com/
```

Expected: `HTTP 404` (nginx serves from `web/dist`, which doesn't exist until Task 13 deploys a build) — this confirms TLS + nginx routing work; Task 13 fixes the 404 by deploying the actual build.

- [ ] **Step 6: Commit the nginx site source (for reference; the live file lives outside the repo)**

```bash
cd /home/mayasoft/app/Pronouns
mkdir -p ops/nginx
cp /etc/nginx/sites-available/app-furkanca ops/nginx/app-furkanca.conf
git add ops/nginx/app-furkanca.conf
git commit -m "Add app.furkanca.algorixstudio.com nginx config for the React web app"
```

---

## Task 13: Deployment — Web Build and Android APK

**Files:**
- Create: `web/deploy-web.sh`
- Create: `web/deploy-android.sh`

**Interfaces:**
- Consumes: `npm run build` (Task 1), `npm run cap:build:apk` (Task 11), the nginx site from Task 12 (which reverse-proxies `app.furkanca.algorixstudio.com` to a local `serve` process on `127.0.0.1:4003` — nginx does NOT serve `web/dist` directly via `root`, because `/home/mayasoft` is `750` and unreadable by the `www-data` nginx user; see Task 12's ledger note).
- Produces: the live production web app (via a restarted `serve` process) and the APK served from the existing `/apk` endpoint.

- [ ] **Step 1: Add a web deploy script**

`web/deploy-web.sh` — builds, then restarts the local static-file-server process nginx proxies to (kills any previous instance via a PID file, so re-running this script is idempotent):
```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"
npm run build

PIDFILE="/tmp/furkanca-web-serve.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")"
  sleep 1
fi

nohup npx serve -s dist -l 4003 > /tmp/furkanca-web-serve.log 2>&1 &
echo $! > "$PIDFILE"
sleep 2

echo "✅ Deployed: serve process running (PID $(cat "$PIDFILE")) on 127.0.0.1:4003, nginx proxies app.furkanca.algorixstudio.com to it."
echo "   Verify: curl -sS -o /dev/null -w 'HTTP %{http_code}\n' https://app.furkanca.algorixstudio.com/"
```

```bash
chmod +x /home/mayasoft/app/Pronouns/web/deploy-web.sh
```

- [ ] **Step 2: Run the web deploy and verify**

```bash
cd /home/mayasoft/app/Pronouns/web
./deploy-web.sh
curl -sS -o /dev/null -w "HTTP %{http_code}\n" https://app.furkanca.algorixstudio.com/
```

Expected: `HTTP 200`.

- [ ] **Step 3: Add an Android deploy script**

`web/deploy-android.sh`:
```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"
npm run cap:build:apk

SRC="android/app/build/outputs/apk/release/app-release.apk"
DEST="/home/mayasoft/app/Pronouns/data/apk/pronouns.apk"
BACKUP="/home/mayasoft/app/Pronouns/data/apk/pronouns_backup_$(date +%Y%m%d_%H%M%S).apk"

cp "$DEST" "$BACKUP"
cp "$SRC" "$DEST"

echo "✅ Deployed: $DEST ($(stat -c%s "$DEST") bytes)"
echo "   Previous build backed up to: $BACKUP"
```

```bash
chmod +x /home/mayasoft/app/Pronouns/web/deploy-android.sh
```

- [ ] **Step 4: Run the Android deploy and verify**

```bash
cd /home/mayasoft/app/Pronouns/web
./deploy-android.sh
curl -sS -o /dev/null -w "HTTP %{http_code} size:%{size_download}\n" https://furkanca.algorixstudio.com/apk
```

Expected: `HTTP 200`, `size` matches the new APK's byte count (not the old Flutter APK's size).

- [ ] **Step 5: Manual end-to-end verification (both platforms)**

On the actual phone and in a desktop browser, against `https://app.furkanca.algorixstudio.com` (web) and the freshly installed APK from `https://furkanca.algorixstudio.com/apk` (Android):

- Furkanca: speak a word, confirm a corrected transcription appears in history and the corrected audio plays back.
- Kayıt: select a word set, record a word, confirm it appears under "Bekleyen Kayıtlar", upload it, confirm it disappears from the pending list and progress increments.
- Ayarlar: log in with `admin`/`adminpassword`, confirm model info and settings load, change a setting and save, confirm "Ses Profilini Güncelle" works.
- Compare a freshly recorded word's resulting SNR/duration (via `data/users/FurkanV1/metadata_words.csv`, same as the SNR audit done for the Flutter app) against typical Flutter-era values — confirms the new PCM16→WAV path didn't regress audio quality.

- [ ] **Step 6: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add web/deploy-web.sh web/deploy-android.sh
git commit -m "Add web and Android deploy scripts for the React Furkanca app"
```

---

## Post-Plan: Flutter Retirement

Once Task 13's manual verification passes on both platforms (per the design spec's approved decision), stop distributing/building the Flutter app: leave `lib/`, `android/` (Flutter's, not `web/android/`), and `pubspec.yaml` in the repository untouched as an archive, but do not run `flutter build apk` again. This is a follow-up conversation, not a task in this plan — flag it to the user once Task 13 is verified working.
