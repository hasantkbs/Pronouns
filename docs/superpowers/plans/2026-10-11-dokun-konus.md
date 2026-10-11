# Dokun-Başlat / Dokun-Bitir Kayıt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Furkanca sayfasındaki ses-şiddeti tabanlı eller serbest dinlemeyi, büyük tek düğmeli dokun-başlat / dokun-bitir kayıtla değiştirmek; sunucuda konuşmacı doğrulamayı reddetmeyen (yalnızca loglayan) hale getirip CPU/GPU cihaz hatasını düzeltmek.

**Architecture:** İstemcide iki saf, test edilebilir birim: `ManualRecorder` (PCM biriktirme, süre/seviye, 0,5 sn / 30 sn sınırları) ve `flowReducer` (sayfa durum makinesi). `FurkancaPage` bunları mikrofon ve `/translate` ile bağlar; `VadSegmenter` silinir, ortak WAV yardımcıları `wav.ts`'e taşınır. Sunucuda `/translate` doğrulama skorunu ayrı bir log dosyasına yazar ve `SPEAKER_VERIFY_BLOCKING=False` iken reddetmez.

**Tech Stack:** İstemci: React 18 + TypeScript 5 + Vite 5 + vitest 2 + Capacitor 6 (Android). Sunucu: Python 3.12, FastAPI, SpeechBrain ECAPA, pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-dokun-konus-design.md`

## Global Constraints

- İki depo konumu, iki dal:
  - **Sunucu:** `/home/mayasoft/app/Pronouns`, dal `dokun-konus` (spec commit'i `42858c1b`, `master` üstünde). Python: `/home/mayasoft/app/venv/bin/python`. Suite: `/home/mayasoft/app/venv/bin/python -m pytest tests/ --ignore=tests/test_utils.py -q -p no:cacheprovider` (`tests/test_utils.py` değişikliklerden önce de toplanamıyor; kapsam dışı).
  - **İstemci:** worktree `/home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite`, dal `worktree-react-rewrite`; komutlar `web/` içinde. Test: `npm test`; tip kontrolü + derleme: `npm run build`.
- Kayıt sınırları: en az **500 ms** (`MIN_RECORDING_MS`), en çok **30 000 ms** (`MAX_RECORDING_MS`).
- Kullanıcıya görünen metinler (aynen): `Çok kısa, tekrar dene` · `Ses tanınmadı, tekrar dene` · `Mikrofon açılamadı: <hata>` · durumlar `Hazır`, `Mikrofon açılıyor...`, `Kaydediliyor...`, `İşleniyor...`, `Yanıt seslendiriliyor...` · düğme `🎤 Konuşmaya Başla` / `⏹ Bitir`.
- Düğme yalnızca `idle` ve `recording` fazlarında basılabilir.
- `config.SPEAKER_VERIFY_BLOCKING = False` (varsayılan); log satırı biçimi: `user=<id> match=<bool> score=<float|None> blocking=<bool>`; dosya `logs/speaker_verification.log` (`config.LOG_DIR`).
- `/translate` yanıt sözleşmesi ve `web/src/lib/api.ts` değişmez.
- Eller serbest mod / VAD tamamen kaldırılır (seçenek olarak da kalmaz).
- `deploy-web.sh` **kullanılmaz** (systemd `pronouns-web.service` ile port çakışması); web dağıtımı yalnızca `npm run build`.
- `pronouns-api.service` yeniden başlatma ve Furkan'ın telefonuna APK kurulumu **yalnızca kullanıcı** tarafından.
- Commit mesajları şu satırla biter: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **Bitir'e basılırken 30 sn sınırının aynı anda dolması** (çift `finishRecording`) — tek gönderim olmalı, ikinci çağrı yok sayılmalı. → `finishRecording` ref'i ilk çağrıda sıfırlar (Task 5 kodu); reducer tarafı Task 4 `test_ignores a late STOP after the recording already ended`.
2. **İşlenirken düğmeye tekrar basmak / sayfadan çıkmak** — yeni kayıt başlamamalı; sayfadan çıkınca mikrofon kapanmalı. → Task 4 `ignores START while uploading or speaking`; Task 5 unmount'ta `stopCapture`.
3. **Mikrofon izni reddi** (ilk açılış, Android izin penceresi) — sayfa kilitlenmemeli, hata görünmeli, tekrar denenebilmeli. → Task 4 `shows the microphone error and returns to idle`.
4. **Profil diskten CPU'ya yüklenip embedding GPU'da** (canlıdaki durum) — skor hesaplanmalı. → Task 1 `test_verify_scores_when_profile_and_embedding_are_on_different_devices`.
5. **`audio_url: null` ya da oynatma hatası** — sayfa `speaking`'de takılı kalmamalı. → Task 4 `skips speaking when the result has no audio`; Task 5 `audio.onerror`/`play().catch` → `AUDIO_DONE`.

---

## File Structure

| Dosya | Sorumluluk |
|---|---|
| Modify `src/services/speaker_verification_service.py` (`verify`) | Profili embedding'in cihazına taşı |
| Modify `config.py:168` | `SPEAKER_VERIFY_BLOCKING = False` |
| Modify `api.py` (globaller `:73` sonrası, `/translate` doğrulama bloğu) | Dosya logger'ı + reddetmeyen doğrulama |
| Modify `tests/test_speaker_verification_service.py`, `tests/test_translate_speaker_verification.py` | Cihaz, blocking, log testleri |
| Create `web/src/lib/wav.ts`, `web/src/lib/wav.test.ts` | `pcmToWav`, `calculateRms`, `pcmChunkDurationMs` |
| Create `web/src/lib/manualRecorder.ts` (+ test) | Kayıt biriktirme mantığı |
| Create `web/src/lib/furkancaFlow.ts` (+ test) | Sayfa durum makinesi |
| Rewrite `web/src/pages/FurkancaPage.tsx` | Düğmeli kayıt akışı |
| Modify `web/src/pages/KayitPage.tsx:5` | Import yolu |
| Delete `web/src/lib/vad.ts`, `web/src/lib/vad.test.ts` | Eller serbest mod kalkar |

---

### Task 1: Sunucu — reddetmeyen, loglanan, cihaz hatası düzeltilmiş doğrulama

**Files:**
- Modify: `src/services/speaker_verification_service.py` (`verify` içindeki `torch.dot` satırı)
- Modify: `config.py:168` (`SPEAKER_VERIFY_ENABLED = True` satırının altı)
- Modify: `api.py` (`tts = PiperTTS(...)` satırının altı; `/translate` içindeki `if config.SPEAKER_VERIFY_ENABLED:` bloğu)
- Test: `tests/test_speaker_verification_service.py`, `tests/test_translate_speaker_verification.py`

**Interfaces:**
- Consumes: mevcut `SpeakerVerificationService.verify(user_id, audio_path) -> (bool, Optional[float])`; test dosyasındaki mevcut `_FakeEncoder`, `_write_wav`, `_FakeSpeakerService(match, score)`, `_FakeTextAsr`, `_FakeCorrector`, `_FakeTts`, `_post_translate`.
- Produces: `config.SPEAKER_VERIFY_BLOCKING: bool = False`; `api._speaker_logger` (`logging.getLogger("speaker_verification")`, INFO, dosya handler'ı). İstemci task'ları bunlara bağlı değil.

- [ ] **Step 1: Cihaz testini yaz**

`tests/test_speaker_verification_service.py` sonuna:

```python
class _CudaFakeEncoder(_FakeEncoder):
    def encode_batch(self, signal):
        return super().encode_batch(signal).to("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")
def test_verify_scores_when_profile_and_embedding_are_on_different_devices(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_PATH", str(tmp_path))
    monkeypatch.setattr(config, "SPEAKER_VERIFY_THRESHOLD", 0.99)
    monkeypatch.setattr(
        SpeakerVerificationService, "_get_encoder",
        classmethod(lambda cls: _CudaFakeEncoder()),
    )
    words_dir = tmp_path / "Furkan" / "words" / "merhaba"
    words_dir.mkdir(parents=True)
    for i in range(3):
        _write_wav(words_dir / f"rep{i + 1}.wav", freq=220.0)
    SpeakerVerificationService().enroll("Furkan")

    probe = tmp_path / "probe.wav"
    _write_wav(probe, freq=220.0)
    # Canlidaki gibi: yeni servis profili diskten CPU'ya yukler, embedding GPU'da
    is_match, score = SpeakerVerificationService().verify("Furkan", str(probe))
    assert score is not None
    assert is_match is True
```

- [ ] **Step 2: Blocking ve log testlerini yaz**

`tests/test_translate_speaker_verification.py` başındaki importlara `import logging` ekle (`import io` satırının üstüne).

Mevcut `test_translate_rejects_when_speaker_does_not_match` fonksiyonunun ilk satırının **önüne** ekle (ret davranışı artık yalnızca `BLOCKING=True` iken):

```python
    monkeypatch.setattr(config, "SPEAKER_VERIFY_BLOCKING", True)
```

Dosyanın sonuna:

```python
def test_speaker_verification_is_not_blocking_by_default():
    assert config.SPEAKER_VERIFY_BLOCKING is False


def _patch_non_blocking_mismatch(monkeypatch):
    monkeypatch.setattr(config, "SPEAKER_VERIFY_ENABLED", True)
    monkeypatch.setattr(config, "SPEAKER_VERIFY_BLOCKING", False)
    monkeypatch.setattr(api, "speaker_service", _FakeSpeakerService(match=False, score=0.1))
    monkeypatch.setattr(api, "get_asr", lambda user_id: _FakeTextAsr())
    monkeypatch.setattr(api, "get_corrector", lambda user_id, vocabulary: _FakeCorrector())
    monkeypatch.setattr(api, "tts", _FakeTts())


def test_translate_does_not_reject_when_verification_is_not_blocking(monkeypatch):
    _patch_non_blocking_mismatch(monkeypatch)

    body = _post_translate().json()

    assert "rejected" not in body
    assert body["response_text"] == "Ben okula geldim."


def test_translate_logs_speaker_decision(monkeypatch, caplog):
    _patch_non_blocking_mismatch(monkeypatch)

    with caplog.at_level(logging.INFO, logger="speaker_verification"):
        _post_translate()

    assert "user=FurkanV1 match=False score=0.1 blocking=False" in caplog.text
```

- [ ] **Step 3: Testleri çalıştır, başarısız olduklarını gör**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_speaker_verification_service.py tests/test_translate_speaker_verification.py -q -p no:cacheprovider`
Expected: 5 FAIL — cihaz testi `assert None is not None` (plan yazılırken doğrulandı); `test_translate_rejects_when_speaker_does_not_match`, `..._not_blocking_by_default`, `..._does_not_reject_...`, `..._logs_speaker_decision` → `AttributeError: <module 'config' ...> has no attribute 'SPEAKER_VERIFY_BLOCKING'`. Diğerleri PASS.

- [ ] **Step 4: Cihaz hatasını düzelt**

`src/services/speaker_verification_service.py` `verify` içinde:

```python
            score = float(torch.dot(embedding, profile))
```
→
```python
            # Profil diskten CPU'ya yuklenir, embedding GPU'da olabilir
            score = float(torch.dot(embedding, profile.to(embedding.device)))
```

- [ ] **Step 5: Config bayrağını ekle**

`config.py`'de `SPEAKER_VERIFY_ENABLED = True` satırının altına:

```python
# Dokun-konus akisinda dugmeye Furkan basar; dogrulama karar vermez, sadece loglar.
# True yapilirsa eslesmeyen ses eskisi gibi reddedilir.
SPEAKER_VERIFY_BLOCKING = False
```

- [ ] **Step 6: Logger'ı ve reddetmeyen doğrulamayı ekle**

`api.py`'de `tts = PiperTTS(Path(config.PIPER_VOICE_PATH))  # CPU, API acilisinda bir kez` satırının altına:

```python
_speaker_logger = logging.getLogger("speaker_verification")
if not _speaker_logger.handlers:
    _speaker_logger.setLevel(logging.INFO)
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    _sv_handler = logging.FileHandler(str(Path(config.LOG_DIR) / "speaker_verification.log"), encoding="utf-8")
    _sv_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    _speaker_logger.addHandler(_sv_handler)
```

`/translate` içindeki:

```python
    if config.SPEAKER_VERIFY_ENABLED:
        is_match, score = speaker_service.verify(user_id, temp_input)
        logging.getLogger("speaker_verification").info(
            f"user={user_id} match={is_match} score={score}"
        )
        if not is_match:
            os.remove(temp_input)
            return {"rejected": True, "reason": "speaker_mismatch"}
```
→
```python
    if config.SPEAKER_VERIFY_ENABLED:
        is_match, score = speaker_service.verify(user_id, temp_input)
        blocking = config.SPEAKER_VERIFY_BLOCKING
        _speaker_logger.info(f"user={user_id} match={is_match} score={score} blocking={blocking}")
        if blocking and not is_match:
            os.remove(temp_input)
            return {"rejected": True, "reason": "speaker_mismatch"}
```

- [ ] **Step 7: Testleri çalıştır, geçtiğini gör**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_speaker_verification_service.py tests/test_translate_speaker_verification.py -q -p no:cacheprovider`
Expected: hepsi PASS (`8` + `10` = `18 passed`).

Run (suite): `/home/mayasoft/app/venv/bin/python -m pytest tests/ --ignore=tests/test_utils.py -q -p no:cacheprovider`
Expected: `82 passed` (önceki 78 + 4 yeni).

- [ ] **Step 8: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add src/services/speaker_verification_service.py config.py api.py \
  tests/test_speaker_verification_service.py tests/test_translate_speaker_verification.py
git commit -m "fix: make speaker verification log-only and fix its CPU/GPU mismatch

The stored profile is loaded on CPU while ECAPA embeddings come back on
CUDA, so torch.dot raised on every request and verification silently
failed open; nothing was logged because the logger had no handler.
Move the profile to the embedding's device, write every decision to
logs/speaker_verification.log, and stop rejecting by default
(config.SPEAKER_VERIFY_BLOCKING=False): with tap-to-record the user's
own press is the intent, and rejecting mixed audio lost their sentence.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: İstemci — `wav.ts` (ortak WAV yardımcıları)

**Files:**
- Create: `web/src/lib/wav.ts`, `web/src/lib/wav.test.ts`
- Modify: `web/src/lib/vad.ts` (taşınan fonksiyonları sil, `./wav`'dan import et), `web/src/lib/vad.test.ts` (taşınan testleri sil), `web/src/pages/KayitPage.tsx:5`

**Interfaces:**
- Consumes: yok.
- Produces: `web/src/lib/wav.ts` → `calculateRms(samples: Int16Array): number`, `pcmChunkDurationMs(samples: Int16Array, sampleRate: number): number`, `pcmToWav(pcm: Int16Array, sampleRate: number, numChannels = 1): Blob` (davranış `vad.ts`'tekiyle birebir). Task 3 ve 5 kullanır.

- [ ] **Step 1: Failing testi yaz**

`web/src/lib/wav.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { calculateRms, pcmChunkDurationMs, pcmToWav } from './wav';

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

describe('pcmChunkDurationMs', () => {
  it('converts a sample count to milliseconds', () => {
    expect(pcmChunkDurationMs(new Int16Array(1600), 16000)).toBe(100);
  });
});
```

- [ ] **Step 2: Testi çalıştır, başarısız olduğunu gör**

Run: `cd /home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite/web && npx vitest run src/lib/wav.test.ts`
Expected: FAIL — `Failed to resolve import "./wav"`.

- [ ] **Step 3: `wav.ts`'i oluştur**

`web/src/lib/wav.ts` (içerik `vad.ts`'in ilk üç fonksiyonuyla birebir aynı):

```ts
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
```

`web/src/lib/vad.ts`'ten `calculateRms`, `pcmChunkDurationMs`, `pcmToWav` fonksiyonlarını sil ve dosyanın en üstüne ekle:

```ts
import { calculateRms, pcmChunkDurationMs } from './wav';
```

`web/src/lib/vad.test.ts`'te `describe('calculateRms', ...)` ve `describe('pcmToWav', ...)` bloklarını sil; import satırını şu yap:

```ts
import { VadSegmenter } from './vad';
```

(`silentChunk`/`loudChunk` yardımcıları `VadSegmenter` testlerinde kullanıldığı için kalır.)

`web/src/pages/KayitPage.tsx:5`:

```ts
import { pcmToWav } from '../lib/vad';
```
→
```ts
import { pcmToWav } from '../lib/wav';
```

- [ ] **Step 4: Testleri ve derlemeyi çalıştır**

Run: `npm test && npm run build`
Expected: `Tests  11 passed (11)` (wav 4 + vad 3 + pendingRecordingsStore 4); build `✓ built`.

- [ ] **Step 5: Commit**

```bash
cd /home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite
git add web/src/lib/wav.ts web/src/lib/wav.test.ts web/src/lib/vad.ts web/src/lib/vad.test.ts web/src/pages/KayitPage.tsx
git commit -m "refactor: move shared WAV/RMS helpers from vad.ts to wav.ts

Prepares removing the VAD: KayitPage and the upcoming tap-to-record
flow need pcmToWav/calculateRms, not VadSegmenter.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: İstemci — `ManualRecorder`

**Files:**
- Create: `web/src/lib/manualRecorder.ts`, `web/src/lib/manualRecorder.test.ts`

**Interfaces:**
- Consumes: Task 2 `calculateRms` (`./wav`).
- Produces: `MIN_RECORDING_MS = 500`, `MAX_RECORDING_MS = 30000`; `type FinishResult = { tooShort: true } | { tooShort: false; pcm: Int16Array }`; `class ManualRecorder(sampleRate: number, minDurationMs = MIN_RECORDING_MS, maxDurationMs = MAX_RECORDING_MS)` → `push(chunk: Int16Array): boolean` (azami süreye ulaşıldıysa `true`), `get durationMs(): number`, `get level(): number`, `finish(): FinishResult`. Task 5 kullanır.

- [ ] **Step 1: Failing testi yaz**

`web/src/lib/manualRecorder.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { ManualRecorder, MAX_RECORDING_MS, MIN_RECORDING_MS } from './manualRecorder';

const RATE = 16000;

function chunk(ms: number, amplitude = 0): Int16Array {
  return new Int16Array((RATE * ms) / 1000).fill(amplitude);
}

describe('ManualRecorder', () => {
  it('uses 0.5 s and 30 s limits by default', () => {
    expect(MIN_RECORDING_MS).toBe(500);
    expect(MAX_RECORDING_MS).toBe(30000);
  });

  it('accumulates chunks in order into one PCM buffer', () => {
    const recorder = new ManualRecorder(RATE);
    recorder.push(chunk(300, 1));
    recorder.push(chunk(300, 2));
    const result = recorder.finish();
    expect(result.tooShort).toBe(false);
    if (result.tooShort) return;
    expect(result.pcm.length).toBe(RATE * 0.6);
    expect(result.pcm[0]).toBe(1);
    expect(result.pcm[result.pcm.length - 1]).toBe(2);
  });

  it('tracks duration in milliseconds', () => {
    const recorder = new ManualRecorder(RATE);
    recorder.push(chunk(250));
    recorder.push(chunk(250));
    expect(recorder.durationMs).toBe(500);
  });

  it('reports the level of the latest chunk', () => {
    const recorder = new ManualRecorder(RATE);
    recorder.push(chunk(100, 10000));
    expect(recorder.level).toBeGreaterThan(0.3);
    recorder.push(chunk(100, 0));
    expect(recorder.level).toBe(0);
  });

  it('marks recordings shorter than the minimum as too short', () => {
    const recorder = new ManualRecorder(RATE);
    recorder.push(chunk(400));
    expect(recorder.finish()).toEqual({ tooShort: true });
  });

  it('treats a recording with no chunks as too short', () => {
    expect(new ManualRecorder(RATE).finish()).toEqual({ tooShort: true });
  });

  it('signals auto-stop once the maximum duration is reached', () => {
    const recorder = new ManualRecorder(RATE, 500, 1000);
    expect(recorder.push(chunk(600))).toBe(false);
    expect(recorder.push(chunk(400))).toBe(true);
  });
});
```

- [ ] **Step 2: Testi çalıştır, başarısız olduğunu gör**

Run: `npx vitest run src/lib/manualRecorder.test.ts`
Expected: FAIL — `Failed to resolve import "./manualRecorder"`.

- [ ] **Step 3: `ManualRecorder`'ı yaz**

`web/src/lib/manualRecorder.ts`:

```ts
import { calculateRms } from './wav';

export const MIN_RECORDING_MS = 500;
export const MAX_RECORDING_MS = 30000;

export type FinishResult = { tooShort: true } | { tooShort: false; pcm: Int16Array };

/**
 * Dokun-baslat / dokun-bitir kaydinin saf mantigi: PCM chunk'larini biriktirir,
 * sure ve ses seviyesini hesaplar. Mikrofona dokunmaz (test edilebilir).
 */
export class ManualRecorder {
  private chunks: Int16Array[] = [];
  private totalSamples = 0;
  private lastLevel = 0;

  constructor(
    private readonly sampleRate: number,
    private readonly minDurationMs = MIN_RECORDING_MS,
    private readonly maxDurationMs = MAX_RECORDING_MS,
  ) {}

  /** Chunk ekler; azami sureye ulasildiysa true doner (kayit otomatik bitmeli). */
  push(chunk: Int16Array): boolean {
    this.chunks.push(chunk);
    this.totalSamples += chunk.length;
    this.lastLevel = calculateRms(chunk);
    return this.durationMs >= this.maxDurationMs;
  }

  get durationMs(): number {
    return (this.totalSamples / this.sampleRate) * 1000;
  }

  /** Son chunk'in RMS seviyesi (0..1), ses seviyesi cubugu icin. */
  get level(): number {
    return this.lastLevel;
  }

  finish(): FinishResult {
    if (this.durationMs < this.minDurationMs) return { tooShort: true };
    const pcm = new Int16Array(this.totalSamples);
    let offset = 0;
    for (const chunk of this.chunks) {
      pcm.set(chunk, offset);
      offset += chunk.length;
    }
    return { tooShort: false, pcm };
  }
}
```

- [ ] **Step 4: Testleri ve derlemeyi çalıştır**

Run: `npm test && npm run build`
Expected: `Tests  18 passed (18)`; build `✓ built`.

- [ ] **Step 5: Commit**

```bash
cd /home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite
git add web/src/lib/manualRecorder.ts web/src/lib/manualRecorder.test.ts
git commit -m "feat: add ManualRecorder for tap-to-record capture

Accumulates PCM chunks, reports duration and level, rejects recordings
under 0.5 s and signals auto-stop at 30 s. Pure logic, no microphone.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: İstemci — `furkancaFlow` durum makinesi

**Files:**
- Create: `web/src/lib/furkancaFlow.ts`, `web/src/lib/furkancaFlow.test.ts`

**Interfaces:**
- Consumes: yok.
- Produces: `type Phase = 'idle' | 'starting' | 'recording' | 'uploading' | 'speaking'`; `interface HistoryEntry { text: string; time: Date }`; `interface FlowState { phase: Phase; error: string | null; history: HistoryEntry[] }`; `type FlowEvent` (`START`, `MIC_READY`, `MIC_FAILED{message}`, `TOO_SHORT`, `STOP`, `RESULT{text, hasAudio, time}`, `REJECTED`, `FAILED{message}`, `AUDIO_DONE`); `initialFlowState`; `flowReducer(state, event): FlowState` (geçersiz fazdaki olay aynı state nesnesini döner); `buttonEnabled(phase): boolean`; `formatElapsed(ms): string` (`m:ss`); `TOO_SHORT_MESSAGE`, `REJECTED_MESSAGE`. Task 5 kullanır.

- [ ] **Step 1: Failing testi yaz**

`web/src/lib/furkancaFlow.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import {
  buttonEnabled,
  flowReducer,
  formatElapsed,
  initialFlowState,
  REJECTED_MESSAGE,
  TOO_SHORT_MESSAGE,
  type FlowEvent,
  type FlowState,
} from './furkancaFlow';

const T = new Date('2026-10-11T12:00:00Z');

function run(...events: FlowEvent[]): FlowState {
  return events.reduce(flowReducer, initialFlowState);
}

describe('flowReducer', () => {
  it('walks the happy path and returns to idle after the audio', () => {
    const states = [
      { type: 'START' },
      { type: 'MIC_READY' },
      { type: 'STOP' },
      { type: 'RESULT', text: 'Ben geldim.', hasAudio: true, time: T },
      { type: 'AUDIO_DONE' },
    ].reduce<FlowState[]>(
      (acc, event) => [...acc, flowReducer(acc[acc.length - 1], event as FlowEvent)],
      [initialFlowState],
    );
    expect(states.map((s) => s.phase)).toEqual(['idle', 'starting', 'recording', 'uploading', 'speaking', 'idle']);
    expect(states[states.length - 1].history).toEqual([{ text: 'Ben geldim.', time: T }]);
  });

  it('skips speaking when the result has no audio', () => {
    const state = run(
      { type: 'START' },
      { type: 'MIC_READY' },
      { type: 'STOP' },
      { type: 'RESULT', text: 'Ben geldim.', hasAudio: false, time: T },
    );
    expect(state.phase).toBe('idle');
    expect(state.history).toHaveLength(1);
  });

  it('puts the newest result first', () => {
    const first = run(
      { type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' },
      { type: 'RESULT', text: 'Bir.', hasAudio: false, time: T },
    );
    const second = [
      { type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' },
      { type: 'RESULT', text: 'İki.', hasAudio: false, time: T },
    ].reduce((s, e) => flowReducer(s, e as FlowEvent), first);
    expect(second.history.map((h) => h.text)).toEqual(['İki.', 'Bir.']);
  });

  it('shows a too-short message and returns to idle', () => {
    const state = run({ type: 'START' }, { type: 'MIC_READY' }, { type: 'TOO_SHORT' });
    expect(state).toMatchObject({ phase: 'idle', error: TOO_SHORT_MESSAGE });
  });

  it('shows the microphone error and returns to idle', () => {
    const state = run({ type: 'START' }, { type: 'MIC_FAILED', message: 'Permission denied' });
    expect(state).toMatchObject({ phase: 'idle', error: 'Mikrofon açılamadı: Permission denied' });
  });

  it('shows server errors instead of silently going back', () => {
    const state = run(
      { type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' },
      { type: 'FAILED', message: 'Konuşma anlaşılamadı.' },
    );
    expect(state).toMatchObject({ phase: 'idle', error: 'Konuşma anlaşılamadı.' });
  });

  it('shows a message when the server rejects the speaker', () => {
    const state = run({ type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' }, { type: 'REJECTED' });
    expect(state).toMatchObject({ phase: 'idle', error: REJECTED_MESSAGE });
  });

  it('clears the previous error on the next start', () => {
    const failed = run({ type: 'START' }, { type: 'MIC_READY' }, { type: 'TOO_SHORT' });
    expect(flowReducer(failed, { type: 'START' }).error).toBeNull();
  });

  it('ignores START while uploading or speaking', () => {
    const uploading = run({ type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' });
    expect(flowReducer(uploading, { type: 'START' })).toBe(uploading);
    const speaking = flowReducer(uploading, { type: 'RESULT', text: 'X.', hasAudio: true, time: T });
    expect(flowReducer(speaking, { type: 'START' })).toBe(speaking);
  });

  it('ignores a late STOP after the recording already ended', () => {
    const uploading = run({ type: 'START' }, { type: 'MIC_READY' }, { type: 'STOP' });
    expect(flowReducer(uploading, { type: 'STOP' })).toBe(uploading);
  });
});

describe('buttonEnabled', () => {
  it('is enabled only while idle or recording', () => {
    expect(buttonEnabled('idle')).toBe(true);
    expect(buttonEnabled('recording')).toBe(true);
    expect(buttonEnabled('starting')).toBe(false);
    expect(buttonEnabled('uploading')).toBe(false);
    expect(buttonEnabled('speaking')).toBe(false);
  });
});

describe('formatElapsed', () => {
  it('formats milliseconds as m:ss', () => {
    expect(formatElapsed(0)).toBe('0:00');
    expect(formatElapsed(3400)).toBe('0:03');
    expect(formatElapsed(65000)).toBe('1:05');
  });
});
```

- [ ] **Step 2: Testi çalıştır, başarısız olduğunu gör**

Run: `npx vitest run src/lib/furkancaFlow.test.ts`
Expected: FAIL — `Failed to resolve import "./furkancaFlow"`.

- [ ] **Step 3: `furkancaFlow.ts`'i yaz**

`web/src/lib/furkancaFlow.ts`:

```ts
export type Phase = 'idle' | 'starting' | 'recording' | 'uploading' | 'speaking';

export interface HistoryEntry {
  text: string;
  time: Date;
}

export interface FlowState {
  phase: Phase;
  error: string | null;
  history: HistoryEntry[];
}

export type FlowEvent =
  | { type: 'START' }
  | { type: 'MIC_READY' }
  | { type: 'MIC_FAILED'; message: string }
  | { type: 'TOO_SHORT' }
  | { type: 'STOP' }
  | { type: 'RESULT'; text: string; hasAudio: boolean; time: Date }
  | { type: 'REJECTED' }
  | { type: 'FAILED'; message: string }
  | { type: 'AUDIO_DONE' };

export const TOO_SHORT_MESSAGE = 'Çok kısa, tekrar dene';
export const REJECTED_MESSAGE = 'Ses tanınmadı, tekrar dene';

export const initialFlowState: FlowState = { phase: 'idle', error: null, history: [] };

/** Dugme yalnizca bostayken (baslat) ve kayittayken (bitir) basilabilir. */
export function buttonEnabled(phase: Phase): boolean {
  return phase === 'idle' || phase === 'recording';
}

export function formatElapsed(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

/**
 * Furkanca sayfasinin durum makinesi. Gecersiz durumda gelen olay yok sayilir
 * (ornegin islenirken ikinci bir START yeni kayit baslatmaz).
 */
export function flowReducer(state: FlowState, event: FlowEvent): FlowState {
  switch (event.type) {
    case 'START':
      return state.phase === 'idle' ? { ...state, phase: 'starting', error: null } : state;
    case 'MIC_READY':
      return state.phase === 'starting' ? { ...state, phase: 'recording' } : state;
    case 'MIC_FAILED':
      return state.phase === 'starting'
        ? { ...state, phase: 'idle', error: `Mikrofon açılamadı: ${event.message}` }
        : state;
    case 'TOO_SHORT':
      return state.phase === 'recording' ? { ...state, phase: 'idle', error: TOO_SHORT_MESSAGE } : state;
    case 'STOP':
      return state.phase === 'recording' ? { ...state, phase: 'uploading' } : state;
    case 'RESULT':
      return state.phase === 'uploading'
        ? {
            ...state,
            phase: event.hasAudio ? 'speaking' : 'idle',
            history: [{ text: event.text, time: event.time }, ...state.history],
          }
        : state;
    case 'REJECTED':
      return state.phase === 'uploading' ? { ...state, phase: 'idle', error: REJECTED_MESSAGE } : state;
    case 'FAILED':
      return state.phase === 'uploading' ? { ...state, phase: 'idle', error: event.message } : state;
    case 'AUDIO_DONE':
      return state.phase === 'speaking' ? { ...state, phase: 'idle' } : state;
  }
}
```

- [ ] **Step 4: Testleri ve derlemeyi çalıştır**

Run: `npm test && npm run build`
Expected: `Tests  30 passed (30)`; build `✓ built`.

- [ ] **Step 5: Commit**

```bash
cd /home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite
git add web/src/lib/furkancaFlow.ts web/src/lib/furkancaFlow.test.ts
git commit -m "feat: add furkancaFlow state machine for tap-to-record

idle -> starting -> recording -> uploading -> speaking -> idle. Every
failure (mic, too short, server error, speaker rejection) returns to
idle with a visible message instead of silently resuming; events in the
wrong phase are ignored, so a second press while processing does nothing.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: İstemci — `FurkancaPage` yeniden yazımı, VAD'ın silinmesi

**Files:**
- Rewrite: `web/src/pages/FurkancaPage.tsx`
- Delete: `web/src/lib/vad.ts`, `web/src/lib/vad.test.ts`

**Interfaces:**
- Consumes: Task 2 `pcmToWav`; Task 3 `ManualRecorder`; Task 4 `flowReducer`, `initialFlowState`, `buttonEnabled`, `formatElapsed`, `Phase`; mevcut `startAudioCapture(onChunk) -> Promise<AudioCaptureHandle{sampleRate, stop}>`, `translateSpeech(blob) -> Promise<TranslateResponse>` (`error`'u fırlatır), `resolveUrl(path)`.
- Produces: yeni Furkanca sayfası (default export `FurkancaPage`, route değişmez).

Bu task'ta bileşen testi yok (proje vitest `environment: 'node'`, DOM test altyapısı yok); davranış Task 3–4'ün birim testleriyle ve Task 6'nın elle testiyle doğrulanır. Gate: `tsc` + suite + build.

- [ ] **Step 1: Sayfayı yeniden yaz**

`web/src/pages/FurkancaPage.tsx` dosyasının tamamını şununla değiştir:

```tsx
import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { resolveUrl, translateSpeech } from '../lib/api';
import { startAudioCapture, type AudioCaptureHandle } from '../lib/audioCapture';
import { buttonEnabled, flowReducer, formatElapsed, initialFlowState, type Phase } from '../lib/furkancaFlow';
import { ManualRecorder } from '../lib/manualRecorder';
import { pcmToWav } from '../lib/wav';

const STATUS_TEXT: Record<Phase, string> = {
  idle: 'Hazır',
  starting: 'Mikrofon açılıyor...',
  recording: 'Kaydediliyor...',
  uploading: 'İşleniyor...',
  speaking: 'Yanıt seslendiriliyor...',
};

export default function FurkancaPage() {
  const [state, dispatch] = useReducer(flowReducer, initialFlowState);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [level, setLevel] = useState(0);

  const captureRef = useRef<AudioCaptureHandle | null>(null);
  const recorderRef = useRef<ManualRecorder | null>(null);

  const stopCapture = useCallback(() => {
    captureRef.current?.stop();
    captureRef.current = null;
  }, []);

  // Sayfadan cikilinca mikrofon kapanir
  useEffect(() => stopCapture, [stopCapture]);

  const send = useCallback(async (pcm: Int16Array, sampleRate: number) => {
    try {
      const result = await translateSpeech(pcmToWav(pcm, sampleRate, 1));
      if (result.rejected) {
        dispatch({ type: 'REJECTED' });
        return;
      }
      const text = result.response_text?.trim() || '(boş yanıt)';
      dispatch({ type: 'RESULT', text, hasAudio: Boolean(result.audio_url), time: new Date() });
      if (result.audio_url) {
        const audio = new Audio(resolveUrl(result.audio_url));
        await new Promise<void>((resolve) => {
          audio.onended = () => resolve();
          audio.onerror = () => resolve();
          audio.play().catch(() => resolve());
        });
        dispatch({ type: 'AUDIO_DONE' });
      }
    } catch (e) {
      dispatch({ type: 'FAILED', message: (e as Error).message });
    }
  }, []);

  const finishRecording = useCallback(() => {
    const recorder = recorderRef.current;
    const capture = captureRef.current;
    // Ikinci cagri (dugme + 30 sn siniri ayni anda) yok sayilir
    if (!recorder || !capture) return;
    recorderRef.current = null;
    const sampleRate = capture.sampleRate;
    stopCapture();

    const result = recorder.finish();
    if (result.tooShort) {
      dispatch({ type: 'TOO_SHORT' });
      return;
    }
    dispatch({ type: 'STOP' });
    void send(result.pcm, sampleRate);
  }, [send, stopCapture]);

  const startRecording = useCallback(async () => {
    dispatch({ type: 'START' });
    setElapsedMs(0);
    setLevel(0);
    try {
      const handle = await startAudioCapture((chunk) => {
        const recorder = recorderRef.current;
        if (!recorder) return;
        const reachedMax = recorder.push(chunk);
        setElapsedMs(recorder.durationMs);
        setLevel(recorder.level);
        if (reachedMax) finishRecording();
      });
      captureRef.current = handle;
      recorderRef.current = new ManualRecorder(handle.sampleRate);
      // Dugme ancak mikrofon gercekten acildiktan sonra "Bitir" olur: ilk hece kacmaz
      dispatch({ type: 'MIC_READY' });
    } catch (e) {
      dispatch({ type: 'MIC_FAILED', message: (e as Error).message });
    }
  }, [finishRecording]);

  const recording = state.phase === 'recording';

  return (
    <main>
      <h1>Furkanca</h1>
      <button
        type="button"
        disabled={!buttonEnabled(state.phase)}
        onClick={() => (recording ? finishRecording() : void startRecording())}
        style={{ width: '100%', minHeight: '30vh', fontSize: '2rem' }}
      >
        {recording ? '⏹ Bitir' : '🎤 Konuşmaya Başla'}
      </button>
      <p>Durum: {STATUS_TEXT[state.phase]}</p>
      {recording && (
        <p>
          {formatElapsed(elapsedMs)}{' '}
          <meter min={0} max={0.3} value={Math.min(level, 0.3)} aria-label="Ses seviyesi" />
        </p>
      )}
      {state.error && <p role="alert">{state.error}</p>}
      <ul>
        {state.history.map((entry, i) => (
          <li key={i}>
            {entry.time.toLocaleTimeString()} — {entry.text}
          </li>
        ))}
      </ul>
    </main>
  );
}
```

- [ ] **Step 2: VAD'ı sil ve kalan referans olmadığını doğrula**

```bash
cd /home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite/web
git rm -q src/lib/vad.ts src/lib/vad.test.ts
grep -rn "lib/vad\|VadSegmenter" src || echo "vad referansı yok"
```

Expected: `vad referansı yok`.

- [ ] **Step 3: Tip kontrolü, testler, derleme**

Run: `npm test && npm run build`
Expected: `Test Files  4 passed (4)`, `Tests  27 passed (27)` (vad'ın 3 testi gitti); build `✓ built` (bu dosyalarla plan yazılırken izole kopyada doğrulandı: tsc temiz, 27/27, build başarılı).

- [ ] **Step 4: Commit**

```bash
cd /home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite
git add web/src/pages/FurkancaPage.tsx
git commit -m "feat: replace hands-free VAD listening with tap-to-record

One large button starts and stops the recording; the microphone is
open only while recording and the button turns into 'Bitir' only once
the mic is actually live. Recordings under 0.5 s are not sent, 30 s
auto-stops, and every error is shown. Removes VadSegmenter: RMS-only
VAD started recordings on any noise and its 1 s silence cut split
sentences.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Dağıtım ve elle doğrulama (insan onayı gerekli)

**Files:** yok.

**Interfaces:**
- Consumes: Task 1–5.
- Produces: canlıda sunucu + web + APK.

- [ ] **Step 1: Sunucuyu yayına hazırla ve DUR**

Sunucu değişiklikleri `dokun-konus` dalında; canlı API bu dizinden (`/home/mayasoft/app/Pronouns`) çalışıyor ve dal zaten checkout'ta. Kullanıcıdan iste ve bekle:

```
! sudo systemctl restart pronouns-api.service
```

(`master`'a merge/push kararı kullanıcınındır; ayrıca sor.)

- [ ] **Step 2: Sunucuyu doğrula**

```bash
cd /home/mayasoft/app/Pronouns
systemctl show pronouns-api.service -p ActiveEnterTimestamp
KEY=$(/home/mayasoft/app/venv/bin/python -c "import config; print(config.API_SECRET_KEY)" 2>/dev/null)
F=$(/home/mayasoft/app/venv/bin/python -c "import pandas as pd; print(pd.read_csv('data/users/FurkanV1/eval.csv').file_path.iloc[0])")
curl -sS -H "X-API-Key: $KEY" -F user_id=FurkanV1 -F "audio=@$F" http://127.0.0.1:8001/translate; echo
tail -2 logs/speaker_verification.log
```

Expected: yeniden başlatma zamanı Task 1 commit'inden sonra; yanıtta `"intent":"translate"`; log satırı `user=FurkanV1 match=True score=0.<sayı> blocking=False` (skor `None` **değil** — cihaz hatası gitti).

- [ ] **Step 3: Web'i derle**

```bash
cd /home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite/web
npm run build
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' https://app.furkanca.algorixstudio.com/
curl -sS https://app.furkanca.algorixstudio.com/ | grep -o 'assets/index-[^"]*\.js'
ls dist/assets/
```

Expected: `HTTP 200`; sitenin sunduğu `index-*.js` adı `dist/assets/` içindeki yeni dosyayla aynı (`pronouns-web.service` diskten okur, yeniden başlatma yok).

- [ ] **Step 4: APK'yı derle ve dağıt**

```bash
cd /home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite/web
bash deploy-android.sh
```

Expected: `✅ Deployed: /home/mayasoft/app/Pronouns/data/apk/pronouns.apk (...)` ve yedek yolu. Derleme hatası olursa DUR ve bildir (imzalama/SDK).

- [ ] **Step 5: Elle test (kullanıcı)**

Kullanıcıdan önce web'de, sonra Furkan'ın telefonuna `/apk`'dan **yeni APK'yı kurup** şunları denemesini iste ve sonuçları ledger'a yaz:
1. Normal cümle: dokun → konuş → Bitir → düzeltilmiş cümle + ses.
2. Kayıt sırasında arkadan başka bir ses (TV/konuşma): cümle kaybolmamalı.
3. Cümle ortasında 2–3 sn duraklama: cümle bölünmemeli.
4. Hemen dokun-bitir (< 0,5 sn): "Çok kısa, tekrar dene".
5. 30 sn boyunca Bitir'e basmama: otomatik gönderim.
6. İşleniyorken düğme kilitli.
7. Mikrofon izni reddi (yalnızca ilk kurulumda denenebilir): hata mesajı, tekrar deneme mümkün.
