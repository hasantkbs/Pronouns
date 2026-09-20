# Whisper+LoRA Personalized Training — Design Spec

**Date:** 2026-09-20
**Status:** Approved for planning

## 1. Problem and Goal

FurkanV1's personalized ASR model (Wav2Vec2-XLS-R-300M CTC + LoRA) plateaus around
64-95% WER depending on random seed, with a historical best of 64.99% (2026-08-11,
an unseeded lucky run). This session already fixed two defects in that pipeline
(commit `7160694c`: an `num_proc` bug that silently failed or OOM-killed
production training, and missing RNG seeding), but the underlying CTC/Wav2Vec2
architecture itself is not expected to meaningfully improve beyond this plateau
for atypical/disordered speech.

**Goal:** add a second, independent personalization path based on Whisper
(encoder-decoder, sequence-to-sequence) fine-tuned with LoRA, starting from a
Turkish-tuned Whisper checkpoint, without disturbing the live Wav2Vec2 path
that real patient-facing traffic depends on today. Validate the new model's WER
against the existing model before it ever serves live traffic. Also clean up
config.py's redundant/duplicate constants discovered during this work.

## 2. Background: What Already Exists

- `src/core/asr.py` (live inference) **already supports Whisper**: it detects
  architecture from the personalized adapter's `adapter_config.json`
  (`base_model_name_or_path` containing `"whisper"`), loads `WhisperProcessor`/
  `WhisperForConditionalGeneration` accordingly, and runs fuzzy-correction
  (Levenshtein match against the user's known vocabulary) regardless of
  architecture. It skips KenLM/CTC beam search for Whisper (correct — that
  machinery is CTC-specific) but currently hardcodes confidence to `0.85` for
  Whisper (a stub, left as-is — see Non-Goals).
- `train_whisper_adapter.py` exists at the **repo root** (untracked in
  `src/cli/`, unlike its CTC sibling) with a `WhisperPersonalizedTrainer` using
  HF `Seq2SeqTrainer`, a working `DataCollatorSpeechSeq2SeqWithPadding`, and a
  LoRA config targeting `q_proj`/`v_proj`/`k_proj`. It is an unfinished
  prototype: defaults to `openai/whisper-medium`, **writes its output to
  `data/models/personalized_models/{user_id}` — the same path the live
  Wav2Vec2 model uses** (would silently clobber production), has no seeding,
  no quality/existence filtering, no early stopping, and reuses the exact
  `num_proc` pattern already fixed in `train_adapter.py` (must not be
  reintroduced here).
- `config.py` already has `GRADIENT_CHECKPOINTING` and `MIXED_PRECISION`
  constants with a comment noting they matter "especially for Whisper" —
  someone anticipated this work.
- `config.py` has three constants defined **twice** (`NUM_FINETUNE_EPOCHS`,
  `FINETUNE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`) — Python silently uses
  the second definition in each case. This is a cleanup target (section 4.5).
- `TrainingService.start_training()` (`src/services/training_service.py`)
  hardcodes `from src.cli.train_adapter import PersonalizedTrainer` — it does
  not read `settings["model"]` or `settings["algorithm"]` at all, despite both
  being stored per-user by `SettingsService`. The code contains a comment
  admitting this ("train_adapter.py şu an config'den direkt okuyor").
- The React web app (`web/`, in the `react-rewrite` worktree, already deployed
  to `https://app.furkanca.algorixstudio.com` this session) has an `AyarlarPage`
  with `MODEL_OPTIONS = ['Furkan', 'base', 'large']` and
  `ALGO_OPTIONS = ['lora', 'qlora', 'full']` — both decorative today.

## 3. Architecture

Two training paths stay fully independent end to end — separate trainer
classes, separate output directories, separate log/report naming — sharing
only two genuinely architecture-agnostic utility functions. No shared base
class or unified `Trainer` abstraction: `train_adapter.py` was just stabilized
this session and a shared-code refactor would put that stability at risk for
no clear benefit (YAGNI).

```
src/cli/train_adapter.py          # Wav2Vec2 CTC + LoRA (unchanged this work)
src/cli/train_whisper_adapter.py  # NEW location (moved from repo root), rehabilitated
src/utils/utils.py                # + set_seed(), + build_augment_pipeline() (deduped)
config.py                         # + WHISPER_MODEL_NAME, dedup'd constants
src/services/training_service.py  # branches on settings["model"]
src/services/settings_service.py  # unchanged schema, "model" gains real meaning
src/core/asr.py                   # + Whisper prompt-based hotword biasing
data/models/personalized_models/
  FurkanV1                        # live Wav2Vec2 adapter (untouched, still default)
  FurkanV1_whisper                # NEW: Whisper adapter output, always separate
```

## 4. Components and Data Flow

### 4.1 Shared utilities (`src/utils/utils.py`)

- `set_seed(seed: int)`: moved from `train_adapter.py` verbatim (seeds
  `random`, `numpy`, `torch`, `torch.cuda`). Both trainers import and call it
  from their CLI entry point before constructing the trainer object.
- `build_augment_pipeline(sampling_rate: int)`: moved from `train_adapter.py`
  verbatim (currently byte-for-byte duplicated in `train_whisper_adapter.py`).
  Both trainers import this one copy.

`train_adapter.py` and `train_whisper_adapter.py` are updated to import these
from `src.utils.utils` instead of defining them locally.

### 4.2 `src/cli/train_whisper_adapter.py` (rehabilitated)

Moved from the repo root. Brought to parity with `train_adapter.py`'s
robustness, applied to the Seq2Seq/Whisper shape:

- Default base model becomes `config.WHISPER_MODEL_NAME` (new constant,
  `"selimc/whisper-large-v3-turbo-turkish"`), not `openai/whisper-medium`.
- `self.output_dir` becomes `data/models/personalized_models/{user_id}_whisper`
  — **never** the bare `{user_id}` path. This is the load-bearing safety
  property of the whole design: this trainer can run at any time, including
  while the Wav2Vec2 model serves live traffic, without risk of collision.
- `prepare_dataset()` gains the same guards `train_adapter.py` has: raise on
  missing train data, return `None` on missing eval data (eval optional),
  filter out non-existent audio files, filter by `config.QUALITY_THRESHOLD`,
  filter empty transcripts after `normalize_turkish_text`. Reuses
  `resolve_audio_path` from `src.utils.utils` exactly as `train_adapter.py`
  does, for the same cross-platform path reasons.
- The dataset `.map()` call (feature extraction) applies the same `num_proc`
  safety pattern already fixed in `train_adapter.py`: `map_num_proc =
  num_proc if num_proc > 1 else None`, never the raw `num_proc` value. This
  must be correct from the first commit — it is the exact bug class that
  caused silent training failures and one production OOM this session.
- Add `EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)`
  to the `Seq2SeqTrainer`'s `callbacks=[...]`, reusing the existing constant
  (currently 30) for consistency with the CTC trainer's semantics (both are
  epoch-granularity patience).
- `set_seed(config.SEED)` called once at the top of `if __name__ == "__main__"`,
  before any model/data construction — same placement as `train_adapter.py`.
- Training report (via `ReportingService`, unchanged service) is extended to
  match `train_adapter.py`'s richer shape: `dataset_info` (train/eval sample
  counts), `sys_info` (platform, CUDA, GPU name/memory), `training_parameters`
  including `"model_type": "whisper"` and `"base_model"`, timing. This keeps
  `reports/*.json` comparable across both architectures and keeps
  `TrainingService.get_last_wer()` (which already generically reads
  `data.training_metrics.final_wer`/`best_wer` from any report file) working
  unmodified for Whisper reports.
- CLI signature unchanged: `python -m src.cli.train_whisper_adapter <user_id>
  [--base_model ...]`.

### 4.3 `src/core/asr.py` — hotword prompt biasing (Whisper only)

In the `model_type == "whisper"` branch of `transcribe()`, build a short prompt
string from `self.user_vocabulary` (already loaded for fuzzy-correction) and
pass it via Whisper's documented context-conditioning hook:

```python
if self.user_vocabulary:
    prompt_text = " ".join(self.user_vocabulary[:50])  # cap prompt length
    prompt_ids = self.processor.get_prompt_ids(prompt_text, return_tensors="pt").to(self.device)
    generated_ids = self.model.generate(input_features, language="tr", task="transcribe", prompt_ids=prompt_ids)
else:
    generated_ids = self.model.generate(input_features, language="tr", task="transcribe")
```

The 50-word cap keeps the prompt well within Whisper's context budget; the
exact cap is tuned during validation (section 6) if 50 proves too aggressive
or too weak. `_fuzzy_correct()` still runs afterward unchanged — prompt
biasing and fuzzy-correction are complementary (one nudges generation, the
other repairs the output), not a replacement for each other.

### 4.4 Settings become real: `settings["model"]` selects architecture

`SettingsService`'s schema and storage are unchanged (still a per-user JSON
file with the same fields). What changes is that `settings["model"]` gains
two meaningful values instead of being decorative:

- `"wav2vec2"` — default, matches all current behavior exactly.
- `"whisper"` — new.

**`TrainingService.start_training()`** reads `settings["model"]` and imports/
constructs `PersonalizedTrainer` or `WhisperPersonalizedTrainer` accordingly
(both already expose the same `__init__(user_id)` + a run method shape, so the
branch is a few lines, not a refactor of either trainer).

**Live inference model selection.** `api.py`'s `get_asr(user_id)` (line ~201)
currently does:

```python
def get_asr(user_id: str):
    if user_id not in asr_systems:
        model_path = ModelService.find_personalized_model(user_id)
        asr_systems[user_id] = ASRSystem(model_name=model_path, user_id=user_id)
    return asr_systems[user_id]
```

`ModelService.find_personalized_model(user_id)` (`src/services/model_service.py`)
just resolves `data/models/personalized_models/{user_id}` (plus a legacy root
fallback) — it has no notion of architecture or settings. `get_asr` changes to
resolve an architecture-suffixed lookup key and cache on *that* key instead of
bare `user_id`:

```python
def get_asr(user_id: str):
    settings = settings_service.load_settings(user_id)
    model_key = f"{user_id}_whisper" if settings.get("model") == "whisper" else user_id
    if model_key not in asr_systems:
        model_path = ModelService.find_personalized_model(model_key)
        asr_systems[model_key] = ASRSystem(model_name=model_path, user_id=user_id)
    return asr_systems[model_key]
```

Two details this preserves deliberately: `find_personalized_model` receives the
*suffixed* key (so it resolves `.../FurkanV1_whisper` unchanged, no edit needed
to `model_service.py` itself), while `ASRSystem(..., user_id=user_id)` keeps
the **unsuffixed** `user_id` — that argument drives `_load_user_vocabulary()`,
which reads `data/users/{user_id}/metadata_words.csv` and must never be
suffixed. Keying the cache dict by `model_key` rather than `user_id` also
sidesteps any cache-invalidation logic: switching `settings["model"]` simply
produces a cache miss on the new key on the next `/translate` call, with the
old entry left harmlessly unused in memory (one extra loaded model at most per
user, acceptable at this app's scale).

This is the entire cutover mechanism: once a Whisper adapter has been trained
and validated (section 6), switching production traffic to it is one `POST
/settings {"model": "whisper"}` call — no deploy, no file moves. Switching
back is the same call with `"wav2vec2"`. The Wav2Vec2 adapter directory is
never deleted or overwritten by any part of this design, so rollback is
always available.

`settings["algorithm"]` keeps its existing single valid value, `"lora"` — it
is not being extended to select CTC-vs-Whisper (that is `model`'s job now).
The React `AyarlarPage`'s `ALGO_OPTIONS` array is trimmed to `['lora']` and
`MODEL_OPTIONS` becomes `['wav2vec2', 'whisper']`, to stop presenting dead
options as if they were implemented. (This UI change lands in the
`react-rewrite` worktree, not master, since that's where `AyarlarPage` lives
in this session's work.)

### 4.5 `config.py` cleanup

Remove the first (shadowed, dead) definitions of `NUM_FINETUNE_EPOCHS`,
`FINETUNE_BATCH_SIZE`, and `GRADIENT_ACCUMULATION_STEPS` — keep only the
second definition of each (the one Python actually uses today: epochs=100,
batch_size=8, grad_accum=4), with its existing comment. This is a pure
dead-code removal with zero behavior change (verified by the fact that the
first definitions are already unreachable). Add `WHISPER_MODEL_NAME =
"selimc/whisper-large-v3-turbo-turkish"` near `MODEL_NAME`.

## 5. Error Handling

- **Output path collision** is prevented structurally (separate directory),
  not by a runtime check — see 4.2. No new error handling needed for this;
  it cannot happen by construction.
- **GPU memory fit** for the 0.8B-parameter checkpoint on the 12GB `L40S-12Q`
  slice is unverified. Rather than add defensive try/except around an OOM
  (which cannot be recovered from mid-training anyway), the plan's first task
  is a short smoke test (small subset, 1 epoch) to confirm the run fits before
  committing GPU time to a full run — see section 6.
- **Missing/failed Whisper training** must not affect the live Wav2Vec2 path:
  since `TrainingService` already wraps `run_training()` in try/except and
  records `status: "error"` per-user, this behavior is inherited for free once
  the branch is added — no new error handling required there.
- **`settings["model"] == "whisper"` with no Whisper adapter yet trained**:
  `/translate` should fail closed with a clear error (existing `ASRSystem`
  already raises on missing model directory in `_load_base_model`'s except
  branch, falling back to `config.MODEL_NAME`'s base model — unpersonalized
  but functional, not a hard crash). No new code needed; this existing
  fallback behavior is acceptable and is called out in the plan's testing
  section rather than silently assumed.

## 6. Testing / Validation Plan

1. **Smoke test** (new, first task): run the rehabilitated
   `train_whisper_adapter.py` for 1 epoch on a small slice of the data (or
   the full 6953-sample set with `NUM_FINETUNE_EPOCHS` temporarily overridden
   via CLI/env for this test only) to confirm it completes without OOM and
   produces a saved adapter at the `_whisper`-suffixed path. This is a
   feasibility check, not a quality check.
2. **Full training run**: same 5562/1391 train/eval split used since
   2026-08-11 (unchanged — this data has not changed since that date, per
   this session's investigation). Compare resulting WER/CER against both the
   historical best (64.99%, 2026-08-11) and the current Wav2Vec2 seeded
   baseline (82.50%, 2026-09-19).
3. **Settings wiring test**: `POST /settings {"model": "whisper"}`, confirm
   `/translate` loads the Whisper adapter (log line already exists:
   "Whisper mimarisi tespit edildi: ..."), confirm a live transcription
   round-trip works end to end on the deployed web app.
4. **Hotword biasing check**: manually compare a handful of transcriptions
   with and without `prompt_ids` set, using words from the user's actual
   vocabulary, to confirm the biasing measurably helps rather than degrades
   (autoregressive prompting can occasionally cause repetition artifacts —
   worth a manual sanity pass before trusting it in production).
5. **Rollback check**: `POST /settings {"model": "wav2vec2"}` after having
   switched to Whisper, confirm `/translate` serves the original model again
   with no restart required.

## 7. Non-Goals (explicitly deferred)

- Real Whisper confidence scoring (replacing the hardcoded `0.85`) — noted as
  a known gap, not fixed here.
- QLoRA or full fine-tuning for either architecture — `algorithm` stays
  single-valued (`"lora"`).
- Any change to the Flutter app's Ayarlar screen — it is being superseded by
  the React rewrite already in flight; only the React `AyarlarPage` is
  updated.
- The Furkanca live-listening VAD/state-machine bug (translation not
  restarting correctly if the user speaks again while a previous segment is
  still being analyzed) and the mobile app self-update/OTA mechanism raised
  in the same conversation as this spec — both are real, both are queued as
  their own follow-up work, neither is part of this spec.
