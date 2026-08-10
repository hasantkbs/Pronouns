# Konuşmacı Doğrulamasıyla Sesle Tetikleme — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Furkanca ekranında VAD'ın yakaladığı ses segmentinin gerçekten kayıtlı kullanıcıya mı ait olduğunu sunucu tarafında doğrulayıp, eşleşmeyen segmentleri (başka konuşmacı/gürültü) sessizce filtrelemek.

**Architecture:** Yeni bir `SpeakerVerificationService` (SpeechBrain ECAPA-TDNN embedding'leri + kosinüs benzerliği) kullanıcının mevcut kelime kayıtlarından bir referans profil çıkarır ve `/translate` çağrısında ASR'den önce devreye girer. Eşleşmezse ağır ASR/NLU/sentez pipeline'ı hiç çalışmadan `{"rejected": true}` döner; istemci bunu sessizce yok sayar. Profil yoksa veya bir hata olursa her zaman fail-open (doğrulama atlanır).

**Tech Stack:** SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`), PyTorch/librosa (Python — bu ortamda `torchaudio.load`/`save` çalışmıyor, bkz. Global Constraints); `package:http` (Dart/Flutter).

**Spec:** `docs/superpowers/specs/2026-08-10-speaker-triggered-voice-design.md`

## Global Constraints

- **Ön koşul:** `docs/superpowers/plans/2026-08-10-vpnsiz-erisim.md` planı tamamlanmış olmalı — `api.py` zaten bir `X-API-Key` middleware'i içeriyor (whitelist: `/apk`, `/download/*`). Bu plandaki yeni uçlar (`/voice-profile/rebuild`, `/voice-profile/status`) whitelist'te DEĞİL, yani otomatik olarak anahtar gerektirirler; ek bir şey yapmaya gerek yok.
- Fail-open kuralı her zaman geçerli: profil yok / enrollment hatası / verify hatası → doğrulama atlanır, kullanıcı asla kilitlenmez.
- `config.SPEAKER_VERIFY_ENABLED = False` özelliği tamamen kapatır.
- Python sanal ortamı: `source /home/mayasoft/app/venv/bin/activate` (tüm `pytest`/`pip` komutlarından önce çalıştırılmalı).
- Ses dosyası okuma/yazma için `torchaudio.load`/`torchaudio.save` KULLANILMAZ — bu ortamdaki torchaudio sürümü, kurulu olmayan `torchcodec` paketini zorunlu kılıyor (doğrulandı: her ikisi de `ModuleNotFoundError: No module named 'torchcodec'` ile patlıyor). Bunun yerine proje genelinde zaten kullanılan `librosa.load` (okuma, `src/core/asr.py`'deki gibi) ve `soundfile.write` (test'lerde yazma) kullanılır — ikisi de `requirements.txt`'te zaten mevcut, yeni bağımlılık gerekmez.
- `speechbrain/spkrec-ecapa-voxceleb` modeli bu ortamda uçtan uca denendi ve doğrulandı: `EncoderClassifier.encode_batch(signal)` (signal shape `(1, N)`) 192 boyutlu bir embedding döndürüyor. Model `data/models/speaker_id/spkrec-ecapa-voxceleb` altına indirilip önbelleklendi (Task 1'de tekrar indirilmeyecek).

---

### Task 1: `SpeakerVerificationService` — enrollment ve doğrulama çekirdeği

**Files:**
- Modify: `requirements.txt` (speechbrain ekle)
- Modify: `config.py` (SPEAKER_* ayarları ekle)
- Create: `src/services/speaker_verification_service.py`
- Test: `tests/test_speaker_verification_service.py`

**Interfaces:**
- Produces: `SpeakerVerificationService(repo: Optional[UserDataRepository] = None)` sınıfı:
  - `enroll(user_id: str) -> dict` → `{"sample_count": int, "created_at": str | None}`
  - `verify(user_id: str, audio_path: str) -> tuple[bool, float | None]`
  - `status(user_id: str) -> dict` → `{"exists": bool, "sample_count": int, "created_at": str | None}`

- [ ] **Step 1: Bağımlılığı ekle ve kur**

`requirements.txt` içinde `soxr` satırının hemen altına, `starlette` satırından önce ekle:

```
speechbrain
```

Run: `cd /home/mayasoft/app/Pronouns && source /home/mayasoft/app/venv/bin/activate && pip install -r requirements.txt`
Expected: `speechbrain` kurulur (zaten kuruluysa no-op).

- [ ] **Step 2: `config.py`'ye ayarları ekle**

`config.py` dosyasının sonuna (Task 1 planındaki `API_SECRET_KEY` bloğunun altına) ekle:

```python

# --- Konuşmacı Doğrulama (Speaker Verification) Ayarları ---
# /translate'e gelen ses segmentinin gerçekten kayıtlı kullanıcıya mı ait
# olduğunu kontrol eder. False yapılırsa özellik tamamen devre dışı kalır.
SPEAKER_VERIFY_ENABLED = True
SPEAKER_VERIFY_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
# Kosinüs benzerliği eşiği (0-1). Başlangıç değeri; gerçek kayıtlarla
# kalibre edilmelidir (bkz. Task 4).
SPEAKER_VERIFY_THRESHOLD = 0.72
# Referans profil oluşturulurken kullanılacak azami kayıt sayısı.
SPEAKER_ENROLL_MAX_SAMPLES = 40
```

- [ ] **Step 3: Başarısız testi yaz**

`tests/test_speaker_verification_service.py` dosyasını oluştur:

```python
# -*- coding: utf-8 -*-
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.services.speaker_verification_service import SpeakerVerificationService


class _FakeEncoder:
    """encode_batch, dalga formunun basit istatistiklerinden sabit boyutlu bir
    'embedding' üretir; gerçek SpeechBrain modelini indirmeden enroll/verify
    akışını test etmeye yarar. Kosinüs benzerliği mantığını doğrular; gerçek
    konuşmacı ayırt etme doğruluğunu değil (bu, Task 4'ün kalibrasyon
    adımının konusudur)."""

    def encode_batch(self, signal):
        flat = signal.flatten()
        zcr = ((flat[:-1] * flat[1:]) < 0).float().mean()
        features = torch.stack([
            flat.mean(),
            zcr,
            flat.abs().mean(),
            flat.max(),
        ])
        return features.view(1, 1, -1)


def _write_wav(path: Path, freq: float, amplitude: float = 0.5,
               duration: float = 1.0, sr: int = 16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (amplitude * np.sin(2 * math.pi * freq * t)).astype(np.float32)
    sf.write(str(path), signal, sr)


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_PATH", str(tmp_path))
    monkeypatch.setattr(config, "SPEAKER_VERIFY_THRESHOLD", 0.99)
    monkeypatch.setattr(config, "SPEAKER_ENROLL_MAX_SAMPLES", 10)
    monkeypatch.setattr(
        SpeakerVerificationService, "_get_encoder",
        classmethod(lambda cls: _FakeEncoder()),
    )
    return SpeakerVerificationService()


def test_verify_without_profile_fails_open(service):
    is_match, score = service.verify("Nobody", "does-not-matter.wav")
    assert is_match is True
    assert score is None


def test_enroll_with_no_recordings_leaves_profile_missing(service, tmp_path):
    (tmp_path / "Empty" / "words").mkdir(parents=True)
    result = service.enroll("Empty")
    assert result == {"sample_count": 0, "created_at": None}

    is_match, score = service.verify("Empty", "does-not-matter.wav")
    assert is_match is True
    assert score is None


def test_enroll_then_verify_matching_voice(service, tmp_path):
    words_dir = tmp_path / "Furkan" / "words" / "merhaba"
    words_dir.mkdir(parents=True)
    for i in range(3):
        _write_wav(words_dir / f"rep{i + 1}.wav", freq=220.0)

    result = service.enroll("Furkan")
    assert result["sample_count"] == 3
    assert result["created_at"] is not None

    probe = tmp_path / "probe.wav"
    _write_wav(probe, freq=220.0)
    is_match, score = service.verify("Furkan", str(probe))
    assert is_match is True
    assert score > 0.99


def test_verify_rejects_different_voice(service, tmp_path):
    words_dir = tmp_path / "Furkan" / "words" / "merhaba"
    words_dir.mkdir(parents=True)
    for i in range(3):
        _write_wav(words_dir / f"rep{i + 1}.wav", freq=220.0)
    service.enroll("Furkan")

    probe = tmp_path / "probe.wav"
    _write_wav(probe, freq=880.0, amplitude=0.05)
    is_match, score = service.verify("Furkan", str(probe))
    assert is_match is False


def test_enroll_caps_sample_count(service, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPEAKER_ENROLL_MAX_SAMPLES", 2)
    words_dir = tmp_path / "Furkan" / "words" / "merhaba"
    words_dir.mkdir(parents=True)
    for i in range(5):
        _write_wav(words_dir / f"rep{i + 1}.wav", freq=220.0)

    result = service.enroll("Furkan")
    assert result["sample_count"] == 2


def test_status_reports_existing_profile(service, tmp_path):
    words_dir = tmp_path / "Furkan" / "words" / "merhaba"
    words_dir.mkdir(parents=True)
    _write_wav(words_dir / "rep1.wav", freq=220.0)
    service.enroll("Furkan")

    status = service.status("Furkan")
    assert status["exists"] is True
    assert status["sample_count"] == 1
    assert status["created_at"] is not None


def test_status_reports_missing_profile(service):
    status = service.status("Nobody")
    assert status == {"exists": False, "sample_count": 0, "created_at": None}
```

- [ ] **Step 3b: Testin şu an başarısız olduğunu doğrula**

Run: `cd /home/mayasoft/app/Pronouns && source /home/mayasoft/app/venv/bin/activate && python -m pytest tests/test_speaker_verification_service.py -v`
Expected: `ModuleNotFoundError` / `ImportError` — `src/services/speaker_verification_service.py` henüz yok.

- [ ] **Step 4: `SpeakerVerificationService`'i yaz**

`src/services/speaker_verification_service.py` dosyasını oluştur:

```python
# -*- coding: utf-8 -*-
"""
Speaker Verification Service - Gelen bir ses segmentinin, önceden
kaydedilmiş kullanıcı ses profiliyle (referans embedding) ne kadar
eşleştiğini SpeechBrain ECAPA-TDNN konuşmacı embedding'leri ve kosinüs
benzerliği üzerinden hesaplar.

Tasarım kararı: bu servis hiçbir zaman kullanıcıyı kilitlemez. Referans
profil yoksa, oluşturulamıyorsa veya embedding çıkarımı sırasında
beklenmeyen bir hata olursa doğrulama atlanır (fail-open) ve mevcut akış
olduğu gibi devam eder.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import librosa
import torch

import config
from src.constants import RECORD_TYPE_WORD
from src.data.repository import UserDataRepository

logger = logging.getLogger("speaker_verification")

_PROFILE_FILENAME = "voice_profile.pt"


class SpeakerVerificationService:
    """Kullanıcıya özel ses profili oluşturur ve gelen sesleri bu profille karşılaştırır."""

    _shared_encoder = None

    def __init__(self, repo: Optional[UserDataRepository] = None):
        self.repo = repo or UserDataRepository()
        self._profiles = {}  # user_id -> normalized centroid embedding (torch.Tensor)

    @classmethod
    def _get_encoder(cls):
        if cls._shared_encoder is None:
            from speechbrain.inference.speaker import EncoderClassifier
            model_dir_name = config.SPEAKER_VERIFY_MODEL.split("/")[-1]
            cls._shared_encoder = EncoderClassifier.from_hparams(
                source=config.SPEAKER_VERIFY_MODEL,
                savedir=f"data/models/speaker_id/{model_dir_name}",
            )
        return cls._shared_encoder

    def _profile_path(self, user_id: str) -> Path:
        return self.repo.get_user_path(user_id) / _PROFILE_FILENAME

    def _embed_file(self, audio_path: str) -> Optional[torch.Tensor]:
        try:
            speech, _sr = librosa.load(audio_path, sr=16000, mono=True)
            if len(speech) == 0:
                return None
            signal = torch.from_numpy(speech).unsqueeze(0)
            encoder = self._get_encoder()
            with torch.no_grad():
                embedding = encoder.encode_batch(signal).squeeze()
            return torch.nn.functional.normalize(embedding, dim=0)
        except Exception as e:
            logger.warning(f"Embedding çıkarılamadı ({audio_path}): {e}")
            return None

    def _load_profile(self, user_id: str) -> Optional[torch.Tensor]:
        if user_id in self._profiles:
            return self._profiles[user_id]
        path = self._profile_path(user_id)
        if not path.exists():
            return None
        try:
            data = torch.load(path, map_location="cpu")
            embedding = data["embedding"]
        except Exception as e:
            logger.warning(f"Ses profili okunamadı ({user_id}): {e}")
            return None
        self._profiles[user_id] = embedding
        return embedding

    def enroll(self, user_id: str) -> dict:
        """Kullanıcının mevcut kelime kayıtlarından bir referans ses profili oluşturur."""
        words_dir = self.repo.get_save_path(user_id, RECORD_TYPE_WORD)
        if not words_dir.exists():
            return {"sample_count": 0, "created_at": None}

        wav_paths = sorted(words_dir.rglob("*.wav"))[: config.SPEAKER_ENROLL_MAX_SAMPLES]

        embeddings = []
        for wav_path in wav_paths:
            embedding = self._embed_file(str(wav_path))
            if embedding is not None:
                embeddings.append(embedding)

        if not embeddings:
            return {"sample_count": 0, "created_at": None}

        centroid = torch.nn.functional.normalize(
            torch.stack(embeddings).mean(dim=0), dim=0
        )
        created_at = datetime.utcnow().isoformat()

        profile_path = self._profile_path(user_id)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embedding": centroid,
                "sample_count": len(embeddings),
                "created_at": created_at,
            },
            profile_path,
        )
        self._profiles[user_id] = centroid

        return {"sample_count": len(embeddings), "created_at": created_at}

    def verify(self, user_id: str, audio_path: str) -> Tuple[bool, Optional[float]]:
        """Ses segmentinin kayıtlı profille eşleşip eşleşmediğini döndürür.

        Profil yoksa veya embedding çıkarılamazsa fail-open: (True, None).
        """
        profile = self._load_profile(user_id)
        if profile is None:
            return True, None

        embedding = self._embed_file(audio_path)
        if embedding is None:
            return True, None

        score = float(torch.dot(embedding, profile))
        return score >= config.SPEAKER_VERIFY_THRESHOLD, score

    def status(self, user_id: str) -> dict:
        path = self._profile_path(user_id)
        if not path.exists():
            return {"exists": False, "sample_count": 0, "created_at": None}
        try:
            data = torch.load(path, map_location="cpu")
        except Exception as e:
            logger.warning(f"Ses profili okunamadı ({user_id}): {e}")
            return {"exists": False, "sample_count": 0, "created_at": None}
        return {
            "exists": True,
            "sample_count": data.get("sample_count", 0),
            "created_at": data.get("created_at"),
        }
```

- [ ] **Step 5: Testin artık geçtiğini doğrula**

Run: `cd /home/mayasoft/app/Pronouns && source /home/mayasoft/app/venv/bin/activate && python -m pytest tests/test_speaker_verification_service.py -v`
Expected: 7/7 PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt config.py src/services/speaker_verification_service.py tests/test_speaker_verification_service.py
git commit -m "feat: add SpeakerVerificationService (ECAPA-TDNN enrollment + cosine verify)"
```

---

### Task 2: `/translate` entegrasyonu ve ses profili uçları

**Files:**
- Modify: `api.py` (import bloğu, servis örneği, `/translate` handler, iki yeni endpoint)
- Test: `tests/test_translate_speaker_verification.py`

**Interfaces:**
- Consumes: Task 1'in `SpeakerVerificationService` sınıfı; `2026-08-10-vpnsiz-erisim.md` planının `X-API-Key` middleware'i (`config.API_SECRET_KEY`).
- Produces: `api.speaker_service` modül-seviyesi singleton. `/translate` artık `{"rejected": true, "reason": "speaker_mismatch"}` dönebilir. Yeni uçlar: `POST /voice-profile/rebuild?user_id=`, `GET /voice-profile/status?user_id=`.

- [ ] **Step 1: Başarısız testi yaz**

`tests/test_translate_speaker_verification.py` dosyasını oluştur:

```python
# -*- coding: utf-8 -*-
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import api
from fastapi.testclient import TestClient

client = TestClient(api.app)
_AUTH = {"X-API-Key": config.API_SECRET_KEY}


class _FakeAsr:
    def transcribe(self, path):
        return None, 0.0


class _FakeSpeakerService:
    def __init__(self, match=True, score=0.9):
        self.match = match
        self.score = score
        self.enroll_calls = []

    def verify(self, user_id, audio_path):
        return self.match, self.score

    def enroll(self, user_id):
        self.enroll_calls.append(user_id)
        return {"sample_count": 5, "created_at": "2026-08-10T00:00:00"}

    def status(self, user_id):
        return {"exists": True, "sample_count": 5, "created_at": "2026-08-10T00:00:00"}


def _fake_audio_file():
    return ("audio.wav", io.BytesIO(b"RIFF....WAVEfmt "), "audio/wav")


def test_translate_rejects_when_speaker_does_not_match(monkeypatch):
    monkeypatch.setattr(api, "speaker_service", _FakeSpeakerService(match=False))
    monkeypatch.setattr(api, "get_asr", lambda user_id: _FakeAsr())

    res = client.post(
        "/translate",
        data={"user_id": "FurkanV1"},
        files={"audio": _fake_audio_file()},
        headers=_AUTH,
    )

    assert res.status_code == 200
    assert res.json() == {"rejected": True, "reason": "speaker_mismatch"}


def test_translate_skips_verification_when_disabled(monkeypatch):
    monkeypatch.setattr(api, "speaker_service", _FakeSpeakerService(match=False))
    monkeypatch.setattr(api, "get_asr", lambda user_id: _FakeAsr())
    monkeypatch.setattr(config, "SPEAKER_VERIFY_ENABLED", False)

    res = client.post(
        "/translate",
        data={"user_id": "FurkanV1"},
        files={"audio": _fake_audio_file()},
        headers=_AUTH,
    )

    # Doğrulama devre dışı: reddedilmeden gerçek ASR akışına girer (sahte ASR
    # boş metin döndürdüğü için "Konuşma anlaşılamadı" ile sonuçlanır, ama
    # kesinlikle speaker_mismatch DEĞİL).
    assert res.json() == {"error": "Konuşma anlaşılamadı."}


def test_rebuild_voice_profile_endpoint(monkeypatch):
    fake = _FakeSpeakerService()
    monkeypatch.setattr(api, "speaker_service", fake)

    res = client.post("/voice-profile/rebuild?user_id=FurkanV1", headers=_AUTH)

    assert res.status_code == 200
    assert res.json() == {"sample_count": 5, "created_at": "2026-08-10T00:00:00"}
    assert fake.enroll_calls == ["FurkanV1"]


def test_voice_profile_status_endpoint(monkeypatch):
    monkeypatch.setattr(api, "speaker_service", _FakeSpeakerService())

    res = client.get("/voice-profile/status?user_id=FurkanV1", headers=_AUTH)

    assert res.status_code == 200
    assert res.json() == {
        "exists": True,
        "sample_count": 5,
        "created_at": "2026-08-10T00:00:00",
    }
```

- [ ] **Step 1b: Testin şu an başarısız olduğunu doğrula**

Run: `cd /home/mayasoft/app/Pronouns && source /home/mayasoft/app/venv/bin/activate && python -m pytest tests/test_translate_speaker_verification.py -v`
Expected: `/voice-profile/rebuild` ve `/voice-profile/status` `404` (henüz yok); ilk iki test de mevcut `/translate` doğrulama yapmadığı için FAIL.

- [ ] **Step 2: `api.py`'ye entegrasyonu ekle**

Import bloğunu ve servis örneklerini güncelle:

Eskisi:
```python
from src.data.repository import UserDataRepository
from src.utils.utils import calculate_audio_quality, normalize_path_for_cross_platform

app = FastAPI(title="Pronouns AI API")

# Repository instance
repo = UserDataRepository()
# Settings service instance
settings_service = SettingsService()
```

Yenisi:
```python
from src.data.repository import UserDataRepository
from src.services.speaker_verification_service import SpeakerVerificationService
from src.utils.utils import calculate_audio_quality, normalize_path_for_cross_platform

app = FastAPI(title="Pronouns AI API")

# Repository instance
repo = UserDataRepository()
# Settings service instance
settings_service = SettingsService()
# Konuşmacı doğrulama servisi
speaker_service = SpeakerVerificationService(repo)
```

`/translate` handler'ını güncelle:

Eskisi:
```python
    # Geçici dosyaya kaydet
    temp_id = str(uuid.uuid4())
    temp_input = f"temp_in_{temp_id}.wav"
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    # ASR
    asr = get_asr(user_id)
```

Yenisi:
```python
    # Geçici dosyaya kaydet
    temp_id = str(uuid.uuid4())
    temp_input = f"temp_in_{temp_id}.wav"
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    if config.SPEAKER_VERIFY_ENABLED:
        is_match, _score = speaker_service.verify(user_id, temp_input)
        if not is_match:
            os.remove(temp_input)
            return {"rejected": True, "reason": "speaker_mismatch"}

    # ASR
    asr = get_asr(user_id)
```

Yeni uçları `/translate` fonksiyonunun bitişinden (`missing_words` içeren `return` bloğu) hemen sonra, `/download/{filename}`'dan önce ekle:

Eskisi:
```python
    return {
        "recognized_text": text,
        "intent": intent,
        "response_text": action_response,
        "audio_url": f"/download/{os.path.basename(out_path)}" if out_path else None,
        "missing_words": missing
    }

@app.get("/download/{filename}")
async def download_audio(filename: str):
    return FileResponse(filename, media_type="audio/wav")
```

Yenisi:
```python
    return {
        "recognized_text": text,
        "intent": intent,
        "response_text": action_response,
        "audio_url": f"/download/{os.path.basename(out_path)}" if out_path else None,
        "missing_words": missing
    }

@app.post("/voice-profile/rebuild")
async def rebuild_voice_profile(user_id: str):
    """Kullanıcının mevcut kelime kayıtlarından ses profilini yeniden oluşturur."""
    return speaker_service.enroll(user_id)

@app.get("/voice-profile/status")
async def voice_profile_status(user_id: str):
    """Kullanıcının ses profilinin mevcut durumunu döndürür."""
    return speaker_service.status(user_id)

@app.get("/download/{filename}")
async def download_audio(filename: str):
    return FileResponse(filename, media_type="audio/wav")
```

- [ ] **Step 3: Testin artık geçtiğini doğrula**

Run: `cd /home/mayasoft/app/Pronouns && source /home/mayasoft/app/venv/bin/activate && python -m pytest tests/test_translate_speaker_verification.py -v`
Expected: 4/4 PASS.

- [ ] **Step 4: Middleware testinin hâlâ geçtiğini doğrula (regresyon kontrolü)**

Run: `cd /home/mayasoft/app/Pronouns && source /home/mayasoft/app/venv/bin/activate && python -m pytest tests/test_api_key_middleware.py tests/test_speaker_verification_service.py tests/test_translate_speaker_verification.py -v`
Expected: 16/16 PASS.

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_translate_speaker_verification.py
git commit -m "feat: gate /translate on speaker verification, add voice-profile endpoints"
```

---

### Task 3: İstemci — sessiz reddetme ve Ayarlar'da Ses Profili bölümü

**Files:**
- Modify: `lib/main.dart` (`_FurkancaPageState._handleSegment`, `_AyarlarPageState`)

**Interfaces:**
- Consumes: Task 2'nin `/translate` yanıtındaki `rejected` alanı; `/voice-profile/status` ve `/voice-profile/rebuild` uçları; `2026-08-10-vpnsiz-erisim.md` planının `_authHeaders()` yardımcı fonksiyonu.

- [ ] **Step 1: `_handleSegment`'te sessiz reddetmeyi ekle**

Eskisi:
```dart
      final d = jsonDecode(body) as Map<String, dynamic>;
      final corrected = d['response_text']?.toString() ?? '';

      if (!mounted) return;
      setState(() {
        _history.insert(
          0,
          _TranslationEntry(
            corrected.isEmpty ? '(boş yanıt)' : corrected,
            time: DateTime.now(),
          ),
        );
      });
```

Yenisi:
```dart
      final d = jsonDecode(body) as Map<String, dynamic>;

      if (d['rejected'] == true) {
        // Konuşmacı doğrulaması eşleşmedi: segment sessizce yok sayılır,
        // `finally` bloğu dinlemeye devam eder.
        return;
      }

      final corrected = d['response_text']?.toString() ?? '';

      if (!mounted) return;
      setState(() {
        _history.insert(
          0,
          _TranslationEntry(
            corrected.isEmpty ? '(boş yanıt)' : corrected,
            time: DateTime.now(),
          ),
        );
      });
```

- [ ] **Step 2: `_AyarlarPageState`'e durum alanlarını ekle**

Eskisi:
```dart
  String _trainStatus = '';

  final List<String> _modelOptions = ['Furkan', 'base', 'large'];
```

Yenisi:
```dart
  String _trainStatus = '';

  Map<String, dynamic>? _voiceProfileStatus;
  bool _voiceProfileBusy = false;

  final List<String> _modelOptions = ['Furkan', 'base', 'large'];
```

- [ ] **Step 3: `initState`'te durumu da yükle**

Eskisi:
```dart
  @override
  void initState() {
    super.initState();
    _fetchSettings();
  }
```

Yenisi:
```dart
  @override
  void initState() {
    super.initState();
    _fetchSettings();
    _fetchVoiceProfileStatus();
  }
```

- [ ] **Step 4: Ses profili yardımcı metodlarını ekle**

`_saveSettings()` metodunun hemen üstüne ekle:

```dart
  Future<void> _fetchVoiceProfileStatus() async {
    try {
      final res = await http.get(
        Uri.parse('${_base()}/voice-profile/status?user_id=$kUserId'),
        headers: _authHeaders(),
      );
      if (res.statusCode == 200) {
        final d = jsonDecode(res.body) as Map<String, dynamic>;
        if (!mounted) return;
        setState(() => _voiceProfileStatus = d);
      }
    } catch (_) {}
  }

  Future<void> _rebuildVoiceProfile() async {
    setState(() => _voiceProfileBusy = true);
    try {
      final res = await http.post(
        Uri.parse('${_base()}/voice-profile/rebuild?user_id=$kUserId'),
        headers: _authHeaders(),
      );
      if (res.statusCode == 200) {
        final d = jsonDecode(res.body) as Map<String, dynamic>;
        if (!mounted) return;
        setState(() => _voiceProfileStatus = {'exists': true, ...d});
        _snack('Ses profili güncellendi (${d['sample_count']} kayıt).');
      } else {
        _snack('Ses profili güncellenemedi: ${res.statusCode}');
      }
    } catch (e) {
      _snack('Hata: $e');
    } finally {
      if (mounted) setState(() => _voiceProfileBusy = false);
    }
  }

  String _voiceProfileText() {
    final status = _voiceProfileStatus;
    if (status == null) return 'Durum yükleniyor...';
    final exists = status['exists'] == true;
    if (!exists) return 'Henüz ses profili oluşturulmadı.';
    final count = status['sample_count'] ?? '-';
    final createdAt = status['created_at']?.toString() ?? '-';
    return '$count kayıttan oluşturuldu • $createdAt';
  }
```

- [ ] **Step 5: Ayarlar ekranına "Ses Profili" bölümünü ekle**

Eskisi (Fine-Tune butonunun bitişi ve `build`'in sonu):
```dart
                  // Fine-tune başlat
                  OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      foregroundColor: accent,
                      side: const BorderSide(color: accent),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14)),
                    ),
                    onPressed: (_saving || _training) ? null : _startFineTune,
                    icon: Icon(_training
                        ? Icons.hourglass_top
                        : Icons.model_training_rounded),
                    label: Text(
                      _training ? 'Eğitim başlatıldı...' : 'Fine-Tune Başlat',
                      style: const TextStyle(
                          fontSize: 15, fontWeight: FontWeight.w700),
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}
```

Yenisi:
```dart
                  // Fine-tune başlat
                  OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      foregroundColor: accent,
                      side: const BorderSide(color: accent),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14)),
                    ),
                    onPressed: (_saving || _training) ? null : _startFineTune,
                    icon: Icon(_training
                        ? Icons.hourglass_top
                        : Icons.model_training_rounded),
                    label: Text(
                      _training ? 'Eğitim başlatıldı...' : 'Fine-Tune Başlat',
                      style: const TextStyle(
                          fontSize: 15, fontWeight: FontWeight.w700),
                    ),
                  ),
                  const SizedBox(height: 24),

                  // ── Ses Profili ──
                  const _SectionHeader(label: 'Ses Profili'),
                  const SizedBox(height: 10),
                  _ResultTile(
                    label: 'Durum',
                    value: _voiceProfileText(),
                    color: accent,
                  ),
                  OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      foregroundColor: accent,
                      side: const BorderSide(color: accent),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14)),
                    ),
                    onPressed: _voiceProfileBusy ? null : _rebuildVoiceProfile,
                    icon: Icon(_voiceProfileBusy
                        ? Icons.hourglass_top
                        : Icons.record_voice_over_rounded),
                    label: Text(
                      _voiceProfileBusy
                          ? 'Güncelleniyor...'
                          : 'Ses Profilini Yenile',
                      style: const TextStyle(
                          fontSize: 15, fontWeight: FontWeight.w700),
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}
```

- [ ] **Step 6: Statik analiz ve mevcut testlerin bozulmadığını doğrula**

Run: `cd /home/mayasoft/app/Pronouns && flutter analyze`
Expected: Yeni hata yok.

Run: `cd /home/mayasoft/app/Pronouns && flutter test`
Expected: Tüm testler PASS.

- [ ] **Step 7: Commit**

```bash
git add lib/main.dart
git commit -m "feat: silently drop speaker-mismatched segments, add Voice Profile settings section"
```

---

### Task 4: Sunucuda profil oluştur, eşiği kalibre et, uçtan uca doğrula

**Files:** Yok (operasyon/kalibrasyon adımı — kod değişikliği içermez).

**Interfaces:**
- Consumes: Task 1-3'ün tamamlanmış hali, `data/users/FurkanV1/words/` altındaki mevcut kayıtlar.

- [ ] **Step 1: Sunucuyu güncelle ve yeniden başlat**

Run: `sudo systemctl restart pronouns-api.service`
Expected: `active (running)`. İlk `/translate` veya `/voice-profile/rebuild` çağrısında SpeechBrain modeli internetten indirilir (~80MB); loglardan takip edilebilir: `journalctl -u pronouns-api.service -f`.

- [ ] **Step 2: FurkanV1 için referans profili oluştur**

Run: `curl -X POST -H "X-API-Key: c2a6dd1afc678eb93f0c46c8c1528e2b1c07676fbde0951f" "http://88.255.236.143:8001/voice-profile/rebuild?user_id=FurkanV1"`
Expected: `{"sample_count": <N>, "created_at": "..."}` — `N` sıfırdan büyük olmalı (mevcut `words/` kayıtları kullanılıyor).

- [ ] **Step 3: Durumu doğrula**

Run: `curl -H "X-API-Key: c2a6dd1afc678eb93f0c46c8c1528e2b1c07676fbde0951f" "http://88.255.236.143:8001/voice-profile/status?user_id=FurkanV1"`
Expected: `{"exists": true, "sample_count": <N>, "created_at": "..."}`.

- [ ] **Step 4: Eşik değerini gerçek verilerle kalibre et**

`data/users/FurkanV1/words/` içinden enrollment'a girmemiş birkaç kayıt (varsa `Admin` klasöründeki farklı bir konuşmacının kayıtları) ile sunucuda kısa bir Python betiği çalıştırarak skor dağılımına bak:

```bash
source /home/mayasoft/app/venv/bin/activate
python3 - <<'EOF'
from src.services.speaker_verification_service import SpeakerVerificationService
from src.data.repository import UserDataRepository
from pathlib import Path

svc = SpeakerVerificationService(UserDataRepository())
svc.enroll("FurkanV1")

for wav in sorted(Path("data/users/Admin/words").rglob("*.wav"))[:10]:
    is_match, score = svc.verify("FurkanV1", str(wav))
    print(f"{wav.name}: match={is_match} score={score:.3f}")

for wav in sorted(Path("data/users/FurkanV1/words").rglob("*.wav"))[:10]:
    is_match, score = svc.verify("FurkanV1", str(wav))
    print(f"[Furkan] {wav.name}: match={is_match} score={score:.3f}")
EOF
```

Furkan'ın kendi kayıtları için skorlar, `Admin` kayıtlarının skorlarından belirgin şekilde yüksek çıkmalı. Aradaki boşluğa göre `config.py`'deki `SPEAKER_VERIFY_THRESHOLD` değerini ayarla (gerekirse), değişikliği commit et ve servisi yeniden başlat.

- [ ] **Step 5: Telefonda uçtan uca doğrula**

1. Furkanca ekranında "Başlat" → Furkan konuşunca normal yanıtın geldiğini doğrula.
2. Aynı ekranda başka biri konuşunca (veya arka planda TV/müzik gibi bir ses varken) geçmişe hiçbir şey eklenmediğini, herhangi bir hata gösterilmediğini doğrula.
3. Ayarlar → Ses Profili bölümünde durumun doğru göründüğünü, "Ses Profilini Yenile"ye basınca güncellendiğini doğrula.
