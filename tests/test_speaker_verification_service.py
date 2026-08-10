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
