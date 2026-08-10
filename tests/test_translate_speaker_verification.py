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
