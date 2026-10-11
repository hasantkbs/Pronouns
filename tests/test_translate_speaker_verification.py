# -*- coding: utf-8 -*-
import logging
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
    user_vocabulary = []

    def transcribe(self, path, fuzzy_correct=True):
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
    monkeypatch.setattr(config, "SPEAKER_VERIFY_BLOCKING", True)
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


class _FakeTextAsr:
    user_vocabulary = ["okul"]

    def __init__(self, text="ben okla gldim"):
        self.text = text
        self.fuzzy_flags = []

    def transcribe(self, path, fuzzy_correct=True):
        self.fuzzy_flags.append(fuzzy_correct)
        return self.text, 0.85


class _FakeCorrector:
    def __init__(self, result="Ben okula geldim."):
        self.result = result
        self.inputs = []

    def correct(self, text):
        self.inputs.append(text)
        return self.result


class _FakeTts:
    def __init__(self, ok=True):
        self.ok = ok
        self.texts = []

    def synthesize(self, text, output_path):
        self.texts.append(text)
        return output_path if self.ok else None


def _post_translate():
    return client.post(
        "/translate",
        data={"user_id": "FurkanV1"},
        files={"audio": _fake_audio_file()},
        headers=_AUTH,
    )


def _patch_pipeline(monkeypatch, asr, corrector, tts):
    monkeypatch.setattr(config, "SPEAKER_VERIFY_ENABLED", False)
    monkeypatch.setattr(api, "get_asr", lambda user_id: asr)
    monkeypatch.setattr(api, "get_corrector", lambda user_id, vocabulary: corrector)
    monkeypatch.setattr(api, "tts", tts)


def test_translate_returns_corrected_sentence(monkeypatch):
    asr, corrector, tts = _FakeTextAsr(), _FakeCorrector(), _FakeTts()
    _patch_pipeline(monkeypatch, asr, corrector, tts)

    body = _post_translate().json()

    assert body["recognized_text"] == "ben okla gldim"
    assert body["response_text"] == "Ben okula geldim."
    assert body["intent"] == "translate"
    assert body["missing_words"] == []
    assert body["audio_url"].startswith("/download/")
    assert asr.fuzzy_flags == [False]
    assert corrector.inputs == ["ben okla gldim"]
    assert tts.texts == ["Ben okula geldim."]


def test_translate_without_audio_when_tts_fails(monkeypatch):
    _patch_pipeline(monkeypatch, _FakeTextAsr(), _FakeCorrector(), _FakeTts(ok=False))

    body = _post_translate().json()

    assert body["response_text"] == "Ben okula geldim."
    assert body["audio_url"] is None


def test_translate_empty_correction_is_not_understood(monkeypatch):
    tts = _FakeTts()
    _patch_pipeline(monkeypatch, _FakeTextAsr(text="..."), _FakeCorrector(result=""), tts)

    assert _post_translate().json() == {"error": "Konuşma anlaşılamadı."}
    assert tts.texts == []


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
