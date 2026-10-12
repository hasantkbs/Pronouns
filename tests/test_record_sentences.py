# -*- coding: utf-8 -*-
import csv
import hashlib
import io
import math
import os
import struct
import sys
import wave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient

import api
import config
from src.constants import DATASET_DIRS, RECORD_TYPE_SENTENCE
from src.data.repository import UserDataRepository

client = TestClient(api.app)
_AUTH = {"X-API-Key": config.API_SECRET_KEY}
USER = "TestUser"
SENTENCE = "Uykum geldi, yatmak istiyorum."


def _wav_bytes(seconds=1.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / rate)))
            for i in range(int(seconds * rate))
        )
        w.writeframes(frames)
    return buf.getvalue()


class _FakeTraining:
    def __init__(self):
        self.started = 0

    def start_training(self, background_tasks):
        self.started += 1


@pytest.fixture
def env(tmp_path, monkeypatch):
    users = tmp_path / "users"
    sets = tmp_path / "sentence_sets"
    sets.mkdir()
    (sets / "olcum_sozluk_disi.txt").write_text(f"{SENTENCE}\n", encoding="utf-8")
    monkeypatch.setattr(config, "BASE_PATH", str(users))
    monkeypatch.setattr(api, "repo", UserDataRepository(str(users)))
    monkeypatch.setitem(DATASET_DIRS, RECORD_TYPE_SENTENCE, str(sets))
    monkeypatch.setattr(api.settings_service, "load_settings", lambda user_id: {"self_learning": True})
    training = _FakeTraining()
    monkeypatch.setattr(api, "get_training_service", lambda user_id: training)
    return users, training


def _post(word, kind=None):
    data = {"user_id": USER, "word": word}
    if kind is not None:
        data["kind"] = kind
    return client.post(
        "/record",
        data=data,
        files={"audio": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        headers=_AUTH,
    )


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sentence_recording_is_saved_under_audio_with_hash_dir(env):
    users, _ = env
    res = _post(SENTENCE, kind="sentence")
    assert res.status_code == 200
    assert res.json()["rep"] == 1

    digest = hashlib.sha1(SENTENCE.encode("utf-8")).hexdigest()[:10]
    assert (users / USER / "audio" / f"s_{digest}" / "rep1.wav").exists()

    rows = _rows(users / USER / "metadata.csv")
    assert len(rows) == 1
    assert rows[0]["transcription"] == SENTENCE
    assert rows[0]["file_path"] == f"audio/s_{digest}/rep1.wav"
    assert rows[0]["repetition"] == "1"
    assert rows[0]["set_file"] == "olcum_sozluk_disi.txt"
    assert not (users / USER / "metadata_words.csv").exists()


def test_second_sentence_recording_is_rep_two(env):
    _post(SENTENCE, kind="sentence")
    assert _post(SENTENCE, kind="sentence").json()["rep"] == 2


def test_sentence_recording_never_starts_training(env):
    _, training = env
    for _ in range(config.IDEAL_REPETITIONS):
        _post(SENTENCE, kind="sentence")
    assert training.started == 0


def test_unknown_sentence_is_rejected(env):
    users, _ = env
    res = _post("Listede olmayan cümle.", kind="sentence")
    assert res.status_code == 400
    assert not (users / USER / "audio").exists()


def test_invalid_kind_is_rejected(env):
    assert _post("elma", kind="harf").status_code == 400


def test_record_without_kind_still_saves_a_word(env):
    users, training = env
    for _ in range(config.IDEAL_REPETITIONS):
        res = _post("elma")
    assert res.status_code == 200
    assert (users / USER / "words" / "elma" / f"rep{config.IDEAL_REPETITIONS}.wav").exists()
    assert len(_rows(users / USER / "metadata_words.csv")) == config.IDEAL_REPETITIONS
    assert not (users / USER / "metadata.csv").exists()
    assert training.started == 1
