# -*- coding: utf-8 -*-
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient

import api
import config
from src.constants import DATASET_DIRS, RECORD_TYPE_SENTENCE, RECORD_TYPE_WORD
from src.data.repository import UserDataRepository

client = TestClient(api.app)
_AUTH = {"X-API-Key": config.API_SECRET_KEY}
USER = "TestUser"
S1 = "Çocuk süt istiyor."
S2 = "Uykum geldi, yatmak istiyorum."


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    users = tmp_path / "users"
    sentence_sets = tmp_path / "sentence_sets"
    word_sets = tmp_path / "words_set"
    sentence_sets.mkdir()
    word_sets.mkdir()
    (sentence_sets / "olcum_a.txt").write_text(f"{S1}\n{S2}\n", encoding="utf-8")
    (sentence_sets / "sentence_set_1.txt").write_text("eski cümle\n", encoding="utf-8")
    (word_sets / "wordSet.txt").write_text("elma\narmut\n", encoding="utf-8")
    monkeypatch.setattr(config, "BASE_PATH", str(users))
    monkeypatch.setattr(api, "repo", UserDataRepository(str(users)))
    monkeypatch.setitem(DATASET_DIRS, RECORD_TYPE_SENTENCE, str(sentence_sets))
    monkeypatch.setitem(DATASET_DIRS, RECORD_TYPE_WORD, str(word_sets))
    return users


def _write_sentence_rows(users, sentence, count):
    path = users / USER / "metadata.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["file_path", "transcription", "repetition", "set_file"])
        for rep in range(1, count + 1):
            writer.writerow([f"audio/x/rep{rep}.wav", sentence, rep, "olcum_a.txt"])


def _get(path, **params):
    return client.get(path, params=params, headers=_AUTH)


def test_word_sets_lists_only_measurement_sets_for_sentences(data_dirs):
    assert _get("/collect/word-sets", kind="sentence").json() == {"sets": ["olcum_a.txt"]}


def test_word_sets_without_kind_lists_word_sets(data_dirs):
    assert _get("/collect/word-sets").json() == {"sets": ["wordSet.txt"]}


def test_invalid_kind_is_rejected(data_dirs):
    assert _get("/collect/word-sets", kind="harf").status_code == 400
    assert _get("/collect/next-word", user_id=USER, kind="x").status_code == 400
    assert _get("/collect/progress", user_id=USER, kind="x").status_code == 400


def test_next_sentence_starts_with_first_sentence(data_dirs):
    data = _get("/collect/next-word", user_id=USER, set_file="olcum_a.txt", kind="sentence").json()
    assert data["done"] is False
    assert data["word"] == S1
    assert data["rep"] == 1
    assert data["ideal_repetitions"] == 2
    assert data["total_words"] == 2


def test_next_sentence_moves_on_after_two_recordings(data_dirs):
    _write_sentence_rows(data_dirs, S1, 1)
    data = _get("/collect/next-word", user_id=USER, set_file="olcum_a.txt", kind="sentence").json()
    assert (data["word"], data["rep"]) == (S1, 2)

    _write_sentence_rows(data_dirs, S1, 1)
    data = _get("/collect/next-word", user_id=USER, set_file="olcum_a.txt", kind="sentence").json()
    assert (data["word"], data["rep"]) == (S2, 1)


def test_next_sentence_exclude_is_newline_separated(data_dirs):
    # Cumlede virgul var: virgulle bolunseydi S2 hic eslesmezdi
    data = _get(
        "/collect/next-word", user_id=USER, set_file="olcum_a.txt", kind="sentence",
        exclude=f"{S1}\n{S2}",
    ).json()
    assert data["done"] is True


def test_sentence_mode_rejects_non_measurement_set(data_dirs):
    res = _get("/collect/next-word", user_id=USER, set_file="sentence_set_1.txt", kind="sentence")
    assert res.status_code == 404
    res = _get("/collect/progress", user_id=USER, set_file="yok.txt", kind="sentence")
    assert res.status_code == 404


def test_sentence_mode_rejects_path_outside_sets(data_dirs):
    res = _get("/collect/next-word", user_id=USER, set_file="../words_set/wordSet.txt", kind="sentence")
    assert res.status_code == 404


def test_sentence_progress_counts_completed_sentences(data_dirs):
    _write_sentence_rows(data_dirs, S1, 2)
    _write_sentence_rows(data_dirs, S2, 1)
    data = _get("/collect/progress", user_id=USER, set_file="olcum_a.txt", kind="sentence").json()
    assert data == {
        "set_file": "olcum_a.txt",
        "total_words": 2,
        "completed_words": 1,
        "ideal_repetitions": 2,
    }


def test_word_next_without_kind_is_unchanged(data_dirs):
    data = _get("/collect/next-word", user_id=USER, set_file="wordSet.txt", exclude="elma").json()
    assert (data["word"], data["rep"], data["ideal_repetitions"]) == ("armut", 1, config.IDEAL_REPETITIONS)
