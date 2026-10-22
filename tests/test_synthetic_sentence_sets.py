# -*- coding: utf-8 -*-
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.utils.utils import normalize_turkish_text

ROOT = Path(__file__).resolve().parent.parent
SETS = ROOT / "datasets" / "sentence_sets"
EVAL_SET = SETS / "sentetik_olcum.txt"
TRAIN_SET = SETS / "sentetik_egitim.txt"
USER_DIR = ROOT / "data" / "users" / "FurkanV1"


def _sentences(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def _transcripts(path, column):
    # Kayıtlar kelime kelime ama "sosyal medya" gibi çok kelimeli girdiler
    # tek kayıttır; cümledeki kelime ancak tek başına kaydı varsa üretilebilir.
    if not path.exists():
        pytest.skip(f"{path} yok (canlı veri)")
    with open(path, encoding="utf-8") as f:
        return {normalize_turkish_text(row[column] or "") for row in csv.DictReader(f)}


def test_sets_have_expected_size_and_no_blank_lines():
    assert len(_sentences(EVAL_SET)) >= 140
    assert len(_sentences(TRAIN_SET)) >= 950
    for path in (EVAL_SET, TRAIN_SET):
        assert all(_sentences(path)), path.name


def test_sentences_have_three_to_seven_words():
    for path in (EVAL_SET, TRAIN_SET):
        for sentence in _sentences(path):
            assert 3 <= len(normalize_turkish_text(sentence).split()) <= 7, sentence


def test_eval_sentences_are_covered_by_eval_recordings():
    vocab = _transcripts(USER_DIR / "eval.csv", "transcript")
    for sentence in _sentences(EVAL_SET):
        missing = [w for w in normalize_turkish_text(sentence).split() if w not in vocab]
        assert not missing, f"{sentence}: {missing}"


def test_train_sentences_are_covered_by_train_recordings():
    vocab = _transcripts(USER_DIR / "train.csv", "transcript")
    for sentence in _sentences(TRAIN_SET):
        missing = [w for w in normalize_turkish_text(sentence).split() if w not in vocab]
        assert not missing, f"{sentence}: {missing}"


def test_no_duplicates_within_a_set():
    for path in (EVAL_SET, TRAIN_SET):
        normalized = [normalize_turkish_text(s) for s in _sentences(path)]
        assert len(normalized) == len(set(normalized)), path.name


def test_sets_are_disjoint_from_each_other_and_from_real_measurement_sets():
    eval_set = {normalize_turkish_text(s) for s in _sentences(EVAL_SET)}
    train_set = {normalize_turkish_text(s) for s in _sentences(TRAIN_SET)}
    real = {
        normalize_turkish_text(s)
        for path in SETS.glob("olcum_*.txt")
        for s in _sentences(path)
    }
    assert not eval_set & train_set
    assert not eval_set & real
    assert not train_set & real
