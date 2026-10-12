# -*- coding: utf-8 -*-
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.utils import normalize_turkish_text

ROOT = Path(__file__).resolve().parent.parent
SETS = ROOT / "datasets" / "sentence_sets"
IN_VOCAB = SETS / "olcum_sozluk_ici.txt"
OUT_OF_VOCAB = SETS / "olcum_sozluk_disi.txt"


def _vocabulary():
    # SentenceCorrector ile ayni: cok kelimeli girdiler kelimelerine bolunur
    with open(ROOT / "data" / "users" / "FurkanV1" / "metadata_words.csv", encoding="utf-8") as f:
        return {
            w for row in csv.DictReader(f)
            for w in normalize_turkish_text(row["transcription"] or "").split()
        }


def _sentences(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_set_has_30_non_empty_sentences():
    for path in (IN_VOCAB, OUT_OF_VOCAB):
        sentences = _sentences(path)
        assert len(sentences) == 30, path.name
        assert all(sentences), path.name


def test_in_vocabulary_sentences_use_only_known_words():
    vocab = _vocabulary()
    for sentence in _sentences(IN_VOCAB):
        words = normalize_turkish_text(sentence).split()
        assert 3 <= len(words) <= 7, sentence
        missing = [w for w in words if w not in vocab]
        assert not missing, f"{sentence}: {missing}"


def test_out_of_vocabulary_sentences_have_an_unknown_word():
    vocab = _vocabulary()
    for sentence in _sentences(OUT_OF_VOCAB):
        words = normalize_turkish_text(sentence).split()
        assert any(w not in vocab for w in words), sentence


def test_no_sentence_repeats_across_sets():
    normalized = [
        normalize_turkish_text(s) for path in (IN_VOCAB, OUT_OF_VOCAB) for s in _sentences(path)
    ]
    assert len(normalized) == len(set(normalized))
