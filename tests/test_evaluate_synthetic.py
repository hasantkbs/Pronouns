# -*- coding: utf-8 -*-
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import soundfile as sf

from src.cli import evaluate_synthetic as es

SR = 16000


def test_decide_moves_to_training_at_five_points():
    assert es.decide(0.15, 0.20) == "egitime_gec"
    assert es.decide(0.15, 0.30) == "egitime_gec"


def test_decide_stops_below_five_points():
    assert es.decide(0.15, 0.199) == "dur"
    assert es.decide(0.20, 0.15) == "dur"


def test_sentence_rows_point_into_the_user_dir(tmp_path):
    rows = es.sentence_rows([{"file_path": "synthetic_eval/s001_1.wav", "transcription": "Kedi süt istedi."}], tmp_path)
    assert rows == [{
        "audio_path": str(tmp_path / "synthetic_eval/s001_1.wav"),
        "file_path": "synthetic_eval/s001_1.wav",
        "expected": "Kedi süt istedi.",
        "set_file": "sentetik_cumle",
    }]


def test_word_rows_filters_to_the_given_words():
    takes = {"kedi": ["/a/kedi1.wav", "/a/kedi2.wav"], "süt": ["/a/süt1.wav"], "uçak": ["/a/uçak1.wav"]}
    rows = es.word_rows(takes, {"kedi", "süt"})
    assert [(r["expected"], r["audio_path"]) for r in rows] == [
        ("kedi", "/a/kedi1.wav"), ("kedi", "/a/kedi2.wav"), ("süt", "/a/süt1.wav"),
    ]
    assert {r["set_file"] for r in rows} == {"kelime"}


def test_word_rows_without_filter_returns_all():
    takes = {"kedi": ["/a/kedi1.wav"], "uçak": ["/a/uçak1.wav"]}
    assert len(es.word_rows(takes)) == 2


def test_build_report_adds_difference_and_decision():
    result = {
        "summary": {
            "kelime": {"raw": {"wer": 0.1}},
            "sentetik_cumle": {"raw": {"wer": 0.3}},
        },
        "rows": [],
        "skipped": [],
    }
    report = es.build_report(result, ["Kedi uçak istedi."], "FurkanV1_whisper_cumle")
    assert report["difference"] == pytest.approx(0.2)
    assert report["decision"] == "egitime_gec"
    assert report["model"] == "FurkanV1_whisper_cumle"
    assert report["skipped_sentences"] == ["Kedi uçak istedi."]
    assert report["summary"] is result["summary"]


def test_main_without_eval_csv_exits_with_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(es.config, "BASE_PATH", str(tmp_path))
    assert es.main(["--user", "Yok"]) == 1
    assert "eval.csv yok" in capsys.readouterr().out


def _tone(path, seconds=0.3):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SR)) / SR
    pad = np.zeros(SR // 4, dtype=np.float32)
    sf.write(str(path), np.concatenate([pad, 0.3 * np.sin(2 * np.pi * 220 * t), pad]), SR)


class _FakeAsr:
    user_vocabulary = []
    model_type = "whisper"

    def transcribe(self, path, fuzzy_correct=True):
        # Kelimeleri doğru, cümleleri hep yarım tanıyan bir model
        if "synthetic_eval" in path:
            return ("kedi", 0.9)
        return (path.split(os.sep)[-2], 0.9)


def test_main_measures_words_and_sentences_and_writes_report(tmp_path, monkeypatch, capsys):
    user = tmp_path / "users" / "U"
    for word in ("kedi", "süt", "istedi", "uçak"):
        _tone(user / "words" / word / "rep1.wav")
    (user / "eval.csv").write_text(
        "file_path,transcript\n" + "".join(f"words/{w}/rep1.wav,{w}\n" for w in ("kedi", "süt", "istedi", "uçak")),
        encoding="utf-8",
    )
    sentences = tmp_path / "sentetik_olcum.txt"
    sentences.write_text("Kedi süt istedi.\n", encoding="utf-8")
    reports = tmp_path / "reports"

    monkeypatch.setattr(es.config, "BASE_PATH", str(tmp_path / "users"))
    monkeypatch.setattr(es.config, "REPORTS_DIR", str(reports))
    monkeypatch.setattr(es, "EVAL_SENTENCES", sentences)
    monkeypatch.setattr(es.ev, "_load_asr", lambda user_id, model_name=None: _FakeAsr())

    class _Corrector:
        def __init__(self, vocabulary, lexicon_path):
            pass

        def correct(self, text):
            return text

    import src.core.sentence_corrector as sc
    monkeypatch.setattr(sc, "SentenceCorrector", _Corrector)

    assert es.main(["--user", "U", "--model", "U_whisper_cumle"]) == 0

    out = capsys.readouterr().out
    assert "Karar: egitime_gec" in out
    report = json.loads(next(reports.glob("synthetic_eval_*.json")).read_text(encoding="utf-8"))
    assert report["summary"]["kelime"]["count"] == 3  # uçak cümlede yok
    assert report["summary"]["sentetik_cumle"]["count"] == 2
    assert report["word_wer"] == 0.0
    assert report["sentence_wer"] == pytest.approx(2 / 3)
    assert report["model"] == "U_whisper_cumle"


def test_main_full_words_measures_every_eval_word(tmp_path, monkeypatch):
    user = tmp_path / "users" / "U"
    for word in ("kedi", "süt", "istedi", "uçak"):
        _tone(user / "words" / word / "rep1.wav")
    (user / "eval.csv").write_text(
        "file_path,transcript\n" + "".join(f"words/{w}/rep1.wav,{w}\n" for w in ("kedi", "süt", "istedi", "uçak")),
        encoding="utf-8",
    )
    sentences = tmp_path / "sentetik_olcum.txt"
    sentences.write_text("Kedi süt istedi.\n", encoding="utf-8")
    reports = tmp_path / "reports"
    monkeypatch.setattr(es.config, "BASE_PATH", str(tmp_path / "users"))
    monkeypatch.setattr(es.config, "REPORTS_DIR", str(reports))
    monkeypatch.setattr(es, "EVAL_SENTENCES", sentences)
    monkeypatch.setattr(es.ev, "_load_asr", lambda user_id, model_name=None: _FakeAsr())
    import src.core.sentence_corrector as sc
    monkeypatch.setattr(sc, "SentenceCorrector", lambda vocabulary, lexicon_path: type("C", (), {"correct": staticmethod(lambda t: t)})())

    assert es.main(["--user", "U", "--full-words"]) == 0

    report = json.loads(next(reports.glob("synthetic_eval_*.json")).read_text(encoding="utf-8"))
    assert report["summary"]["kelime"]["count"] == 4
    assert report["full_words"] is True
