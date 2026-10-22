# -*- coding: utf-8 -*-
import csv
import json
import os
import sys
import wave
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.cli import evaluate_sentences as ev


def _silent_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)


def _metadata(user_dir, rows):
    user_dir.mkdir(parents=True, exist_ok=True)
    with open(user_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "transcription", "repetition", "set_file"])
        writer.writeheader()
        for i, (text, set_file) in enumerate(rows, start=1):
            rel = f"audio/s_{i}/rep1.wav"
            _silent_wav(user_dir / rel)
            writer.writerow({"file_path": rel, "transcription": text, "repetition": 1, "set_file": set_file})


def test_normalize_handles_turkish_case_and_punctuation():
    assert ev.normalize("İLAÇ Iıdır, değil mi?") == "ilaç ııdır değil mi"
    assert ev.normalize(None) == ""


def test_score_reports_wer_cer_and_exact_rate():
    result = ev.score(["su yok", "hava çok soğuk"], ["su yok", "hava çok"])
    assert result["exact"] == 0.5
    assert result["wer"] == pytest.approx(1 / 5)
    assert 0 < result["cer"] < 1


def test_load_rows_resolves_paths_relative_to_user_dir(tmp_path):
    user_dir = tmp_path / "U"
    _metadata(user_dir, [("Su yok.", "olcum_a.txt")])
    rows = ev.load_rows(user_dir)
    assert rows == [{
        "audio_path": str(user_dir / "audio/s_1/rep1.wav"),
        "file_path": "audio/s_1/rep1.wav",
        "expected": "Su yok.",
        "set_file": "olcum_a.txt",
    }]


def test_load_rows_without_metadata_is_empty(tmp_path):
    assert ev.load_rows(tmp_path) == []


def test_evaluate_groups_by_set_and_compares_raw_and_corrected(tmp_path):
    user_dir = tmp_path / "U"
    _metadata(user_dir, [
        ("Su yok.", "olcum_sozluk_ici.txt"),
        ("Hava çok soğuk.", "olcum_sozluk_ici.txt"),
        ("Başım ağrıyor.", "olcum_sozluk_disi.txt"),
    ])
    raw = {"su yok": "su yok", "hava çok soğuk": "hava çok sok", "başım ağrıyor": "başım"}
    by_path = {r["audio_path"]: raw[ev.normalize(r["expected"])] for r in ev.load_rows(user_dir)}
    fixes = {"hava çok sok": "Hava çok soğuk."}

    report = ev.evaluate(
        ev.load_rows(user_dir),
        transcribe=lambda path: by_path[path],
        correct=lambda text: fixes.get(text, text),
    )

    ici = report["summary"]["olcum_sozluk_ici.txt"]
    assert ici["count"] == 2
    assert ici["raw"]["exact"] == 0.5
    assert ici["corrected"]["exact"] == 1.0
    assert report["summary"]["olcum_sozluk_disi.txt"]["raw"]["wer"] == 0.5
    assert report["summary"]["toplam"]["count"] == 3
    assert report["skipped"] == []
    assert len(report["rows"]) == 3


def test_evaluate_counts_empty_asr_output_as_all_wrong(tmp_path):
    user_dir = tmp_path / "U"
    _metadata(user_dir, [("Su yok.", "olcum_a.txt")])
    report = ev.evaluate(ev.load_rows(user_dir), transcribe=lambda p: None, correct=lambda t: t or "")
    assert report["summary"]["toplam"]["raw"]["wer"] == 1.0
    assert report["rows"][0]["raw"] == ""


def test_evaluate_skips_missing_and_unreadable_audio(tmp_path):
    user_dir = tmp_path / "U"
    _metadata(user_dir, [("Su yok.", "olcum_a.txt"), ("Hava çok soğuk.", "olcum_a.txt"), ("Kim geldi?", "olcum_a.txt")])
    (user_dir / "audio/s_2/rep1.wav").unlink()
    (user_dir / "audio/s_3/rep1.wav").write_bytes(b"bozuk")
    calls = []

    report = ev.evaluate(
        ev.load_rows(user_dir),
        transcribe=lambda p: calls.append(p) or "su yok",
        correct=lambda t: t,
    )

    assert [s["file_path"] for s in report["skipped"]] == ["audio/s_2/rep1.wav", "audio/s_3/rep1.wav"]
    assert len(calls) == 1
    assert report["summary"]["toplam"]["count"] == 1


def test_worst_orders_by_raw_error(tmp_path):
    rows = [
        {"expected": "a", "raw": "a", "corrected": "a", "raw_wer": 0.0, "corrected_wer": 0.0},
        {"expected": "b", "raw": "x", "corrected": "x", "raw_wer": 1.0, "corrected_wer": 1.0},
        {"expected": "c", "raw": "c d", "corrected": "c", "raw_wer": 0.5, "corrected_wer": 0.0},
    ]
    assert [r["expected"] for r in ev.worst(rows, 2)] == ["b", "c"]


def test_write_report_uses_timestamped_name(tmp_path):
    out = ev.write_report({"summary": {}}, tmp_path, now=datetime(2026, 10, 12, 9, 5, 7))
    assert out == tmp_path / "sentence_eval_20261012-090507.json"
    assert json.loads(out.read_text(encoding="utf-8")) == {"summary": {}}


def test_main_without_recordings_exits_with_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev.config, "BASE_PATH", str(tmp_path))
    assert ev.main(["--user", "Yok"]) == 1
    assert "Cümle kaydı yok" in capsys.readouterr().out


def test_model_key_defaults_to_the_live_model_choice():
    assert ev.model_key("FurkanV1", "whisper") == "FurkanV1_whisper"
    assert ev.model_key("FurkanV1", "wav2vec2") == "FurkanV1"
    assert ev.model_key("FurkanV1", None) == "FurkanV1"


def test_model_key_uses_the_given_model_folder():
    assert ev.model_key("FurkanV1", "whisper", "FurkanV1_whisper_cumle") == "FurkanV1_whisper_cumle"


def test_write_report_accepts_a_prefix(tmp_path):
    out = ev.write_report({}, tmp_path, now=datetime(2026, 10, 22, 1, 2, 3), prefix="synthetic_eval")
    assert out == tmp_path / "synthetic_eval_20261022-010203.json"
