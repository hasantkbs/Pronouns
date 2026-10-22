# -*- coding: utf-8 -*-
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest

import config
from src.cli import train_whisper_adapter as tw

LIVE = tw.PERSONALIZED_ROOT / "FurkanV1_whisper"


def test_default_output_is_the_live_whisper_folder():
    # Self-learning (TrainingService) bugünkü gibi canlı klasöre eğitir
    assert tw.resolve_output_dir("FurkanV1") == LIVE


def test_output_name_writes_to_a_separate_folder():
    out = tw.resolve_output_dir("FurkanV1", "FurkanV1_whisper_cumle", str(LIVE))
    assert out == tw.PERSONALIZED_ROOT / "FurkanV1_whisper_cumle"


def test_output_name_cannot_be_the_live_folder():
    with pytest.raises(ValueError):
        tw.resolve_output_dir("FurkanV1", "FurkanV1_whisper")


def test_continuing_requires_an_output_name():
    with pytest.raises(ValueError):
        tw.resolve_output_dir("FurkanV1", None, str(LIVE))


def test_output_cannot_be_the_init_folder():
    init = tw.PERSONALIZED_ROOT / "FurkanV1_whisper_cumle"
    with pytest.raises(ValueError):
        tw.resolve_output_dir("FurkanV1", "FurkanV1_whisper_cumle", str(init))


def test_training_schedule_continuing_uses_short_low_lr_settings():
    assert tw.training_schedule(True) == (10, 1e-5, 3)
    assert tw.training_schedule(False) == (
        config.NUM_FINETUNE_EPOCHS, config.FINETUNE_LEARNING_RATE, config.EARLY_STOPPING_PATIENCE,
    )


def test_build_train_frame_adds_each_usable_sentence_repeats_times():
    words = pd.DataFrame({"file_path": ["/w/kedi1.wav"], "transcript": ["kedi"], "quality_score": [90]})
    takes = {"kedi": ["/w/kedi1.wav"], "süt": ["/w/süt1.wav"], "istedi": ["/w/istedi1.wav"]}
    df, skipped = tw.build_train_frame(words, ["Kedi süt istedi.", "Kedi uçak istedi."], takes, repeats=2)

    assert list(df.columns) == ["file_path", "transcript", "kind"]
    assert df["kind"].tolist() == ["word", "sentence", "sentence"]
    assert df[df["kind"] == "sentence"]["transcript"].tolist() == ["Kedi süt istedi."] * 2
    assert df[df["kind"] == "sentence"]["file_path"].tolist() == ["", ""]
    assert skipped == ["Kedi uçak istedi."]


def test_build_train_frame_default_repeats_from_config():
    words = pd.DataFrame({"file_path": [], "transcript": []})
    takes = {"kedi": ["a"], "süt": ["b"], "istedi": ["c"]}
    df, _ = tw.build_train_frame(words, ["Kedi süt istedi."], takes)
    assert len(df) == config.SYNTH_SENTENCE_REPEATS


def test_build_eval_frame_appends_synthetic_eval_audio(tmp_path):
    words = pd.DataFrame({"file_path": ["/w/kedi1.wav"], "transcript": ["kedi"]})
    manifest = [{"file_path": "synthetic_eval/s001_1.wav", "transcription": "Kedi süt istedi."}]
    df = tw.build_eval_frame(words, manifest, tmp_path)
    assert df["file_path"].tolist() == ["/w/kedi1.wav", str(tmp_path / "synthetic_eval/s001_1.wav")]
    assert df["transcript"].tolist() == ["kedi", "Kedi süt istedi."]
    assert set(df["kind"]) == {"word"}


def test_build_eval_frame_without_manifest_is_words_only():
    words = pd.DataFrame({"file_path": ["/w/kedi1.wav"], "transcript": ["kedi"]})
    assert len(tw.build_eval_frame(words, [], Path("."))) == 1


def test_parse_args_defaults_match_todays_cli():
    args = tw.parse_args(["FurkanV1"])
    assert args.init_from is None and args.output_name is None and args.no_sentences is False


def test_parse_args_continue_options():
    args = tw.parse_args([
        "FurkanV1", "--init-from", str(LIVE), "--output-name", "FurkanV1_whisper_cumle", "--no-sentences",
    ])
    assert args.init_from == str(LIVE)
    assert args.output_name == "FurkanV1_whisper_cumle"
    assert args.no_sentences is True


def test_side_report_is_not_picked_up_as_the_live_training_report(tmp_path):
    out = tw.write_side_report({"final_wer": 0.1}, "FurkanV1_whisper_cumle", tmp_path,
                               now=datetime(2026, 10, 22, 1, 2, 3))
    assert out.name == "whisper_side_FurkanV1_whisper_cumle_20261022_010203.json"
    assert json.loads(out.read_text(encoding="utf-8")) == {"final_wer": 0.1}
    assert not list(tmp_path.glob("training_FurkanV1_*.json"))
