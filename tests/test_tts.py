# -*- coding: utf-8 -*-
import os
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.core.tts import PiperTTS

REAL_VOICE = ROOT / config.PIPER_VOICE_PATH


@pytest.mark.skipif(not REAL_VOICE.exists(), reason="Piper sesi indirilmemis")
def test_synthesizes_valid_wav(tmp_path):
    out = str(tmp_path / "out.wav")
    assert PiperTTS(REAL_VOICE).synthesize("Ben bugün okula geldim.", out) == out
    with wave.open(out, "rb") as wav:
        assert wav.getframerate() == 22050
        assert wav.getnchannels() == 1
        assert wav.getnframes() > 0


def test_missing_voice_returns_none(tmp_path):
    tts = PiperTTS(tmp_path / "yok.onnx")
    out = tmp_path / "out.wav"
    assert tts.synthesize("Merhaba.", str(out)) is None
    assert not out.exists()


@pytest.mark.skipif(not REAL_VOICE.exists(), reason="Piper sesi indirilmemis")
def test_empty_text_returns_none(tmp_path):
    assert PiperTTS(REAL_VOICE).synthesize("   ", str(tmp_path / "out.wav")) is None


def test_synthesis_error_returns_none_and_cleans_up(tmp_path):
    class _BrokenVoice:
        def synthesize_wav(self, text, wav_file):
            raise RuntimeError("bozuk model")

    tts = PiperTTS(tmp_path / "yok.onnx")
    tts.voice = _BrokenVoice()
    out = tmp_path / "out.wav"
    assert tts.synthesize("Merhaba.", str(out)) is None
    assert not out.exists()
