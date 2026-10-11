# -*- coding: utf-8 -*-
import os
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.asr import ASRSystem


class _FakeWhisperProcessor:
    def __call__(self, speech, sampling_rate, return_tensors):
        return SimpleNamespace(input_features=torch.zeros(1, 1))

    def get_prompt_ids(self, text, return_tensors):
        return torch.zeros(1, dtype=torch.long)

    def batch_decode(self, ids, skip_special_tokens):
        return ["okula"]


class _FakeWhisperModel:
    def generate(self, input_features, **kwargs):
        return torch.zeros(1, 1, dtype=torch.long)


def _fake_whisper_asr():
    asr = ASRSystem.__new__(ASRSystem)
    asr.model_type = "whisper"
    asr.device = "cpu"
    asr.user_vocabulary = ["okul"]
    asr.processor = _FakeWhisperProcessor()
    asr.model = _FakeWhisperModel()
    return asr


def _tone_wav(tmp_path):
    path = tmp_path / "tone.wav"
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    sf.write(path, 0.3 * np.sin(2 * np.pi * 220 * t), 16000)
    return str(path)


def test_transcribe_applies_fuzzy_correction_by_default(tmp_path):
    text, _ = _fake_whisper_asr().transcribe(_tone_wav(tmp_path))
    assert text == "okul"


def test_transcribe_can_skip_fuzzy_correction(tmp_path):
    text, _ = _fake_whisper_asr().transcribe(_tone_wav(tmp_path), fuzzy_correct=False)
    assert text == "okula"


class _RecordingWhisperModel(_FakeWhisperModel):
    def __init__(self):
        self.generate_kwargs = []

    def generate(self, input_features, **kwargs):
        self.generate_kwargs.append(kwargs)
        return super().generate(input_features, **kwargs)


def test_whisper_is_not_prompted_with_vocabulary_by_default(tmp_path):
    # Sozluk ipucu (prompt_ids) ipucusuz egitilen modelde dogrulugu %74'ten %55'e dusurdu
    asr = _fake_whisper_asr()
    asr.model = _RecordingWhisperModel()
    asr.transcribe(_tone_wav(tmp_path), fuzzy_correct=False)
    assert "prompt_ids" not in asr.model.generate_kwargs[0]
    assert asr.model.generate_kwargs[0]["language"] == "tr"
