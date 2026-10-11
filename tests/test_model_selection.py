# -*- coding: utf-8 -*-
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api
from src.services.settings_service import SettingsService
from src.services.training_service import TrainingService


class _FakeAsrSystem:
    def __init__(self, model_name, user_id):
        self.model_name = model_name
        self.user_id = user_id


def _patch_asr(monkeypatch, model):
    lookups = []
    monkeypatch.setattr(api, "asr_systems", {})
    monkeypatch.setattr(api.settings_service, "load_settings", lambda user_id: {"model": model})
    monkeypatch.setattr(
        api.ModelService, "find_personalized_model",
        staticmethod(lambda key: lookups.append(key) or f"models/{key}"),
    )
    monkeypatch.setattr(api, "ASRSystem", _FakeAsrSystem)
    return lookups


def test_default_model_setting_is_wav2vec2(monkeypatch, tmp_path):
    monkeypatch.setattr(SettingsService, "get_settings_path", staticmethod(lambda uid: tmp_path / "settings.json"))
    assert SettingsService.load_settings("FurkanV1")["model"] == "wav2vec2"


def test_get_asr_uses_whisper_adapter_when_selected(monkeypatch):
    lookups = _patch_asr(monkeypatch, "whisper")
    asr = api.get_asr("FurkanV1")
    assert lookups == ["FurkanV1_whisper"]
    assert asr.model_name == "models/FurkanV1_whisper"
    # Kelime haznesi data/users/FurkanV1'den okunur - user_id asla son ek almamali
    assert asr.user_id == "FurkanV1"


def test_get_asr_uses_wav2vec2_adapter_by_default(monkeypatch):
    lookups = _patch_asr(monkeypatch, "wav2vec2")
    asr = api.get_asr("FurkanV1")
    assert lookups == ["FurkanV1"]
    assert asr.user_id == "FurkanV1"


def test_get_asr_does_not_reuse_cached_model_across_architectures(monkeypatch):
    _patch_asr(monkeypatch, "wav2vec2")
    ctc = api.get_asr("FurkanV1")
    monkeypatch.setattr(api.settings_service, "load_settings", lambda user_id: {"model": "whisper"})
    whisper = api.get_asr("FurkanV1")
    assert ctc.model_name == "models/FurkanV1"
    assert whisper.model_name == "models/FurkanV1_whisper"


def test_get_asr_evicts_other_architecture_on_switch(monkeypatch):
    # 12 GB GPU'da iki mimari ayni anda tutulursa CUDA OOM olur
    _patch_asr(monkeypatch, "wav2vec2")
    api.get_asr("FurkanV1")
    monkeypatch.setattr(api.settings_service, "load_settings", lambda user_id: {"model": "whisper"})
    api.get_asr("FurkanV1")
    assert set(api.asr_systems) == {"FurkanV1_whisper"}


def test_get_asr_keeps_other_users_models_on_switch(monkeypatch):
    _patch_asr(monkeypatch, "wav2vec2")
    api.get_asr("Admin")
    monkeypatch.setattr(api.settings_service, "load_settings", lambda user_id: {"model": "whisper"})
    api.get_asr("FurkanV1")
    assert set(api.asr_systems) == {"Admin", "FurkanV1_whisper"}


class _RunNow:
    def add_task(self, fn):
        fn()


def _patch_trainers(monkeypatch, model):
    ran = []

    def fake_trainer(name):
        class _Trainer:
            def __init__(self, user_id):
                self.user_id = user_id

            def run(self):
                ran.append((name, self.user_id))
        return _Trainer

    monkeypatch.setitem(sys.modules, "src.cli.train_adapter",
                        types.SimpleNamespace(PersonalizedTrainer=fake_trainer("wav2vec2")))
    monkeypatch.setitem(sys.modules, "src.cli.train_whisper_adapter",
                        types.SimpleNamespace(WhisperPersonalizedTrainer=fake_trainer("whisper")))
    service = TrainingService("TestUserModelSel")
    TrainingService._statuses["TestUserModelSel"]["status"] = "idle"
    monkeypatch.setattr(service.settings_service, "load_settings", lambda user_id: {"model": model})
    monkeypatch.setattr(service, "get_last_wer", lambda: None)
    return service, ran


def test_start_training_runs_whisper_trainer_when_selected(monkeypatch):
    service, ran = _patch_trainers(monkeypatch, "whisper")
    service.start_training(background_tasks=_RunNow())
    assert ran == [("whisper", "TestUserModelSel")]


def test_start_training_runs_wav2vec2_trainer_by_default(monkeypatch):
    service, ran = _patch_trainers(monkeypatch, "wav2vec2")
    service.start_training(background_tasks=_RunNow())
    assert ran == [("wav2vec2", "TestUserModelSel")]
