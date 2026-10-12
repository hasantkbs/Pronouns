# -*- coding: utf-8 -*-
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient

import api
import config

client = TestClient(api.app)


@pytest.fixture
def apk_dir(tmp_path, monkeypatch):
    # /apk ile ayni yer: BASE_PATH'in kardesi olan apk/ dizini
    monkeypatch.setattr(config, "BASE_PATH", str(tmp_path / "users"))
    d = tmp_path / "apk"
    d.mkdir()
    return d


def test_returns_version_file_without_api_key(apk_dir):
    info = {"versionCode": 3, "versionName": "1.2", "built_at": "2026-10-12T12:00:00"}
    (apk_dir / "version.json").write_text(json.dumps(info), encoding="utf-8")
    res = client.get("/apk/version")
    assert res.status_code == 200
    assert res.json() == info


def test_missing_version_file_means_version_zero(apk_dir):
    assert client.get("/apk/version").json() == {"versionCode": 0}


def test_broken_version_file_means_version_zero(apk_dir):
    (apk_dir / "version.json").write_text("{bozuk", encoding="utf-8")
    assert client.get("/apk/version").json() == {"versionCode": 0}


def test_version_without_code_means_version_zero(apk_dir):
    (apk_dir / "version.json").write_text(json.dumps({"versionName": "1.0"}), encoding="utf-8")
    assert client.get("/apk/version").json() == {"versionCode": 0}
