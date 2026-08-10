# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_request_without_key_is_rejected():
    res = client.get("/collect/word-sets")
    assert res.status_code == 401
    assert res.json()["error"] == "invalid_api_key"


def test_request_with_wrong_key_is_rejected():
    res = client.get(
        "/collect/word-sets", headers={"X-API-Key": "yanlis-anahtar"}
    )
    assert res.status_code == 401


def test_request_with_correct_key_is_allowed():
    res = client.get(
        "/collect/word-sets", headers={"X-API-Key": config.API_SECRET_KEY}
    )
    assert res.status_code == 200


def test_apk_download_is_exempt_from_key():
    res = client.get("/apk")
    assert res.status_code != 401


def test_download_path_is_exempt_from_key():
    res = client.get("/download/does-not-exist.wav")
    assert res.status_code != 401


def test_voice_profile_status_requires_key():
    res = client.get("/voice-profile/status?user_id=FurkanV1")
    assert res.status_code == 401


def test_voice_profile_rebuild_requires_key():
    res = client.post("/voice-profile/rebuild?user_id=FurkanV1")
    assert res.status_code == 401
