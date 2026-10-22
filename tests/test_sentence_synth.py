# -*- coding: utf-8 -*-
import os
import random
import sys
import wave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.training import sentence_synth as ss

SR = 16000


def _word_audio(speech_s=0.5, silence_s=1.0, amp=0.3):
    """Ortada konuşma (sinüs), iki yanda sessizlik."""
    t = np.arange(int(speech_s * SR)) / SR
    speech = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    pad = np.zeros(int(silence_s * SR), dtype=np.float32)
    return np.concatenate([pad, speech, pad])


def _write_wav(path, frames=1600):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"\x01\x00" * frames)


def test_sentence_words_normalizes_case_and_punctuation():
    assert ss.sentence_words("İlaç ÇOK acı, değil mi?") == ["ilaç", "çok", "acı", "değil", "mi"]


def test_read_sentences_skips_blank_lines(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("Kedi süt istedi.\n\n  Çocuk okul gitti.  \n", encoding="utf-8")
    assert ss.read_sentences(path) == ["Kedi süt istedi.", "Çocuk okul gitti."]


def test_trim_word_removes_leading_and_trailing_silence():
    trimmed = ss.trim_word(_word_audio(speech_s=0.5, silence_s=1.0))
    assert 0.45 * SR <= len(trimmed) <= 0.7 * SR


def test_trim_word_keeps_fully_silent_audio():
    silent = np.zeros(SR, dtype=np.float32)
    assert len(ss.trim_word(silent)) == SR


def test_build_sentence_length_is_words_plus_gaps_plus_edges():
    takes = {"kedi": ["a"], "süt": ["b"], "istedi": ["c"]}
    audio = {"a": _word_audio(0.4), "b": _word_audio(0.5), "c": _word_audio(0.6)}
    out = ss.build_sentence(["kedi", "süt", "istedi"], takes, random.Random(1), sr=SR, load=audio.get)

    words_len = sum(len(ss.trim_word(a)) for a in audio.values())
    edges = 2 * int(ss.EDGE_S * SR)
    gaps = len(out) - words_len - edges
    assert 2 * int(ss.GAP_MIN_S * SR) <= gaps <= 2 * int(ss.GAP_MAX_S * SR)


def test_build_sentence_starts_and_ends_with_silence():
    takes = {"kedi": ["a"], "süt": ["b"]}
    audio = {"a": _word_audio(), "b": _word_audio()}
    out = ss.build_sentence(["kedi", "süt"], takes, random.Random(1), sr=SR, load=audio.get)
    edge = int(ss.EDGE_S * SR)
    assert not np.any(out[:edge]) and not np.any(out[-edge:])


def test_build_sentence_normalizes_peak():
    takes = {"kedi": ["a"], "süt": ["b"]}
    audio = {"a": _word_audio(amp=0.05), "b": _word_audio(amp=1.0)}
    out = ss.build_sentence(["kedi", "süt"], takes, random.Random(1), sr=SR, load=audio.get)
    assert np.max(np.abs(out)) == pytest.approx(ss.PEAK, abs=1e-3)


def test_build_sentence_same_seed_same_audio():
    takes = {"kedi": ["a1", "a2"], "süt": ["b1", "b2"]}
    audio = {k: _word_audio(0.3 + i * 0.1) for i, k in enumerate(["a1", "a2", "b1", "b2"])}
    first = ss.build_sentence(["kedi", "süt"], takes, random.Random(7), sr=SR, load=audio.get)
    second = ss.build_sentence(["kedi", "süt"], takes, random.Random(7), sr=SR, load=audio.get)
    assert np.array_equal(first, second)


def test_build_sentence_missing_word_raises():
    with pytest.raises(ss.MissingWordError):
        ss.build_sentence(["kedi", "uçak"], {"kedi": ["a"]}, random.Random(1), sr=SR,
                          load=lambda p: _word_audio())


def test_usable_sentences_splits_by_recording_coverage():
    takes = {"kedi": ["a"], "süt": ["b"], "istedi": ["c"]}
    usable, skipped = ss.usable_sentences(["Kedi süt istedi.", "Kedi uçak istedi."], takes)
    assert usable == ["Kedi süt istedi."]
    assert skipped == ["Kedi uçak istedi."]


def test_load_takes_groups_by_normalized_transcript_and_drops_bad_rows(tmp_path):
    user = tmp_path / "user"
    _write_wav(user / "words" / "Kedi" / "rep1.wav")
    _write_wav(user / "words" / "Kedi" / "rep2.wav")
    _write_wav(user / "words" / "süt" / "rep1.wav")
    (user / "words" / "bozuk").mkdir(parents=True)
    (user / "words" / "bozuk" / "rep1.wav").write_bytes(b"not a wav")
    csv_path = user / "eval.csv"
    csv_path.write_text(
        "file_path,transcript,quality_score\n"
        "words/Kedi/rep1.wav,Kedi,90\n"
        "/eski/makine/data/users/X/words/Kedi/rep2.wav,KEDİ,90\n"
        "words/süt/rep1.wav,süt,10\n"
        "words/bozuk/rep1.wav,bozuk,90\n"
        "words/yok/rep1.wav,yok,90\n",
        encoding="utf-8",
    )
    takes = ss.load_takes(csv_path, user)
    assert set(takes) == {"kedi"}
    assert len(takes["kedi"]) == 2


def _tone_wav(path, speech_s=0.3):
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), _word_audio(speech_s=speech_s, silence_s=0.5), SR)


def _eval_user(tmp_path):
    user = tmp_path / "user"
    for word in ("kedi", "süt", "istedi"):
        for rep in (1, 2):
            _tone_wav(user / "words" / word / f"rep{rep}.wav", speech_s=0.2 + 0.1 * rep)
    lines = ["file_path,transcript"] + [
        f"words/{w}/rep{r}.wav,{w}" for w in ("kedi", "süt", "istedi") for r in (1, 2)
    ]
    (user / "eval.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sentences = tmp_path / "sentetik_olcum.txt"
    sentences.write_text("Kedi süt istedi.\nKedi uçak istedi.\nSüt istedi kedi.\n", encoding="utf-8")
    return user, sentences


def test_ensure_eval_set_writes_two_takes_per_usable_sentence(tmp_path):
    user, sentences = _eval_user(tmp_path)
    rows, skipped = ss.ensure_eval_set(user, sentences, seed=1)

    assert [r["file_path"] for r in rows] == [
        "synthetic_eval/s001_1.wav", "synthetic_eval/s001_2.wav",
        "synthetic_eval/s003_1.wav", "synthetic_eval/s003_2.wav",
    ]
    assert rows[0]["transcription"] == "Kedi süt istedi."
    assert skipped == ["Kedi uçak istedi."]
    assert all((user / r["file_path"]).exists() for r in rows)
    assert ss.read_manifest(user) == rows


def test_ensure_eval_set_reuses_existing_audio(tmp_path):
    user, sentences = _eval_user(tmp_path)
    ss.ensure_eval_set(user, sentences, seed=1)
    first = (user / "synthetic_eval" / "s001_1.wav").read_bytes()

    rows, skipped = ss.ensure_eval_set(user, sentences, seed=999)

    assert (user / "synthetic_eval" / "s001_1.wav").read_bytes() == first
    assert len(rows) == 4
    assert skipped == ["Kedi uçak istedi."]


def test_ensure_eval_set_same_seed_same_audio(tmp_path):
    user_a, sentences_a = _eval_user(tmp_path / "a")
    user_b, sentences_b = _eval_user(tmp_path / "b")
    ss.ensure_eval_set(user_a, sentences_a, seed=3)
    ss.ensure_eval_set(user_b, sentences_b, seed=3)
    for name in ("s001_1.wav", "s003_2.wav"):
        assert (user_a / "synthetic_eval" / name).read_bytes() == (user_b / "synthetic_eval" / name).read_bytes()


def test_read_manifest_without_eval_set_is_empty(tmp_path):
    assert ss.read_manifest(tmp_path) == []


def test_ensure_eval_set_regenerates_when_eval_csv_changes(tmp_path):
    # eval.csv yeniden bölünürse eski sesler artık eğitimdeki kayıtlardan olabilir (sızıntı)
    user, sentences = _eval_user(tmp_path)
    ss.ensure_eval_set(user, sentences, seed=1)
    lines = (user / "eval.csv").read_text(encoding="utf-8").splitlines()
    (user / "eval.csv").write_text("\n".join(l for l in lines if "süt" not in l) + "\n", encoding="utf-8")

    rows, skipped = ss.ensure_eval_set(user, sentences, seed=1)

    assert rows == []
    assert skipped == ["Kedi süt istedi.", "Kedi uçak istedi.", "Süt istedi kedi."]
    assert not list((user / "synthetic_eval").glob("*.wav"))


def test_current_manifest_requires_matching_sources(tmp_path):
    user, sentences = _eval_user(tmp_path)
    assert ss.current_manifest(user, sentences) == []
    rows, _ = ss.ensure_eval_set(user, sentences)
    assert ss.current_manifest(user, sentences) == rows

    sentences.write_text("Kedi süt istedi.\n", encoding="utf-8")
    assert ss.current_manifest(user, sentences) == []
