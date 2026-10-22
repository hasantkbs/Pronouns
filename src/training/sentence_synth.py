# -*- coding: utf-8 -*-
"""
Kelime kayıtlarından sentetik cümle sesi üretir (alt proje B2).

Her kelime için o kelimenin kayıtlarından biri seçilir, baş/son sessizliği
kırpılır, tepe genliği eşitlenir ve kelimeler kısa, rastgele duraklarla
birleştirilir. Ölçüm sesleri eval.csv'den, eğitim sesleri train.csv'den
üretilir; böylece ölçülen ses eğitimde hiç duyulmaz.
"""
import hashlib
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from src.utils.utils import normalize_turkish_text, resolve_audio_path

TRIM_TOP_DB = 30
PEAK = 0.9
GAP_MIN_S = 0.08
GAP_MAX_S = 0.35
EDGE_S = 0.2


class MissingWordError(KeyError):
    """Cümledeki bir kelimenin ilgili bölümde (train/eval) kaydı yok."""


def sentence_words(sentence: str) -> List[str]:
    return normalize_turkish_text(sentence).split()


def read_sentences(path) -> List[str]:
    """Satır başına bir cümle; boş satırlar atlanır."""
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_takes(csv_path, user_dir) -> Dict[str, List[str]]:
    """train.csv / eval.csv kayıtlarını normalize transkripte göre gruplar."""
    import soundfile as sf

    df = pd.read_csv(csv_path, encoding="utf-8")
    if "quality_score" in df.columns:
        df = df[df["quality_score"] >= config.QUALITY_THRESHOLD]
    takes: Dict[str, List[str]] = {}
    for raw_path, transcript in zip(df["file_path"], df["transcript"]):
        key = normalize_turkish_text(transcript if isinstance(transcript, str) else "")
        path = resolve_audio_path(raw_path, Path(user_dir))
        if not key or path is None:
            continue
        try:
            sf.info(path)
        except Exception:
            continue
        takes.setdefault(key, []).append(path)
    return takes


def usable_sentences(
    sentences: List[str], takes_by_word: Dict[str, List[str]]
) -> Tuple[List[str], List[str]]:
    """(her kelimesinin kaydı olan cümleler, atlanan cümleler)."""
    usable, skipped = [], []
    for sentence in sentences:
        words = sentence_words(sentence)
        if words and all(takes_by_word.get(w) for w in words):
            usable.append(sentence)
        else:
            skipped.append(sentence)
    return usable, skipped


def trim_word(audio: np.ndarray, top_db: int = TRIM_TOP_DB) -> np.ndarray:
    """Baş/son sessizliği kırpar; geriye ses kalmazsa orijinali döndürür."""
    import librosa

    if audio.size == 0 or not np.any(audio):
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed if trimmed.size else audio


def _load_audio(path: str, sr: int) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=sr)
    return audio


def build_sentence(
    words: List[str],
    takes_by_word: Dict[str, List[str]],
    rng: random.Random,
    sr: int = config.ORNEKLEME_ORANI,
    load: Optional[Callable[[str], np.ndarray]] = None,
) -> np.ndarray:
    load = load or (lambda path: _load_audio(path, sr))
    edge = np.zeros(int(EDGE_S * sr), dtype=np.float32)
    parts = [edge]
    for i, word in enumerate(words):
        takes = takes_by_word.get(word)
        if not takes:
            raise MissingWordError(word)
        audio = trim_word(np.asarray(load(rng.choice(takes)), dtype=np.float32))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio * (PEAK / peak)
        if i > 0:
            parts.append(np.zeros(int(rng.uniform(GAP_MIN_S, GAP_MAX_S) * sr), dtype=np.float32))
        parts.append(audio.astype(np.float32))
    parts.append(edge)
    return np.concatenate(parts)


SYNTH_EVAL_DIR = "synthetic_eval"
MANIFEST = "manifest.csv"
SKIPPED = "skipped.txt"
SOURCES = "sources.sha1"
EVAL_SENTENCES = Path("datasets") / "sentence_sets" / "sentetik_olcum.txt"


def read_manifest(user_dir) -> List[Dict[str, str]]:
    """synthetic_eval/manifest.csv satırları (file_path kullanıcı dizinine göre göreli); yoksa []."""
    path = Path(user_dir) / SYNTH_EVAL_DIR / MANIFEST
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8")
    return df.to_dict("records")


def source_fingerprint(user_dir, sentences_path) -> str:
    """Ölçüm seslerinin kaynağı: eval.csv + cümle dosyası. Biri değişirse sesler bayattır."""
    digest = hashlib.sha1()
    digest.update((Path(user_dir) / "eval.csv").read_bytes())
    digest.update(Path(sentences_path).read_bytes())
    return digest.hexdigest()


def current_manifest(user_dir, sentences_path) -> List[Dict[str, str]]:
    """
    Manifest satırları yalnızca sesler bugünkü eval.csv ve cümle dosyasından
    üretildiyse; aksi halde []. eval.csv yeniden bölününce eski sesler eğitim
    kayıtlarından olabilir ve ölçüm sızdırır.
    """
    stamp = Path(user_dir) / SYNTH_EVAL_DIR / SOURCES
    if not stamp.exists() or not (Path(user_dir) / "eval.csv").exists():
        return []
    if stamp.read_text(encoding="utf-8").strip() != source_fingerprint(user_dir, sentences_path):
        return []
    return read_manifest(user_dir)


def ensure_eval_set(
    user_dir,
    sentences_path,
    seed: int = config.SEED,
    per_sentence: int = 2,
    sr: int = config.ORNEKLEME_ORANI,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Sabit sentetik ölçüm seslerini döndürür; yoksa ya da eval.csv / cümle dosyası
    değiştiyse eval.csv kayıtlarından yeniden üretir. Manifest en son yazılır:
    üretim yarıda kalırsa bir sonraki çağrı baştan üretir.
    """
    import soundfile as sf

    user_dir = Path(user_dir)
    out_dir = user_dir / SYNTH_EVAL_DIR
    rows = current_manifest(user_dir, sentences_path)
    if rows:
        skipped_path = out_dir / SKIPPED
        skipped = read_sentences(skipped_path) if skipped_path.exists() else []
        return rows, skipped

    sentences = read_sentences(sentences_path)
    takes = load_takes(user_dir / "eval.csv", user_dir)
    usable, skipped = usable_sentences(sentences, takes)
    usable_set = set(usable)
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST).unlink(missing_ok=True)
    for old in out_dir.glob("*.wav"):
        old.unlink()
    rows = []
    for index, sentence in enumerate(sentences, start=1):
        if sentence not in usable_set:
            continue
        for n in range(1, per_sentence + 1):
            audio = build_sentence(sentence_words(sentence), takes, rng, sr=sr)
            rel = f"{SYNTH_EVAL_DIR}/s{index:03d}_{n}.wav"
            sf.write(str(user_dir / rel), audio, sr)
            rows.append({"file_path": rel, "transcription": sentence})
    (out_dir / SKIPPED).write_text("".join(f"{s}\n" for s in skipped), encoding="utf-8")
    (out_dir / SOURCES).write_text(source_fingerprint(user_dir, sentences_path), encoding="utf-8")
    pd.DataFrame(rows, columns=["file_path", "transcription"]).to_csv(
        out_dir / MANIFEST, index=False, encoding="utf-8"
    )
    return rows, skipped
