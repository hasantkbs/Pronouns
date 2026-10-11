# -*- coding: utf-8 -*-
"""
Speaker Verification Service - Gelen bir ses segmentinin, önceden
kaydedilmiş kullanıcı ses profiliyle (referans embedding) ne kadar
eşleştiğini SpeechBrain ECAPA-TDNN konuşmacı embedding'leri ve kosinüs
benzerliği üzerinden hesaplar.

Tasarım kararı: bu servis hiçbir zaman kullanıcıyı kilitlemez. Referans
profil yoksa, oluşturulamıyorsa veya embedding çıkarımı sırasında
beklenmeyen bir hata olursa doğrulama atlanır (fail-open) ve mevcut akış
olduğu gibi devam eder.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import librosa
import torch

import config
from src.constants import RECORD_TYPE_WORD
from src.data.repository import UserDataRepository

logger = logging.getLogger("speaker_verification")

_PROFILE_FILENAME = "voice_profile.pt"


class SpeakerVerificationService:
    """Kullanıcıya özel ses profili oluşturur ve gelen sesleri bu profille karşılaştırır."""

    _shared_encoder = None

    def __init__(self, repo: Optional[UserDataRepository] = None):
        self.repo = repo or UserDataRepository()
        self._profiles = {}  # user_id -> normalized centroid embedding (torch.Tensor)

    @classmethod
    def _get_encoder(cls):
        if cls._shared_encoder is None:
            from speechbrain.inference.speaker import EncoderClassifier
            model_dir_name = config.SPEAKER_VERIFY_MODEL.split("/")[-1]
            cls._shared_encoder = EncoderClassifier.from_hparams(
                source=config.SPEAKER_VERIFY_MODEL,
                savedir=f"data/models/speaker_id/{model_dir_name}",
            )
        return cls._shared_encoder

    def _profile_path(self, user_id: str) -> Path:
        return self.repo.get_user_path(user_id) / _PROFILE_FILENAME

    def _embed_file(self, audio_path: str) -> Optional[torch.Tensor]:
        try:
            speech, _sr = librosa.load(audio_path, sr=16000, mono=True)
            if len(speech) == 0:
                return None
            signal = torch.from_numpy(speech).unsqueeze(0)
            encoder = self._get_encoder()
            with torch.no_grad():
                embedding = encoder.encode_batch(signal).squeeze()
            return torch.nn.functional.normalize(embedding, dim=0)
        except Exception as e:
            logger.warning(f"Embedding çıkarılamadı ({audio_path}): {e}")
            return None

    def _load_profile(self, user_id: str) -> Optional[torch.Tensor]:
        if user_id in self._profiles:
            return self._profiles[user_id]
        path = self._profile_path(user_id)
        if not path.exists():
            return None
        try:
            data = torch.load(path, map_location="cpu")
            embedding = data["embedding"]
        except Exception as e:
            logger.warning(f"Ses profili okunamadı ({user_id}): {e}")
            return None
        self._profiles[user_id] = embedding
        return embedding

    def enroll(self, user_id: str) -> dict:
        """Kullanıcının mevcut kelime kayıtlarından bir referans ses profili oluşturur."""
        words_dir = self.repo.get_save_path(user_id, RECORD_TYPE_WORD)
        if not words_dir.exists():
            return {"sample_count": 0, "created_at": None}

        wav_paths = sorted(words_dir.rglob("*.wav"))[: config.SPEAKER_ENROLL_MAX_SAMPLES]

        embeddings = []
        for wav_path in wav_paths:
            embedding = self._embed_file(str(wav_path))
            if embedding is not None:
                embeddings.append(embedding)

        if not embeddings:
            return {"sample_count": 0, "created_at": None}

        centroid = torch.nn.functional.normalize(
            torch.stack(embeddings).mean(dim=0), dim=0
        )
        created_at = datetime.now(timezone.utc).isoformat()

        profile_path = self._profile_path(user_id)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embedding": centroid,
                "sample_count": len(embeddings),
                "created_at": created_at,
            },
            profile_path,
        )
        self._profiles[user_id] = centroid

        return {"sample_count": len(embeddings), "created_at": created_at}

    def verify(self, user_id: str, audio_path: str) -> Tuple[bool, Optional[float]]:
        """Ses segmentinin kayıtlı profille eşleşip eşleşmediğini döndürür.

        Profil yoksa veya embedding çıkarılamazsa fail-open: (True, None).
        """
        profile = self._load_profile(user_id)
        if profile is None:
            return True, None

        embedding = self._embed_file(audio_path)
        if embedding is None:
            return True, None

        try:
            # Profil diskten CPU'ya yuklenir, embedding GPU'da olabilir
            score = float(torch.dot(embedding, profile.to(embedding.device)))
        except Exception as e:
            logger.warning(f"Benzerlik hesaplanamadı ({user_id}): {e}")
            return True, None

        return score >= config.SPEAKER_VERIFY_THRESHOLD, score

    def status(self, user_id: str) -> dict:
        path = self._profile_path(user_id)
        if not path.exists():
            return {"exists": False, "sample_count": 0, "created_at": None}
        try:
            data = torch.load(path, map_location="cpu")
        except Exception as e:
            logger.warning(f"Ses profili okunamadı ({user_id}): {e}")
            return {"exists": False, "sample_count": 0, "created_at": None}
        return {
            "exists": True,
            "sample_count": data.get("sample_count", 0),
            "created_at": data.get("created_at"),
        }
