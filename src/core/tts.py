# -*- coding: utf-8 -*-
"""Piper ile yerel Turkce metin-konusma (CPU, GPU'ya dokunmaz)."""
import logging
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PiperTTS:
    def __init__(self, voice_path: Path):
        self.voice = None
        try:
            from piper import PiperVoice
            self.voice = PiperVoice.load(str(voice_path))
        except Exception as e:
            # Ses yoksa API yine acilir; /translate metni sessiz dondurur.
            logger.warning("Piper sesi yuklenemedi (%s): %s", voice_path, e)

    def synthesize(self, text: str, output_path: str) -> Optional[str]:
        if self.voice is None or not text or not text.strip():
            return None
        try:
            with wave.open(output_path, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)
            return output_path
        except Exception as e:
            logger.error("Piper sentez hatasi: %s", e)
            Path(output_path).unlink(missing_ok=True)
            return None
