# -*- coding: utf-8 -*-
"""
Kelime Tabanlı Konuşma Sentezleyici
====================================
Bireyin kaydedilmiş kelime örneklerini birleştirerek anlaşılır cümleler üretir.

Temel akış:
1. Hedef cümledeki her kelime için en kaliteli kayıtları seç.
2. Seçilen WAV dosyalarını crossfade + duraklama ile birleştir.
3. Sonuç dosyasını çal veya döndür.

Bu bileşen, konuşma bozukluğu olan bireyin kendi ses kalıplarıyla
başka bir bireyle anlaşılabilir iletişim kurmasını sağlar.
"""

import os
import csv
import random
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import numpy as np
    import soundfile as sf
    import librosa
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

import config


class WordSynthesizer:
    """
    Kaydedilmiş kelime örneklerinden cümle sentezi yapar.

    Parameters
    ----------
    user_id : str
        Kullanıcı kimliği; kayıtlar ``data/users/<user_id>/words/`` altında aranır.
    metadata_path : str | None
        metadata_words.csv yolu. None ise varsayılan yol kullanılır.
    """

    def __init__(self, user_id: str, metadata_path: Optional[str] = None):
        if not AUDIO_LIBS_AVAILABLE:
            raise ImportError(
                "Ses işleme için 'soundfile', 'librosa' ve 'numpy' gereklidir.\n"
                "Yüklemek için: pip install soundfile librosa numpy"
            )

        self.user_id = user_id
        self.sr = config.ORNEKLEME_ORANI
        self.words_dir = Path(config.BASE_PATH) / user_id / "words"

        if metadata_path is None:
            metadata_path = str(Path(config.BASE_PATH) / user_id / "metadata_words.csv")

        self._word_index: dict[str, List[Tuple[str, float]]] = {}
        self._load_index(metadata_path)

    # ------------------------------------------------------------------
    # Index yükleme
    # ------------------------------------------------------------------

    def _load_index(self, metadata_path: str):
        """
        metadata_words.csv'den her kelime için (dosya_yolu, kalite_skoru)
        çiftlerini indeksler. Yalnızca mevcut ve kalite eşiğini geçen
        kayıtlar dahil edilir.
        """
        if not os.path.exists(metadata_path):
            print(f"Metadata bulunamadi: {metadata_path}")
            return

        min_quality = getattr(config, "SYNTHESIS_MIN_QUALITY", 50)

        with open(metadata_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row.get("transcription", "").strip().lower()
                if not word:
                    continue

                raw_path = row.get("file_path", "")
                file_path = self._resolve_path(raw_path)
                if not file_path or not os.path.exists(file_path):
                    continue

                try:
                    quality = float(row.get("quality_score", 0))
                except ValueError:
                    quality = 0.0

                if quality < min_quality:
                    continue

                if word not in self._word_index:
                    self._word_index[word] = []
                self._word_index[word].append((file_path, quality))

        for word in self._word_index:
            self._word_index[word].sort(key=lambda x: x[1], reverse=True)

        print(
            f"Sentez indeksi hazir: {len(self._word_index)} benzersiz kelime, "
            f"{sum(len(v) for v in self._word_index.values())} toplam kayit."
        )

    def _resolve_path(self, raw_path: str) -> Optional[str]:
        """
        CSV'den gelen herhangi bir yol bicimini (Windows ters slash, POSIX ileri slash,
        mutlak, goreli) mevcut platformda gecerli mutlak yola cevirir.
        """
        from src.utils.utils import resolve_audio_path
        return resolve_audio_path(raw_path, Path(config.BASE_PATH) / self.user_id)

    # ------------------------------------------------------------------
    # Kelime seçimi
    # ------------------------------------------------------------------

    def available_words(self) -> List[str]:
        """İndekste bulunan kelimelerin listesini döndürür."""
        return sorted(self._word_index.keys())

    def _pick_recording(self, word: str, strategy: str = "best") -> Optional[str]:
        """
        Bir kelime için kayıt seçer.

        strategy:
            "best"   – en yüksek kaliteli kayıt
            "random" – en iyi K kayıt arasından rastgele seçim (varyasyon için)
        """
        candidates = self._word_index.get(word.lower())
        if not candidates:
            return None

        best_k = getattr(config, "SYNTHESIS_BEST_K", 3)
        top = candidates[:best_k]

        if strategy == "random" and len(top) > 1:
            return random.choice(top)[0]
        return top[0][0]

    # ------------------------------------------------------------------
    # Ses işleme yardımcıları
    # ------------------------------------------------------------------

    def _load_audio(self, path: str) -> Optional[np.ndarray]:
        """Ses dosyasını yükler ve örnekleme oranını normalize eder."""
        try:
            audio, sr = librosa.load(path, sr=self.sr)
            audio = self._trim_silence(audio)
            return audio
        except Exception as e:
            print(f"Ses yuklenemedi ({path}): {e}")
            return None

    @staticmethod
    def _trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
        """Baş ve sondaki sessizliği kırpar."""
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed

    def _make_pause(self, duration_ms: int) -> np.ndarray:
        """Belirtilen sürede sessizlik dizisi üretir."""
        samples = int(self.sr * duration_ms / 1000)
        return np.zeros(samples, dtype=np.float32)

    @staticmethod
    def _crossfade(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
        """İki ses dizisini crossfade ile birleştirir."""
        if fade_samples <= 0 or len(a) < fade_samples or len(b) < fade_samples:
            return np.concatenate([a, b])

        fade_out = np.linspace(1.0, 0.0, fade_samples)
        fade_in = np.linspace(0.0, 1.0, fade_samples)

        result = np.concatenate([
            a[:-fade_samples],
            a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in,
            b[fade_samples:],
        ])
        return result

    # ------------------------------------------------------------------
    # Ana sentez API'si
    # ------------------------------------------------------------------

    def synthesize(
        self,
        words: List[str],
        output_path: Optional[str] = None,
        strategy: str = "best",
    ) -> Tuple[Optional[str], List[str]]:
        """
        Kelime listesinden sentezlenmiş ses dosyası üretir.

        Parameters
        ----------
        words : list[str]
            Sentezlenecek kelimeler sırası.
        output_path : str | None
            Çıktı WAV dosyası yolu. None ise geçici bir dosya oluşturulur.
        strategy : str
            "best" veya "random" (bkz. _pick_recording).

        Returns
        -------
        output_path : str | None
            Oluşturulan WAV dosyasının yolu. Hata durumunda None.
        missing_words : list[str]
            İndekste kaydı bulunmayan kelimeler.
        """
        if not words:
            return None, []

        crossfade_ms = getattr(config, "SYNTHESIS_CROSSFADE_MS", 30)
        pause_ms = getattr(config, "SYNTHESIS_PAUSE_MS", 150)
        fade_samples = int(self.sr * crossfade_ms / 1000)

        segments: List[np.ndarray] = []
        missing_words: List[str] = []
        pause_segment = self._make_pause(pause_ms)

        for i, word in enumerate(words):
            path = self._pick_recording(word, strategy=strategy)
            if path is None:
                print(f"Kelime bulunamadi: '{word}' – atlanıyor.")
                missing_words.append(word)
                if segments:
                    segments.append(pause_segment.copy())
                continue

            audio = self._load_audio(path)
            if audio is None or len(audio) == 0:
                missing_words.append(word)
                continue

            if segments:
                last = segments[-1]
                merged = self._crossfade(last, audio, fade_samples)
                segments[-1] = merged
                segments.append(pause_segment.copy())
            else:
                segments.append(audio)
                if i < len(words) - 1:
                    segments.append(pause_segment.copy())

        if not segments:
            print("Sentez icin gecerli ses bulunamadi.")
            return None, missing_words

        combined = np.concatenate(segments)
        combined = np.clip(combined, -1.0, 1.0)

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()

        sf.write(output_path, combined, self.sr, subtype="PCM_16")
        return output_path, missing_words

    def synthesize_sentence(
        self,
        sentence: str,
        output_path: Optional[str] = None,
        strategy: str = "best",
    ) -> Tuple[Optional[str], List[str]]:
        """
        Cümleyi boşluktan kelimelere bölerek synthesize() çağırır.

        Parameters
        ----------
        sentence : str
            Sentezlenecek Türkçe cümle.

        Returns
        -------
        Aynı synthesize() dönüş değerleri.
        """
        words = [w.strip().lower() for w in sentence.split() if w.strip()]
        return self.synthesize(words, output_path=output_path, strategy=strategy)

    def play(self, audio_path: str):
        """Sentezlenmiş ses dosyasını oynatır (çapraz platform)."""
        import platform
        import subprocess

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.run(["afplay", audio_path], check=True)
            elif system == "Linux":
                subprocess.run(["aplay", audio_path], check=True)
            elif system == "Windows":
                subprocess.run(
                    ["powershell", "-c", f'(New-Object Media.SoundPlayer "{audio_path}").PlaySync()'],
                    check=True,
                )
        except Exception as e:
            print(f"Ses oynatma hatasi: {e}")


if __name__ == "__main__":
    import sys

    user_id = sys.argv[1] if len(sys.argv) > 1 else "FurkanV1"
    sentence = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "merhaba nasılsın"

    print(f"Kullanici : {user_id}")
    print(f"Cumle     : {sentence}")

    synth = WordSynthesizer(user_id=user_id)
    print(f"Mevcut kelime sayisi: {len(synth.available_words())}")

    out_path, missing = synth.synthesize_sentence(sentence)
    if out_path:
        print(f"Sentez tamamlandi: {out_path}")
        if missing:
            print(f"Eksik kelimeler  : {missing}")
        synth.play(out_path)
    else:
        print("Sentez basarisiz.")
