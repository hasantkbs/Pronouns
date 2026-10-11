# -*- coding: utf-8 -*-
"""
Sozluk korumali cumle duzeltici.

ASR'in ham metnini alir; genel Turkce sozlukte (veya kullanicinin sozlugunde)
bulunan kelimelere dokunmaz, sadece taninmayan (bozuk) kelimeleri en yakin
adaya duzeltir ve cumleyi bicimlendirir (buyuk harf + nokta/soru isareti).
"""
import logging
import re
import string
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from Levenshtein import distance as lev_dist

import config
from src.utils.utils import normalize_turkish_text

logger = logging.getLogger(__name__)

# Turkceye ozgu harfleri ASCII karsiligina katlar ("aldım" -> "aldim")
_TR_FOLD = str.maketrans("ışğçöü", "isgcou")

_QUESTION_PARTICLE = re.compile(
    r"^(mı|mi|mu|mü)"
    r"(yım|yim|yum|yüm|sın|sin|sun|sün|yız|yiz|yuz|yüz|"
    r"sınız|siniz|sunuz|sünüz|ydı|ydi|ydu|ydü|ymış|ymiş|ymuş|ymüş)?$"
)
_QUESTION_WORDS = {
    "ne", "nerede", "nereye", "nereden", "kim", "kime", "kimi",
    "nasıl", "neden", "niye", "niçin", "hangi", "kaç",
}

# Kesme isareti disindaki noktalama: "İstanbul'a," -> "İstanbul'a"
_STRIP_CHARS = string.punctuation.replace("'", "") + "“”"


def _read_lexicon(lexicon_path: Path) -> List[Tuple[str, int]]:
    """'kelime frekans' satirlarini okur (dosya siklik sirasina gore azalan)."""
    rows = []
    with open(lexicon_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                rows.append((parts[0], int(parts[1])))
    return rows


def _turkish_capitalize(text: str) -> str:
    first = {"i": "İ", "ı": "I"}.get(text[0], text[0].upper())
    return first + text[1:]


class SentenceCorrector:
    def __init__(self, user_vocabulary: Iterable[str], lexicon_path: Path):
        # Cok kelimeli girdiler ("geri bildirim") kelimelerine bolunur
        self.user_words = sorted({
            w for entry in user_vocabulary for w in normalize_turkish_text(entry).split()
        })
        self.known = set(self.user_words)
        self.candidates: List[str] = []
        self.rank = {}
        self.lexicon_loaded = False

        try:
            rows = _read_lexicon(lexicon_path)
        except (OSError, ValueError) as e:
            # Sozluk yoksa kelime duzeltmesi tamamen atlanir: yalnizca kullanici
            # sozlugune zorla yaklastirmak cumledeki ekleri bozar.
            logger.warning("Sozluk yuklenemedi (%s): %s - kelime duzeltmesi kapali", lexicon_path, e)
            return

        freq = dict(rows)
        most_frequent_spelling = {}
        for word, count in rows:
            key = word.translate(_TR_FOLD)
            best = most_frequent_spelling.get(key)
            if best is None or count > freq[best]:
                most_frequent_spelling[key] = word

        self.known.update(freq)
        # ASCII varyanti ("aldim"), daha sik Turkce yazimi ("aldım") varsa aday olamaz;
        # ama taninir, yani "su"/"kir" gibi gecerli kelimelere dokunulmaz.
        self.candidates = [
            word for word, _ in rows
            if not (word == word.translate(_TR_FOLD) and most_frequent_spelling[word] != word)
        ]
        self.rank = {word: i for i, word in enumerate(self.candidates)}
        self.lexicon_loaded = True

    def correct(self, raw_text: Optional[str]) -> str:
        if not raw_text:
            return ""

        words = []
        for raw_token in raw_text.replace("’", "'").split():
            if "'" in raw_token:
                # Ozel isim + ek ("İstanbul'a"): oldugu gibi korunur
                token = raw_token.strip(_STRIP_CHARS)
                if token.strip("'"):
                    words.append(token)
                continue
            for word in normalize_turkish_text(raw_token).split():
                words.append(self._correct_word(word))

        if not words:
            return ""

        is_question = any(
            _QUESTION_PARTICLE.match(w) or w in _QUESTION_WORDS
            for w in (normalize_turkish_text(w) for w in words)
        )
        return _turkish_capitalize(" ".join(words)) + ("?" if is_question else ".")

    def _correct_word(self, word: str) -> str:
        if not self.lexicon_loaded or word in self.known or len(word) <= 2:
            return word

        limit = (1 - config.FUZZY_MATCH_THRESHOLD) * len(word) + 1e-9
        from_user = self._best(word, self.user_words, limit)
        from_lexicon = self._best(word, self.candidates, limit)

        if from_user and (from_lexicon is None or from_user[0] <= from_lexicon[0]):
            return from_user[2]
        if from_lexicon:
            return from_lexicon[2]
        return word

    def _best(self, word: str, words: List[str], limit: float):
        """Esik icindeki en iyi aday: (mesafe, siklik_sirasi, kelime) ya da None."""
        best = None
        for other in words:
            if abs(len(other) - len(word)) > limit:
                continue
            d = lev_dist(word, other)
            if d > limit:
                continue
            key = (d, self.rank.get(other, len(self.rank)), other)
            if best is None or key < best:
                best = key
        return best
