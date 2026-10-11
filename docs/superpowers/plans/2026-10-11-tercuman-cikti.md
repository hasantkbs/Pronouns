# Tercüman Çıktısı (Düzeltilmiş Cümle + Piper TTS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/translate`, Furkan'ın söylediği cümlenin sözlük korumalı düzeltilmiş halini (`response_text`) döndürsün ve yerel Piper TTS ile seslendirsin; anahtar kelime asistanı ve kelime-birleştirme sentezi bu akıştan çıksın.

**Architecture:** İki yeni, bağımsız birim: `SentenceCorrector` (sözlük → düzeltilmiş + biçimlendirilmiş cümle) ve `PiperTTS` (cümle → WAV, hata durumunda `None`). `ASRSystem.transcribe` eski Furkan-sözlüğüne-zorla düzeltmeyi bir parametreyle kapatabilir hale gelir; `/translate` bu üçünü sırayla bağlar. İstemci değişmez.

**Tech Stack:** Python 3.12, FastAPI, `python-Levenshtein` (mevcut), `piper-tts==1.8.0` (yeni; `onnxruntime` CPU), pytest. Venv: `/home/mayasoft/app/venv`.

**Spec:** `docs/superpowers/specs/2026-10-11-tercuman-cikti-design.md`

## Global Constraints

- Dal: `tercuman-cikti` (zaten var, `whisper-model-selection` üstünde; spec commit'i `971bbd98`). Repo kökü: `/home/mayasoft/app/Pronouns`.
- Python komutları her zaman venv ile: `/home/mayasoft/app/venv/bin/python`, pip: `/home/mayasoft/app/venv/bin/pip`.
- Test komutu (tüm suite): `/home/mayasoft/app/venv/bin/python -m pytest tests/ --ignore=tests/test_utils.py -q -p no:cacheprovider` — `tests/test_utils.py` değişikliklerden önce de toplanamıyor (`ModuleNotFoundError: utils`), kapsam dışı.
- Dışarıya kullanıcı verisi gönderilmez; tek dış erişim iki statik dosya indirmesi (sözlük, Piper sesi).
- Düzeltme eşiği: `mesafe ≤ (1 - config.FUZZY_MATCH_THRESHOLD) × len(kelime)` (`FUZZY_MATCH_THRESHOLD = 0.6`, değiştirilmez).
- ≤2 harfli sözlük dışı kelimeye dokunulmaz.
- Sözlük dosyası: `data/lexicon/tr_50k.txt` (hermitdave/FrequencyWords 2018 tr_50k, sha256 `480192844e7fdfe9591bbf0b9cfbb96ed5751cfbba7024fd45d8cb7508e07d5c`). `data/` `.gitignore`'da → `git add -f` ile eklenir.
- Piper sesi: `data/models/tts/tr_TR-dfki-medium.onnx` (sha256 `2844717f524ab965d3fe86e60562cbb601d3e456836efcc2196cc3a14112a8fb`) + `.onnx.json` (sha256 `13ebd7810f1b61b5027583cf3131a0a233b6ea81c38f2200ebc4ff41c3cca039`); repoya **eklenmez**.
- `/translate` yanıt alanları: `recognized_text` (ham ASR), `response_text` (düzeltilmiş), `audio_url` (`/download/<dosya>` | `null`), `intent` = `"translate"`, `missing_words` = `[]`.
- `pronouns-api.service` yeniden başlatma **yalnızca kullanıcı** tarafından (`sudo` gerekiyor; ajan yapamaz, yapmaya çalışmaz).
- Commit mesajları şu satırla biter: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **Özel isim + kesme işareti** (`İstanbul'a`, Whisper `’` ile de yazabilir) — normalize kesmeyi silip `istanbula` yapar ve kelimeyi "düzeltilebilir" kılar; kişi özel ismin olduğu gibi kalmasını bekler. → Task 1 `test_proper_noun_with_apostrophe_is_kept`.
2. **Whisper biçimli girdi** (büyük harf, nokta, virgül: `"Ben bugün okula geldim."`) — çift noktalama ya da `İ`/`I` hatası olmamalı. → Task 1 `test_whisper_style_input_is_normalised`.
3. **Sadece noktalama / boş ASR çıktısı** (`" ... !"`) — boş cümle seslendirilmemeli, "Konuşma anlaşılamadı." dönmeli. → Task 1 `test_empty_or_punctuation_only_gives_empty` + Task 4 `test_translate_empty_correction_is_not_understood`.
4. **Rakamlar** (`"3 bir"`) — rakam korunmalı, bir kelimeye "düzeltilmemeli". → Task 1 `test_digits_are_kept`.
5. **Piper modeli diskte yok / piper kurulu değil** (indirme scripti çalıştırılmadan API başlatıldı) — API açılmalı, metin dönmeli, `audio_url: null`. → Task 2 `test_missing_voice_returns_none` + Task 4 `test_translate_without_audio_when_tts_fails`.

---

## File Structure

| Dosya | Sorumluluk |
|---|---|
| Create `src/core/sentence_corrector.py` | `SentenceCorrector`: sözlük yükleme, kelime düzeltme, cümle biçimlendirme |
| Create `data/lexicon/tr_50k.txt` (force-add) | Genel Türkçe sıklık sözlüğü (statik veri) |
| Create `tests/test_sentence_corrector.py` | Düzeltici birim + gerçek-sözlük regresyon testleri |
| Create `src/core/tts.py` | `PiperTTS`: cümle → WAV, hata yutan sarmalayıcı |
| Create `scripts/download_piper_voice.sh` | Piper Türkçe sesini bir kez indirir |
| Create `tests/test_tts.py` | TTS testleri |
| Create `tests/test_asr_fuzzy_flag.py` | `transcribe(fuzzy_correct=...)` testleri |
| Modify `src/core/asr.py:228,278` | `fuzzy_correct` parametresi |
| Modify `api.py` (importlar `:17-20`, globaller `:71-74`, `/translate` `:385-409`) | Akışı bağlama, `get_corrector`, `tts` |
| Modify `tests/test_translate_speaker_verification.py` | `_FakeAsr` imzası + yeni `/translate` testleri |
| Modify `config.py:81` civarı | `LEXICON_PATH`, `PIPER_VOICE_PATH` |
| Modify `requirements.txt` | `piper-tts==1.8.0` |

---

### Task 1: Sözlük ve `SentenceCorrector`

**Files:**
- Create: `data/lexicon/tr_50k.txt`
- Create: `src/core/sentence_corrector.py`
- Create: `tests/test_sentence_corrector.py`
- Modify: `config.py` (`FUZZY_MATCH_THRESHOLD = 0.6` satırının hemen altı)

**Interfaces:**
- Consumes: `src.utils.utils.normalize_turkish_text(text: str) -> str` (mevcut), `config.FUZZY_MATCH_THRESHOLD` (mevcut).
- Produces: `config.LEXICON_PATH: str = "data/lexicon/tr_50k.txt"`; `SentenceCorrector(user_vocabulary: Iterable[str], lexicon_path: Path)`; `SentenceCorrector.correct(raw_text: Optional[str]) -> str` (boş/sadece noktalama → `""`; aksi halde ilk harfi büyük, `.` ya da `?` ile biten cümle). Task 4 bunları kullanır.

- [ ] **Step 1: Sözlük dosyasını indir ve doğrula**

```bash
cd /home/mayasoft/app/Pronouns
mkdir -p data/lexicon
curl -fsSL --retry 3 -o data/lexicon/tr_50k.txt \
  https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/tr/tr_50k.txt
sha256sum data/lexicon/tr_50k.txt
head -3 data/lexicon/tr_50k.txt; wc -l data/lexicon/tr_50k.txt
```

Expected: sha256 `480192844e7fdfe9591bbf0b9cfbb96ed5751cfbba7024fd45d8cb7508e07d5c`; ilk satırlar `bir 6034903`, `bu 3529532`, `ne 2550646`; 50000 satır. Hash farklıysa DUR (upstream dosya değişmiş) ve bildir.

- [ ] **Step 2: Config sabitini ekle**

`config.py`'de `FUZZY_MATCH_THRESHOLD = 0.6` satırının hemen altına:

```python
LEXICON_PATH = "data/lexicon/tr_50k.txt"  # Genel Turkce siklik sozlugu (cumle duzeltici)
```

- [ ] **Step 3: Failing testleri yaz**

`tests/test_sentence_corrector.py`:

```python
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.sentence_corrector import SentenceCorrector

REAL_LEXICON = ROOT / "data" / "lexicon" / "tr_50k.txt"

# Siklik sirasina gore azalan, gercek dosyayla ayni bicim
_SMALL_LEXICON = """\
bir 1000
ben 990
şu 950
geldim 900
iyi 880
aldım 800
aldim 700
su 600
bugün 500
okula 400
yaram 300
kır 200
kir 100
"""


@pytest.fixture
def lexicon(tmp_path):
    path = tmp_path / "lexicon.txt"
    path.write_text(_SMALL_LEXICON, encoding="utf-8")
    return path


def _corrector(lexicon, vocabulary=("yardım", "kampanya", "geri bildirim")):
    return SentenceCorrector(list(vocabulary), lexicon)


def test_known_words_are_kept(lexicon):
    assert _corrector(lexicon).correct("ben bugün okula geldim") == "Ben bugün okula geldim."


def test_unknown_word_is_corrected_to_nearest(lexicon):
    assert _corrector(lexicon).correct("gldim") == "Geldim."


def test_ascii_variant_is_never_a_correction(lexicon):
    # "aldm" -> "aldım" ve "aldim" ikisine de 1 uzakta; ASCII varyanti aday olamaz
    assert _corrector(lexicon).correct("aldm") == "Aldım."


def test_ascii_variant_is_still_recognised(lexicon):
    # "su" ve "kir" gecerli kelimeler: daha sik "şu"/"kır" olsa da dokunulmaz
    assert _corrector(lexicon).correct("su kir") == "Su kir."


def test_user_vocabulary_wins_a_tie(lexicon):
    # "yardm": "yardım" (Furkan) ve "yaram" (genel) ikisi de 1 uzakta
    assert _corrector(lexicon).correct("yardm") == "Yardım."


def test_word_without_close_candidate_is_kept(lexicon):
    assert _corrector(lexicon).correct("xqzvw") == "Xqzvw."


def test_short_unknown_word_is_kept(lexicon):
    assert _corrector(lexicon).correct("zq bir") == "Zq bir."


def test_multi_word_vocabulary_entry_is_known(lexicon):
    assert _corrector(lexicon).correct("geri bildirim") == "Geri bildirim."


def test_question_particle_gets_question_mark(lexicon):
    assert _corrector(lexicon).correct("iyi misin") == "İyi misin?"


def test_question_word_gets_question_mark(lexicon):
    assert _corrector(lexicon).correct("ne aldım") == "Ne aldım?"


def test_dotted_capital_i(lexicon):
    assert _corrector(lexicon).correct("iyi") == "İyi."


def test_whisper_style_input_is_normalised(lexicon):
    assert _corrector(lexicon).correct("Ben bugün okula geldim.") == "Ben bugün okula geldim."


def test_proper_noun_with_apostrophe_is_kept(lexicon):
    assert _corrector(lexicon).correct("ben istanbul'a geldim,") == "Ben istanbul'a geldim."
    assert _corrector(lexicon).correct("İstanbul’a geldim") == "İstanbul'a geldim."


def test_digits_are_kept(lexicon):
    assert _corrector(lexicon).correct("3 bir") == "3 bir."


def test_empty_or_punctuation_only_gives_empty(lexicon):
    corrector = _corrector(lexicon)
    assert corrector.correct("") == ""
    assert corrector.correct(None) == ""
    assert corrector.correct(" ... !") == ""


def test_missing_lexicon_disables_word_correction(tmp_path):
    corrector = SentenceCorrector(["okul"], tmp_path / "yok.txt")
    # Yalnizca Furkan sozlugune zorla yaklastirma yapilmaz ("okula" -> "okul" olmaz)
    assert corrector.correct("okula gldim") == "Okula gldim."


@pytest.mark.skipif(not REAL_LEXICON.exists(), reason="data/lexicon/tr_50k.txt yok")
@pytest.mark.parametrize("sentence, expected", [
    ("ben bugün okula geldim", "Ben bugün okula geldim."),
    ("yarın arkadaşımla sinemaya gideceğiz", "Yarın arkadaşımla sinemaya gideceğiz."),
    ("su istiyorum", "Su istiyorum."),
    ("gldim", "Geldim."),
    ("kampnya", "Kampanya."),
    ("tşekkür ederim", "Teşekkür ederim."),
    ("su kir isim", "Su kir isim."),
])
def test_real_lexicon_regressions(sentence, expected):
    corrector = SentenceCorrector(["okul", "geldi", "istiyor", "yakın", "arkadaş", "kampanya"], REAL_LEXICON)
    assert corrector.correct(sentence) == expected
```

- [ ] **Step 4: Testi çalıştır, başarısız olduğunu gör**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_sentence_corrector.py -q -p no:cacheprovider`
Expected: koleksiyon hatası — `ModuleNotFoundError: No module named 'src.core.sentence_corrector'`.

- [ ] **Step 5: `SentenceCorrector`'ı yaz**

`src/core/sentence_corrector.py`:

```python
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
```

- [ ] **Step 6: Testleri çalıştır, geçtiğini gör**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_sentence_corrector.py -q -p no:cacheprovider`
Expected: `23 passed` (bu kod ve testler plan yazılırken izole bir kopyada çalıştırıldı: 23/23).

- [ ] **Step 7: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add -f data/lexicon/tr_50k.txt
git add config.py src/core/sentence_corrector.py tests/test_sentence_corrector.py
git commit -m "feat: add lexicon-guarded SentenceCorrector

Only words missing from a 50k Turkish frequency lexicon (plus the user's
vocabulary) are fuzzy-corrected, so inflected words like 'geldim' and
'arkadaşımla' survive. ASCII spellings ('aldim') are recognised but never
offered as corrections; ties go to the more frequent word, and the user's
vocabulary wins equal-distance ties. Formats the sentence (Turkish
capital İ, '.' or '?').

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `PiperTTS` ve ses modeli

**Files:**
- Create: `scripts/download_piper_voice.sh`
- Create: `src/core/tts.py`
- Create: `tests/test_tts.py`
- Modify: `config.py` (Task 1'de eklenen `LEXICON_PATH` satırının altı)
- Modify: `requirements.txt` (alfabetik konuma `piper-tts==1.8.0`)

**Interfaces:**
- Consumes: yok.
- Produces: `config.PIPER_VOICE_PATH: str = "data/models/tts/tr_TR-dfki-medium.onnx"`; `PiperTTS(voice_path: Path)` (yükleme hatasını yutar); `PiperTTS.synthesize(text: str, output_path: str) -> Optional[str]` (başarıda `output_path`, her hatada ya da boş metinde `None`; istisna fırlatmaz). Task 4 bunları kullanır.

- [ ] **Step 1: `piper-tts`'i kur**

```bash
cd /home/mayasoft/app/Pronouns
/home/mayasoft/app/venv/bin/pip install piper-tts==1.8.0 2>&1 | tail -1
```

Expected: `Successfully installed onnxruntime-1.31.0 pathvalidate-3.3.1 piper-tts-1.8.0` (dry-run'da başka paket değişmiyordu).

`requirements.txt`'te alfabetik sıraya (ör. `pillow`/`pip...` civarı, `p` harfi bölümüne) şu satırı ekle:

```
piper-tts==1.8.0
```

- [ ] **Step 2: İndirme scriptini yaz ve çalıştır**

`scripts/download_piper_voice.sh`:

```bash
#!/usr/bin/env bash
# Piper Turkce sesini (tr_TR-dfki-medium, ~60 MB) bir kez indirir.
# Kullanim: bash scripts/download_piper_voice.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=data/models/tts
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium
mkdir -p "$DEST"

for f in tr_TR-dfki-medium.onnx tr_TR-dfki-medium.onnx.json; do
  if [ -s "$DEST/$f" ]; then
    echo "zaten var: $DEST/$f"
    continue
  fi
  curl -fSL --retry 3 -o "$DEST/$f.part" "$BASE/$f"
  mv "$DEST/$f.part" "$DEST/$f"
  echo "indirildi: $DEST/$f"
done
```

```bash
chmod +x scripts/download_piper_voice.sh
bash scripts/download_piper_voice.sh
sha256sum data/models/tts/tr_TR-dfki-medium.onnx data/models/tts/tr_TR-dfki-medium.onnx.json
```

Expected: hash'ler sırasıyla `2844717f524ab965d3fe86e60562cbb601d3e456836efcc2196cc3a14112a8fb` ve `13ebd7810f1b61b5027583cf3131a0a233b6ea81c38f2200ebc4ff41c3cca039`. Farklıysa DUR ve bildir.

- [ ] **Step 3: Config sabitini ekle**

`config.py`'de `LEXICON_PATH` satırının altına:

```python
PIPER_VOICE_PATH = "data/models/tts/tr_TR-dfki-medium.onnx"  # scripts/download_piper_voice.sh indirir
```

- [ ] **Step 4: Failing testleri yaz**

`tests/test_tts.py`:

```python
# -*- coding: utf-8 -*-
import os
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.core.tts import PiperTTS

REAL_VOICE = ROOT / config.PIPER_VOICE_PATH


@pytest.mark.skipif(not REAL_VOICE.exists(), reason="Piper sesi indirilmemis")
def test_synthesizes_valid_wav(tmp_path):
    out = str(tmp_path / "out.wav")
    assert PiperTTS(REAL_VOICE).synthesize("Ben bugün okula geldim.", out) == out
    with wave.open(out, "rb") as wav:
        assert wav.getframerate() == 22050
        assert wav.getnchannels() == 1
        assert wav.getnframes() > 0


def test_missing_voice_returns_none(tmp_path):
    tts = PiperTTS(tmp_path / "yok.onnx")
    out = tmp_path / "out.wav"
    assert tts.synthesize("Merhaba.", str(out)) is None
    assert not out.exists()


@pytest.mark.skipif(not REAL_VOICE.exists(), reason="Piper sesi indirilmemis")
def test_empty_text_returns_none(tmp_path):
    assert PiperTTS(REAL_VOICE).synthesize("   ", str(tmp_path / "out.wav")) is None


def test_synthesis_error_returns_none_and_cleans_up(tmp_path):
    class _BrokenVoice:
        def synthesize_wav(self, text, wav_file):
            raise RuntimeError("bozuk model")

    tts = PiperTTS(tmp_path / "yok.onnx")
    tts.voice = _BrokenVoice()
    out = tmp_path / "out.wav"
    assert tts.synthesize("Merhaba.", str(out)) is None
    assert not out.exists()
```

- [ ] **Step 5: Testi çalıştır, başarısız olduğunu gör**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_tts.py -q -p no:cacheprovider`
Expected: koleksiyon hatası — `ModuleNotFoundError: No module named 'src.core.tts'`.

- [ ] **Step 6: `PiperTTS`'i yaz**

`src/core/tts.py`:

```python
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
```

- [ ] **Step 7: Testleri çalıştır, geçtiğini gör**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_tts.py -q -p no:cacheprovider`
Expected: `4 passed` (ses Step 2'de indirildiği için skip yok).

- [ ] **Step 8: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add scripts/download_piper_voice.sh src/core/tts.py tests/test_tts.py config.py requirements.txt
git commit -m "feat: add local Piper TTS wrapper and Turkish voice download script

PiperTTS loads tr_TR-dfki-medium on CPU and never raises: a missing voice
or a synthesis error is logged and returns None so callers can still
return text.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `ASRSystem.transcribe(fuzzy_correct=...)`

**Files:**
- Create: `tests/test_asr_fuzzy_flag.py`
- Modify: `src/core/asr.py:228` (imza), `src/core/asr.py:276-278` (düzeltme çağrısı)

**Interfaces:**
- Consumes: yok.
- Produces: `ASRSystem.transcribe(audio_path, fuzzy_correct: bool = True) -> tuple[Optional[str], float]`; `fuzzy_correct=False` iken metin ASR'ın ham çıktısıdır (yalnızca `_fuzzy_correct` atlanır). Task 4 `fuzzy_correct=False` ile çağırır. Diğer çağıranlar (`evaluate_fuzzy.py`, `src/utils/utils.py:492`, `src/cli/*`) değişmez.

- [ ] **Step 1: Failing testleri yaz**

`tests/test_asr_fuzzy_flag.py`:

```python
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
```

- [ ] **Step 2: Testi çalıştır, başarısız olduğunu gör**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_asr_fuzzy_flag.py -q -p no:cacheprovider`
Expected: `1 failed, 1 passed` — `TypeError: ASRSystem.transcribe() got an unexpected keyword argument 'fuzzy_correct'` (varsayılan-davranış testi bugünkü kodla zaten geçer; bu doğrulandı).

- [ ] **Step 3: Parametreyi ekle**

`src/core/asr.py`'de imzayı değiştir:

```python
    def transcribe(self, audio_path):
```
→
```python
    def transcribe(self, audio_path, fuzzy_correct=True):
```

ve aynı fonksiyonun sonundaki:

```python
            # --- Kelime duzeltme ve anlamli cumle kurma ---
            text = self._fuzzy_correct(text)
```
→
```python
            # --- Kelime duzeltme ve anlamli cumle kurma ---
            # /translate bunu kapatir: cumle duzeltmeyi SentenceCorrector yapar
            if fuzzy_correct:
                text = self._fuzzy_correct(text)
```

- [ ] **Step 4: Testleri çalıştır, geçtiğini gör**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_asr_fuzzy_flag.py -q -p no:cacheprovider`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add src/core/asr.py tests/test_asr_fuzzy_flag.py
git commit -m "feat: let ASRSystem.transcribe skip vocabulary fuzzy correction

Default stays True so evaluation scripts are unchanged; /translate will
pass False and correct sentences with SentenceCorrector instead.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `/translate`'i tercüman akışına bağla

**Files:**
- Modify: `api.py` (importlar `:17-20`, globaller `:71-74`, `/translate` gövdesi `:385-409`)
- Modify: `tests/test_translate_speaker_verification.py`

**Interfaces:**
- Consumes: Task 1 `SentenceCorrector(user_vocabulary, lexicon_path).correct(raw_text) -> str`, `config.LEXICON_PATH`; Task 2 `PiperTTS(voice_path).synthesize(text, output_path) -> Optional[str]`, `config.PIPER_VOICE_PATH`; Task 3 `asr.transcribe(path, fuzzy_correct=False)`; mevcut `asr.user_vocabulary: list[str]`.
- Produces: `api.get_corrector(user_id: str, vocabulary: list[str]) -> SentenceCorrector` (kullanıcı başına önbellek `api.correctors`); `api.tts: PiperTTS` (modül seviyesi tekil); `/translate` yanıtı Global Constraints'teki sözleşme.

- [ ] **Step 1: Mevcut sahte ASR'ı yeni imzaya uyarla ve failing testleri yaz**

`tests/test_translate_speaker_verification.py`'de `_FakeAsr`'ı:

```python
class _FakeAsr:
    def transcribe(self, path):
        return None, 0.0
```
→
```python
class _FakeAsr:
    user_vocabulary = []

    def transcribe(self, path, fuzzy_correct=True):
        return None, 0.0
```

Dosyanın sonuna ekle:

```python
class _FakeTextAsr:
    user_vocabulary = ["okul"]

    def __init__(self, text="ben okla gldim"):
        self.text = text
        self.fuzzy_flags = []

    def transcribe(self, path, fuzzy_correct=True):
        self.fuzzy_flags.append(fuzzy_correct)
        return self.text, 0.85


class _FakeCorrector:
    def __init__(self, result="Ben okula geldim."):
        self.result = result
        self.inputs = []

    def correct(self, text):
        self.inputs.append(text)
        return self.result


class _FakeTts:
    def __init__(self, ok=True):
        self.ok = ok
        self.texts = []

    def synthesize(self, text, output_path):
        self.texts.append(text)
        return output_path if self.ok else None


def _post_translate():
    return client.post(
        "/translate",
        data={"user_id": "FurkanV1"},
        files={"audio": _fake_audio_file()},
        headers=_AUTH,
    )


def _patch_pipeline(monkeypatch, asr, corrector, tts):
    monkeypatch.setattr(config, "SPEAKER_VERIFY_ENABLED", False)
    monkeypatch.setattr(api, "get_asr", lambda user_id: asr)
    monkeypatch.setattr(api, "get_corrector", lambda user_id, vocabulary: corrector)
    monkeypatch.setattr(api, "tts", tts)


def test_translate_returns_corrected_sentence(monkeypatch):
    asr, corrector, tts = _FakeTextAsr(), _FakeCorrector(), _FakeTts()
    _patch_pipeline(monkeypatch, asr, corrector, tts)

    body = _post_translate().json()

    assert body["recognized_text"] == "ben okla gldim"
    assert body["response_text"] == "Ben okula geldim."
    assert body["intent"] == "translate"
    assert body["missing_words"] == []
    assert body["audio_url"].startswith("/download/")
    assert asr.fuzzy_flags == [False]
    assert corrector.inputs == ["ben okla gldim"]
    assert tts.texts == ["Ben okula geldim."]


def test_translate_without_audio_when_tts_fails(monkeypatch):
    _patch_pipeline(monkeypatch, _FakeTextAsr(), _FakeCorrector(), _FakeTts(ok=False))

    body = _post_translate().json()

    assert body["response_text"] == "Ben okula geldim."
    assert body["audio_url"] is None


def test_translate_empty_correction_is_not_understood(monkeypatch):
    tts = _FakeTts()
    _patch_pipeline(monkeypatch, _FakeTextAsr(text="..."), _FakeCorrector(result=""), tts)

    assert _post_translate().json() == {"error": "Konuşma anlaşılamadı."}
    assert tts.texts == []
```

- [ ] **Step 2: Testi çalıştır, başarısız olduğunu gör**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_translate_speaker_verification.py -q -p no:cacheprovider`
Expected: üç yeni test FAIL — `AttributeError: <module 'api' ...> has no attribute 'get_corrector'` (özellik yok); mevcut 4 test PASS (`_FakeAsr` yeni imzaya rağmen eski davranışta).

- [ ] **Step 3: `api.py` importlarını değiştir**

```python
from src.core.asr import ASRSystem
from src.core.synthesizer import WordSynthesizer
from src.core.nlu import NLU_System
from src.core.actions import run_action
```
→
```python
from src.core.asr import ASRSystem
from src.core.sentence_corrector import SentenceCorrector
from src.core.tts import PiperTTS
```

(`WordSynthesizer`, `NLU_System`, `run_action` `api.py`'de yalnızca `/translate`'te kullanılıyordu — `grep -n "nlu\|WordSynthesizer\|run_action" api.py` ile doğrula; başka kullanım varsa importu koru ve ledger'a yaz.)

- [ ] **Step 4: Globalleri değiştir**

```python
asr_systems = {}  # user_id -> ASRSystem
nlu = NLU_System()
training_services = {} # user_id -> TrainingService
```
→
```python
asr_systems = {}  # user_id -> ASRSystem
correctors: Dict[str, SentenceCorrector] = {}  # user_id -> SentenceCorrector
tts = PiperTTS(Path(config.PIPER_VOICE_PATH))  # CPU, API acilisinda bir kez
training_services = {} # user_id -> TrainingService
```

ve `def get_asr(...)` fonksiyonunun hemen altına:

```python
def get_corrector(user_id: str, vocabulary: List[str]) -> SentenceCorrector:
    if user_id not in correctors:
        correctors[user_id] = SentenceCorrector(vocabulary, Path(config.LEXICON_PATH))
    return correctors[user_id]
```

- [ ] **Step 5: `/translate` gövdesini değiştir**

`# ASR` yorumundan `return {...}` sonuna kadar olan bölüm:

```python
    # ASR
    asr = get_asr(user_id)
    text, confidence = asr.transcribe(temp_input)
    os.remove(temp_input)
    
    if not text:
        return {"error": "Konuşma anlaşılamadı."}
    
    # NLU & Action
    intent, entities = nlu.process_text(text)
    action_response = run_action(intent, entities)
    
    # Synthesis
    synth = WordSynthesizer(user_id=user_id)
    out_filename = f"{uuid.uuid4()}.wav"
    out_path = str(_TTS_OUTPUT_DIR / out_filename)
    out_path, missing = synth.synthesize_sentence(action_response, output_path=out_path)
    
    return {
        "recognized_text": text,
        "intent": intent,
        "response_text": action_response,
        "audio_url": f"/download/{os.path.basename(out_path)}" if out_path else None,
        "missing_words": missing
    }
```
→
```python
    # ASR (ham metin: cumle duzeltmeyi SentenceCorrector yapar)
    asr = get_asr(user_id)
    text, confidence = asr.transcribe(temp_input, fuzzy_correct=False)
    os.remove(temp_input)

    if not text:
        return {"error": "Konuşma anlaşılamadı."}

    # Tercuman: Furkan'in cumlesinin duzeltilmis hali
    corrected = get_corrector(user_id, asr.user_vocabulary).correct(text)
    if not corrected:
        return {"error": "Konuşma anlaşılamadı."}

    # Yerel TTS (Piper); basarisizsa metin yine doner
    out_path = tts.synthesize(corrected, str(_TTS_OUTPUT_DIR / f"{uuid.uuid4()}.wav"))

    return {
        "recognized_text": text,
        "intent": "translate",
        "response_text": corrected,
        "audio_url": f"/download/{os.path.basename(out_path)}" if out_path else None,
        "missing_words": [],
    }
```

- [ ] **Step 6: Testleri çalıştır, geçtiğini gör**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_translate_speaker_verification.py -q -p no:cacheprovider`
Expected: `7 passed`.

- [ ] **Step 7: Tüm suite**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/ --ignore=tests/test_utils.py -q -p no:cacheprovider`
Expected: hepsi geçer — önceki 26 + Task 1 (23) + Task 2 (4) + Task 3 (2) + Task 4 (3) = `58 passed`.

- [ ] **Step 8: Commit**

```bash
cd /home/mayasoft/app/Pronouns
git add api.py tests/test_translate_speaker_verification.py
git commit -m "feat: make /translate an interpreter (corrected sentence + Piper)

response_text is now the user's own sentence, corrected by
SentenceCorrector, instead of the keyword assistant's reply; audio comes
from local Piper TTS instead of concatenated word recordings. ASR runs
without the vocabulary-only fuzzy pass that mangled inflected words.
Response fields are kept for compatibility (intent='translate',
missing_words=[]); a TTS failure returns text with audio_url null.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Devreye alma ve canlı doğrulama (insan onayı gerekli)

**Files:** yok (yalnızca doğrulama).

**Interfaces:**
- Consumes: Task 1-4'ün tamamı; çalışan `pronouns-api.service` (port 8001).
- Produces: canlıda çalışan tercüman akışının kanıtı.

- [ ] **Step 1: Kullanıcıya bildir ve DUR**

Ajan API'yi yeniden başlatamaz (`sudo`). Kullanıcıya şunu iste ve cevabı bekle:

```
! sudo systemctl restart pronouns-api.service
```

(Canlı dizin `tercuman-cikti` dalında olduğu için yeniden başlatma bu dalın kodunu yükler. Merge kararı kullanıcınındır.)

- [ ] **Step 2: Açılışı doğrula**

```bash
systemctl is-active pronouns-api.service
journalctl -u pronouns-api.service --since "5 min ago" --no-pager | grep -iE "Piper|Sozluk|Error|Traceback|Application startup complete" | tail -5
```

Expected: `active`; `Application startup complete.`; `Piper sesi yuklenemedi` ya da `Sozluk yuklenemedi` uyarısı **yok**.

- [ ] **Step 3: Gerçek kayıtlarla `/translate`**

```bash
cd /home/mayasoft/app/Pronouns
KEY=$(/home/mayasoft/app/venv/bin/python -c "import config; print(config.API_SECRET_KEY)")
for f in $(/home/mayasoft/app/venv/bin/python -c "
import pandas as pd; print(' '.join(pd.read_csv('data/users/FurkanV1/eval.csv').sample(5, random_state=3).file_path))"); do
  curl -sS -H "X-API-Key: $KEY" -F user_id=FurkanV1 -F "audio=@$f" http://127.0.0.1:8001/translate; echo
done
```

Expected: her yanıtta `"intent":"translate"`, `response_text` büyük harfle başlayıp `.`/`?` ile biten bir cümle, `audio_url` `/download/...` (ya da konuşmacı doğrulama reddiyse `{"rejected":true,...}` — o kayıt için bilgi amaçlı, hata değil). Bir `audio_url`'i indirip WAV olduğunu doğrula:

```bash
curl -sS -H "X-API-Key: $KEY" -o /tmp/claude-1000/check.wav "http://127.0.0.1:8001<audio_url>" && file /tmp/claude-1000/check.wav
```

Expected: `RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 22050 Hz`.

- [ ] **Step 4: Kullanıcı testi**

Kullanıcıdan: Furkan uygulamadan birkaç cümle söylesin; ekranda düzeltilmiş cümle görünüyor ve net bir sesle okunuyor mu? Sonucu ve gözlenen hataları ledger'a yaz. Kod değişikliği gerekirse yeni bir TDD döngüsü (bu task'ın kapsamı dışında).
