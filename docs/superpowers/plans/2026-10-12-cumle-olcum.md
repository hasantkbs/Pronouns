# Cümle Kaydı ve Cümle Doğruluğu Ölçümü Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Furkan'ın 60 ölçüm cümlesini Kayıt sayfasında ikişer kez okuyup kaydedebilmesi ve bu kayıtlarla Whisper'ın cümle düzeyindeki doğruluğunun (ham ve düzeltilmiş) ölçülmesi.

**Architecture:** Sunucu (`/home/mayasoft/app/Pronouns`, dal `cumle-olcum`) mevcut `/collect/*` ve `/record` uç noktalarına isteğe bağlı `kind` (`word`|`sentence`) alır; cümleler `audio/s_<sha1[:10]>/repN.wav` + `metadata.csv`'ye yazılır ve eğitime karışmaz. İstemci (`.claude/worktrees/react-rewrite`, dal `worktree-react-rewrite`) Kayıt sayfasına "Kelime | Cümle" modu ekler; çevrimdışı kuyruk kaydın türünü taşır. Ayrı bir CLI (`src/cli/evaluate_sentences.py`) kayıtları canlıdaki modelle çözüp `jiwer` ile WER/CER/tam isabet raporlar.

**Tech Stack:** Python 3 + FastAPI + pytest (`/home/mayasoft/app/venv/bin/python`), jiwer 4.0, soundfile; React 18 + TypeScript + Vite + vitest (node ortamı) + fake-indexeddb + idb.

**Spec:** `docs/superpowers/specs/2026-10-12-cumle-olcum-design.md`

## Global Constraints

- Python her zaman `/home/mayasoft/app/venv/bin/python`; sunucu test takımı: `/home/mayasoft/app/venv/bin/python -m pytest tests/ --ignore=tests/test_utils.py -q -p no:cacheprovider` (`test_utils.py` bu işten önce de bozuk).
- Sunucu işi `/home/mayasoft/app/Pronouns` içinde `cumle-olcum` dalında (zaten açık; spec commit'leri üstünde). İstemci işi `/home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite` içinde `worktree-react-rewrite` dalında; komutlar `web/` altından (`npm test`, `npx tsc -b`).
- `kind` gönderilmezse bugünkü kelime davranışı **aynen** sürer (eski APK uyumlu); geçersiz `kind` → 400.
- Cümle modunda yalnızca `olcum_*.txt` setleri; `config.SENTENCE_REPETITIONS = 2`.
- Cümle kayıtları eğitimi tetiklemez ve `metadata_words.csv`'ye yazılmaz.
- `data/users/FurkanV1/*` altındaki commit edilmemiş canlı veri değişikliklerine (`metadata_words.csv`, `words/*.wav`) **dokunma, stage etme**; her commit'te dosyaları tek tek `git add` et.
- Commit mesajları şu satırla biter: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- Dağıtım (merge, push, `npm run build`, `deploy-android.sh`, API yeniden başlatma) bu planın kapsamı **dışında**; kullanıcı onayıyla ayrıca yapılır. `deploy-web.sh` hiç kullanılmaz.

## Review Focus

1. **Mod değiştirirken geç gelen yanıt:** kelime modunda giden `next-word` isteği, kullanıcı "Cümle"ye geçtikten sonra dönerse cümle ekranında bir kelime gösterilmemeli (kelime kaydı cümle olarak kuyruğa girer, yüklemede 400 alır). Task 7'de `wordRequestRef` ile korunur; DOM test altyapısı olmadığı için elle doğrulanır.
2. **30 sn sınırı ile "Bitir" aynı anda:** kayıt iki kez kuyruğa yazılmamalı. Task 7'de `finishSentence` ilk satırda `recorderRef`'i boşaltır (Furkanca sayfasıyla aynı desen).
3. **Cümle kaydı sırasında sayfadan çıkış:** mikrofon açık kalmamalı. Task 7'de unmount temizliği + `startUnlessCancelled`.
4. **Kuyrukta bekleyen cümlenin seti sonradan değişirse:** yükleme 400 alır; Bekleyen Kayıtlar sayfasında "Hata" görünür ve kayıt silinebilir (mevcut davranış). Task 3 `test_unknown_sentence_is_rejected` sunucu tarafını sabitler.
5. **Telefon kaydı 44,1/48 kHz:** ölçüm betiği örnekleme hızına bakmaz; `ASRSystem.transcribe` `librosa.load(sr=16000)` ile yeniden örnekler. Task 4 testleri 16 kHz sahte kayıtla çalışır; gerçek telefon kayıtları ilk ölçümde doğrulanır.

---

### Task 1: Ölçüm cümle listeleri

**Files:**
- Create: `datasets/sentence_sets/olcum_sozluk_ici.txt`
- Create: `datasets/sentence_sets/olcum_sozluk_disi.txt`
- Test: `tests/test_sentence_sets.py`

**Interfaces:**
- Consumes: `src.utils.utils.normalize_turkish_text(text) -> str`; `data/users/FurkanV1/metadata_words.csv` (`transcription` sütunu; **canlı, commit edilmemiş** sürümü okunur — Furkan'ın güncel sözlüğü odur).
- Produces: iki set dosyası (satır başına bir cümle, UTF-8, 30'ar satır). Task 2–4 ve 7 dosya adlarını `olcum_` önekiyle tanır.

- [ ] **Step 1: Write the failing test**

`tests/test_sentence_sets.py`:

```python
# -*- coding: utf-8 -*-
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.utils import normalize_turkish_text

ROOT = Path(__file__).resolve().parent.parent
SETS = ROOT / "datasets" / "sentence_sets"
IN_VOCAB = SETS / "olcum_sozluk_ici.txt"
OUT_OF_VOCAB = SETS / "olcum_sozluk_disi.txt"


def _vocabulary():
    # SentenceCorrector ile ayni: cok kelimeli girdiler kelimelerine bolunur
    with open(ROOT / "data" / "users" / "FurkanV1" / "metadata_words.csv", encoding="utf-8") as f:
        return {
            w for row in csv.DictReader(f)
            for w in normalize_turkish_text(row["transcription"] or "").split()
        }


def _sentences(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_set_has_30_non_empty_sentences():
    for path in (IN_VOCAB, OUT_OF_VOCAB):
        sentences = _sentences(path)
        assert len(sentences) == 30, path.name
        assert all(sentences), path.name


def test_in_vocabulary_sentences_use_only_known_words():
    vocab = _vocabulary()
    for sentence in _sentences(IN_VOCAB):
        words = normalize_turkish_text(sentence).split()
        assert 3 <= len(words) <= 7, sentence
        missing = [w for w in words if w not in vocab]
        assert not missing, f"{sentence}: {missing}"


def test_out_of_vocabulary_sentences_have_an_unknown_word():
    vocab = _vocabulary()
    for sentence in _sentences(OUT_OF_VOCAB):
        words = normalize_turkish_text(sentence).split()
        assert any(w not in vocab for w in words), sentence


def test_no_sentence_repeats_across_sets():
    normalized = [
        normalize_turkish_text(s) for path in (IN_VOCAB, OUT_OF_VOCAB) for s in _sentences(path)
    ]
    assert len(normalized) == len(set(normalized))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_sentence_sets.py -q -p no:cacheprovider`
Expected: 4 FAIL, `FileNotFoundError` (`olcum_sozluk_ici.txt` yok).

- [ ] **Step 3: Create the sentence lists**

`datasets/sentence_sets/olcum_sozluk_ici.txt` (her kelime Furkan'ın sözlüğünde):

```text
Çocuk süt istiyor.
Doktor ilaç verdi.
Kedi balık yedi.
Hemşire soru sordu.
O kitap okudu.
Herkes kahve istiyor.
Hava çok soğuk.
Bu yemek çok sıcak.
Şimdi su yok.
O hemen geldi.
Neden hemen gitti?
O nerede kaldı?
Bu çok önemli bir konu.
Ekmek ve peynir aldı.
Kahve çok pahalı.
Bu telefon çok ucuz.
Doktor iyi bir insan.
Çocuk bir kedi gördü.
Polis bir araç buldu.
Şimdi yağmur var.
Hiç ağrı yok.
Bu sorun çok büyük.
Kim telefon etti?
O ne istiyor?
Bu nasıl oldu?
Doktor erken geldi.
Çocuk biraz su istiyor.
Bu pasta çok tatlı.
Hastane çok uzak.
Bu kapı kapalı.
```

`datasets/sentence_sets/olcum_sozluk_disi.txt` (her cümlede en az bir sözlük dışı kelime; olumsuzluk ve virgüllü cümleler bilerek var):

```text
Bugün kendimi iyi hissediyorum.
Başım çok ağrıyor.
Biraz su içmek istiyorum.
Yarın okula gideceğim.
Annemi aramak istiyorum.
Karnım çok acıktı.
Televizyonu açar mısın?
Dışarı çıkmak istiyorum.
Bu akşam ne yiyeceğiz?
Lütfen biraz yavaş konuş.
Telefonum nerede kaldı?
Kardeşim eve geldi mi?
Ben çok yoruldum.
Saat kaçta yatacağız?
Bana yardım eder misin?
Pencereyi kapatır mısın?
Hafta sonu parka gidelim.
Arkadaşımla oyun oynadım.
Bugün hava çok güzel.
Ödevimi henüz bitirmedim.
Ayakkabılarımı bulamıyorum.
Akşam yemeğinden sonra çay içelim.
Babam işten geç geldi.
Ben bunu anlamadım.
Doktora ne zaman gideceğiz?
Uykum geldi, yatmak istiyorum.
Teşekkür ederim, çok sağ ol.
Bugün okulda resim yaptık.
Müzik dinlemeyi çok seviyorum.
Biraz daha bekler misin?
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_sentence_sets.py -q -p no:cacheprovider`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add datasets/sentence_sets/olcum_sozluk_ici.txt datasets/sentence_sets/olcum_sozluk_disi.txt tests/test_sentence_sets.py
git commit -m "feat: add sentence lists for measuring sentence-level ASR accuracy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `/collect/*` uç noktalarında `kind`

**Files:**
- Modify: `api.py` (importlar; `_metadata_words_path`…`_get_next_word_and_rep` yardımcıları; `/collect/word-sets`, `/collect/next-word`, `/collect/progress`)
- Modify: `config.py` (`IDEAL_REPETITIONS` altına `SENTENCE_REPETITIONS`)
- Test: `tests/test_collect_sentences.py`

**Interfaces:**
- Consumes: `src.constants.RECORD_TYPE_SENTENCE` (`"cümle"`), `DATASET_DIRS`, `METADATA_FILENAMES` (cümle → `metadata.csv`).
- Produces (Task 3 kullanır): `_record_type(kind: str) -> str` (geçersizse `HTTPException(400)`), `_ideal_repetitions(record_type) -> int`, `_metadata_path(user_id, record_type) -> Path`, `_get_recorded_details(user_id, record_type) -> Dict[str, int]`, `_dataset_dir(record_type) -> Path`, `_read_set_file(set_file, record_type=RECORD_TYPE_WORD) -> List[str]`, `_SENTENCE_SET_PREFIX = "olcum_"`, `config.SENTENCE_REPETITIONS = 2`. Yanıt biçimleri değişmez; cümle modunda bulunamayan/geçersiz set → 404 `{"error": ...}`.

- [ ] **Step 1: Write the failing test**

`tests/test_collect_sentences.py`:

```python
# -*- coding: utf-8 -*-
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient

import api
import config
from src.constants import DATASET_DIRS, RECORD_TYPE_SENTENCE, RECORD_TYPE_WORD
from src.data.repository import UserDataRepository

client = TestClient(api.app)
_AUTH = {"X-API-Key": config.API_SECRET_KEY}
USER = "TestUser"
S1 = "Çocuk süt istiyor."
S2 = "Uykum geldi, yatmak istiyorum."


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    users = tmp_path / "users"
    sentence_sets = tmp_path / "sentence_sets"
    word_sets = tmp_path / "words_set"
    sentence_sets.mkdir()
    word_sets.mkdir()
    (sentence_sets / "olcum_a.txt").write_text(f"{S1}\n{S2}\n", encoding="utf-8")
    (sentence_sets / "sentence_set_1.txt").write_text("eski cümle\n", encoding="utf-8")
    (word_sets / "wordSet.txt").write_text("elma\narmut\n", encoding="utf-8")
    monkeypatch.setattr(config, "BASE_PATH", str(users))
    monkeypatch.setattr(api, "repo", UserDataRepository(str(users)))
    monkeypatch.setitem(DATASET_DIRS, RECORD_TYPE_SENTENCE, str(sentence_sets))
    monkeypatch.setitem(DATASET_DIRS, RECORD_TYPE_WORD, str(word_sets))
    return users


def _write_sentence_rows(users, sentence, count):
    path = users / USER / "metadata.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["file_path", "transcription", "repetition", "set_file"])
        for rep in range(1, count + 1):
            writer.writerow([f"audio/x/rep{rep}.wav", sentence, rep, "olcum_a.txt"])


def _get(path, **params):
    return client.get(path, params=params, headers=_AUTH)


def test_word_sets_lists_only_measurement_sets_for_sentences(data_dirs):
    assert _get("/collect/word-sets", kind="sentence").json() == {"sets": ["olcum_a.txt"]}


def test_word_sets_without_kind_lists_word_sets(data_dirs):
    assert _get("/collect/word-sets").json() == {"sets": ["wordSet.txt"]}


def test_invalid_kind_is_rejected(data_dirs):
    assert _get("/collect/word-sets", kind="harf").status_code == 400
    assert _get("/collect/next-word", user_id=USER, kind="x").status_code == 400
    assert _get("/collect/progress", user_id=USER, kind="x").status_code == 400


def test_next_sentence_starts_with_first_sentence(data_dirs):
    data = _get("/collect/next-word", user_id=USER, set_file="olcum_a.txt", kind="sentence").json()
    assert data["done"] is False
    assert data["word"] == S1
    assert data["rep"] == 1
    assert data["ideal_repetitions"] == 2
    assert data["total_words"] == 2


def test_next_sentence_moves_on_after_two_recordings(data_dirs):
    _write_sentence_rows(data_dirs, S1, 1)
    data = _get("/collect/next-word", user_id=USER, set_file="olcum_a.txt", kind="sentence").json()
    assert (data["word"], data["rep"]) == (S1, 2)

    _write_sentence_rows(data_dirs, S1, 1)
    data = _get("/collect/next-word", user_id=USER, set_file="olcum_a.txt", kind="sentence").json()
    assert (data["word"], data["rep"]) == (S2, 1)


def test_next_sentence_exclude_is_newline_separated(data_dirs):
    # Cumlede virgul var: virgulle bolunseydi S2 hic eslesmezdi
    data = _get(
        "/collect/next-word", user_id=USER, set_file="olcum_a.txt", kind="sentence",
        exclude=f"{S1}\n{S2}",
    ).json()
    assert data["done"] is True


def test_sentence_mode_rejects_non_measurement_set(data_dirs):
    res = _get("/collect/next-word", user_id=USER, set_file="sentence_set_1.txt", kind="sentence")
    assert res.status_code == 404
    res = _get("/collect/progress", user_id=USER, set_file="yok.txt", kind="sentence")
    assert res.status_code == 404


def test_sentence_mode_rejects_path_outside_sets(data_dirs):
    res = _get("/collect/next-word", user_id=USER, set_file="../words_set/wordSet.txt", kind="sentence")
    assert res.status_code == 404


def test_sentence_progress_counts_completed_sentences(data_dirs):
    _write_sentence_rows(data_dirs, S1, 2)
    _write_sentence_rows(data_dirs, S2, 1)
    data = _get("/collect/progress", user_id=USER, set_file="olcum_a.txt", kind="sentence").json()
    assert data == {
        "set_file": "olcum_a.txt",
        "total_words": 2,
        "completed_words": 1,
        "ideal_repetitions": 2,
    }


def test_word_next_without_kind_is_unchanged(data_dirs):
    data = _get("/collect/next-word", user_id=USER, set_file="wordSet.txt", exclude="elma").json()
    assert (data["word"], data["rep"], data["ideal_repetitions"]) == ("armut", 1, config.IDEAL_REPETITIONS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_collect_sentences.py -q -p no:cacheprovider`
Expected: `8 failed, 2 passed` (kelime yolu testleri bugün de geçer; cümle testleri `kind`'ı tanımadığı için kalır).

- [ ] **Step 3: Implement**

Repo kökünde (`/home/mayasoft/app/Pronouns`) şu yamayı `git apply` ile uygula (ya da elle aynı değişiklikleri yap):

```diff
diff --git a/api.py b/api.py
index 5978749e..9bda1b4b 100644
--- a/api.py
+++ b/api.py
@@ -2,7 +2,7 @@
 import hmac
 import logging
 from datetime import datetime
-from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
+from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request, HTTPException
 from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
 from fastapi.middleware.cors import CORSMiddleware
 import os
@@ -13,7 +13,7 @@ from typing import List, Optional, Dict, Any, Set
 import csv
 
 import config
-from src.constants import RECORD_TYPE_WORD, DATASET_DIRS, METADATA_FILENAMES
+from src.constants import RECORD_TYPE_WORD, RECORD_TYPE_SENTENCE, DATASET_DIRS, METADATA_FILENAMES
 from src.core.asr import ASRSystem
 from src.core.sentence_corrector import SentenceCorrector
 from src.core.tts import PiperTTS
@@ -81,16 +81,41 @@ if not _speaker_logger.handlers:
     _speaker_logger.addHandler(_sv_handler)
 training_services = {} # user_id -> TrainingService
 
+# Kayit sayfasinin `kind` parametresi -> kayit turu. Parametre yoksa "word" (eski APK).
+_KINDS = {"word": RECORD_TYPE_WORD, "sentence": RECORD_TYPE_SENTENCE}
+# Cumle modunda yalnizca olcum setleri sunulur; sentence_set_* dosyalari rastgele kelime listeleri
+_SENTENCE_SET_PREFIX = "olcum_"
+
+
+def _record_type(kind: str) -> str:
+    if kind not in _KINDS:
+        raise HTTPException(status_code=400, detail=f"Geçersiz kind: {kind}")
+    return _KINDS[kind]
+
+
+def _ideal_repetitions(record_type: str) -> int:
+    if record_type == RECORD_TYPE_SENTENCE:
+        return int(config.SENTENCE_REPETITIONS)
+    return int(config.IDEAL_REPETITIONS)
+
+
+def _metadata_path(user_id: str, record_type: str) -> Path:
+    return Path(config.BASE_PATH) / user_id / METADATA_FILENAMES[record_type]
+
+
 def _metadata_words_path(user_id: str) -> Path:
-    user_path = Path(config.BASE_PATH) / user_id
-    return user_path / METADATA_FILENAMES[RECORD_TYPE_WORD]
+    return _metadata_path(user_id, RECORD_TYPE_WORD)
 
 
 def _get_recorded_details_words(user_id: str) -> Dict[str, int]:
+    return _get_recorded_details(user_id, RECORD_TYPE_WORD)
+
+
+def _get_recorded_details(user_id: str, record_type: str) -> Dict[str, int]:
     """
-    pandas'a ihtiyaç duymadan metadata_words.csv içinden kelime->adet çıkarır.
+    pandas'a ihtiyaç duymadan metadata dosyasından metin->adet çıkarır.
     """
-    path = _metadata_words_path(user_id)
+    path = _metadata_path(user_id, record_type)
     if not path.exists():
         return {}
 
@@ -139,22 +164,25 @@ def _append_metadata_words_row(user_id: str, row: Dict[str, Any]) -> None:
         writer.writerow(row_to_save)
 
 
-def _dataset_words_dir() -> Path:
-    return Path(DATASET_DIRS[RECORD_TYPE_WORD])
+def _dataset_dir(record_type: str) -> Path:
+    return Path(DATASET_DIRS[record_type])
 
 
-def _read_word_set_file(set_file: str) -> List[str]:
+def _read_set_file(set_file: str, record_type: str = RECORD_TYPE_WORD) -> List[str]:
     """
-    collect_data.py mantığına benzer şekilde kelime setini txt'den okur.
+    collect_data.py mantığına benzer şekilde set dosyasını txt'den okur.
     Boş satırları atar.
     """
-    words_dir = _dataset_words_dir()
-    path = (words_dir / set_file).resolve()
+    set_dir = _dataset_dir(record_type).resolve()
+    path = (set_dir / set_file).resolve()
 
-    # Güvenlik: datasets/words_set dışına çıkmayı engelle
-    if words_dir.resolve() not in path.parents and path != words_dir.resolve():
+    # Güvenlik: set dizini dışına çıkmayı engelle
+    if set_dir not in path.parents:
         raise ValueError("Geçersiz set_file yolu.")
 
+    if record_type == RECORD_TYPE_SENTENCE and not path.name.startswith(_SENTENCE_SET_PREFIX):
+        raise FileNotFoundError(f"Set dosyası bulunamadı: {set_file}")
+
     if not path.exists():
         raise FileNotFoundError(f"Set dosyası bulunamadı: {set_file}")
 
@@ -167,7 +195,12 @@ def _read_word_set_file(set_file: str) -> List[str]:
     return lines
 
 
-def _get_next_word_and_rep(user_id: str, set_file: str, exclude: Optional[Set[str]] = None) -> Dict[str, Any]:
+def _get_next_word_and_rep(
+    user_id: str,
+    set_file: str,
+    exclude: Optional[Set[str]] = None,
+    record_type: str = RECORD_TYPE_WORD,
+) -> Dict[str, Any]:
     """
     collect_data.py'deki resume/eksik kayıt mantığının API versiyonu:
     - Setteki kelimeleri sırayla gezer
@@ -177,9 +210,9 @@ def _get_next_word_and_rep(user_id: str, set_file: str, exclude: Optional[Set[st
       kayıtları olan kelimeler için kullanılır; kullanıcı yüklemeyi
       beklemeden bir sonraki kelimeye geçebilsin diye)
     """
-    words = _read_word_set_file(set_file)
-    details = _get_recorded_details_words(user_id)
-    ideal = int(config.IDEAL_REPETITIONS)
+    words = _read_set_file(set_file, record_type)
+    details = _get_recorded_details(user_id, record_type)
+    ideal = _ideal_repetitions(record_type)
     exclude = exclude or set()
 
     for w in words:
@@ -323,39 +356,54 @@ async def upload_record(
     }
 
 
+def _set_error(record_type: str, e: Exception):
+    # Kelime yolu eskisi gibi 200 + error; cumle modunda bulunamayan/gecersiz set 404
+    if record_type == RECORD_TYPE_SENTENCE and isinstance(e, (FileNotFoundError, ValueError)):
+        return JSONResponse(status_code=404, content={"error": str(e)})
+    return {"error": str(e)}
+
+
 @app.get("/collect/word-sets")
-async def list_word_sets():
-    """datasets/words_set içindeki set dosyalarını listeler."""
-    d = _dataset_words_dir()
+async def list_word_sets(kind: str = "word"):
+    """Set dosyalarını listeler (cümle modunda yalnızca olcum_*.txt)."""
+    record_type = _record_type(kind)
+    d = _dataset_dir(record_type)
     if not d.exists():
         return {"sets": []}
-    sets = sorted([p.name for p in d.glob("*.txt")])
+    pattern = f"{_SENTENCE_SET_PREFIX}*.txt" if record_type == RECORD_TYPE_SENTENCE else "*.txt"
+    sets = sorted([p.name for p in d.glob(pattern)])
     return {"sets": sets}
 
 
 @app.get("/collect/next-word")
-async def get_next_word(user_id: str, set_file: str = "wordSet.txt", exclude: str = ""):
+async def get_next_word(user_id: str, set_file: str = "wordSet.txt", exclude: str = "", kind: str = "word"):
     """
-    Mobil tarafın 'collect_data.py' akışına benzer biçimde sıradaki kelimeyi alması için.
-    `exclude`: virgülle ayrılmış, atlanacak kelimeler (cihazda bekleyen kayıtları
-    olan kelimeler için — kullanıcı yüklemeden bir sonraki kelimeye geçebilsin).
+    Mobil tarafın 'collect_data.py' akışına benzer biçimde sıradaki kelimeyi/cümleyi alması için.
+    `exclude`: atlanacak metinler (cihazda bekleyen kayıtları olanlar için — kullanıcı
+    yüklemeden bir sonrakine geçebilsin). Kelimeler virgülle, cümleler (içlerinde
+    virgül olabilir) satır sonuyla ayrılır.
     """
+    record_type = _record_type(kind)
+    separator = "\n" if record_type == RECORD_TYPE_SENTENCE else ","
     try:
-        exclude_set = {w.strip() for w in exclude.split(",") if w.strip()}
-        return _get_next_word_and_rep(user_id=user_id, set_file=set_file, exclude=exclude_set)
+        exclude_set = {w.strip() for w in exclude.split(separator) if w.strip()}
+        return _get_next_word_and_rep(
+            user_id=user_id, set_file=set_file, exclude=exclude_set, record_type=record_type
+        )
     except Exception as e:
-        return {"error": str(e)}
+        return _set_error(record_type, e)
 
 
 @app.get("/collect/progress")
-async def get_progress(user_id: str, set_file: str = "wordSet.txt"):
+async def get_progress(user_id: str, set_file: str = "wordSet.txt", kind: str = "word"):
     """
     Set bazında ilerleme özeti.
     """
+    record_type = _record_type(kind)
     try:
-        words = _read_word_set_file(set_file)
-        details = _get_recorded_details_words(user_id)
-        ideal = int(config.IDEAL_REPETITIONS)
+        words = _read_set_file(set_file, record_type)
+        details = _get_recorded_details(user_id, record_type)
+        ideal = _ideal_repetitions(record_type)
         completed = sum(1 for w in words if int(details.get(w, 0)) >= ideal)
         return {
             "set_file": set_file,
@@ -364,7 +412,7 @@ async def get_progress(user_id: str, set_file: str = "wordSet.txt"):
             "ideal_repetitions": ideal,
         }
     except Exception as e:
-        return {"error": str(e)}
+        return _set_error(record_type, e)
 
 @app.post("/train")
 async def start_training(user_id: str, background_tasks: BackgroundTasks):
diff --git a/config.py b/config.py
index 5af62948..621b159e 100755
--- a/config.py
+++ b/config.py
@@ -40,6 +40,7 @@ CONSISTENCY_TOLERANCE = 0.5
 
 # --- Kayıt Tekrar Ayarları ---
 IDEAL_REPETITIONS = 5
+SENTENCE_REPETITIONS = 2  # Cumle olcum kaydi: her cumle 2 kez okunur
 
 # --- Sistem Ayarları ---
 GECICI_DOSYA_YOLU = "temp_recording.wav"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_collect_sentences.py tests/test_api_key_middleware.py -q -p no:cacheprovider`
Expected: `17 passed`

- [ ] **Step 5: Commit**

```bash
git add api.py config.py tests/test_collect_sentences.py
git commit -m "feat: serve measurement sentences through the collect endpoints

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `/record` cümle kaydı

**Files:**
- Modify: `api.py` (`import hashlib`; yeni `_find_sentence_set`; `upload_record`)
- Test: `tests/test_record_sentences.py`

**Interfaces:**
- Consumes (Task 2): `_record_type`, `_dataset_dir`, `_read_set_file`, `_SENTENCE_SET_PREFIX`, `RECORD_TYPE_SENTENCE`.
- Produces: `POST /record` form alanı `kind` (varsayılan `"word"`). `kind=sentence`: `data/users/<id>/audio/s_<sha1(cümle)[:10]>/rep<N>.wav`; `metadata.csv` satırı `file_path` (kullanıcı dizinine göre göreli, `audio/s_…/repN.wav`), `transcription`, `repetition`, `quality_score`, `rms`, `snr_db`, `duration`, `set_file`. Ölçüm setlerinde olmayan cümle → 400, dosya yazılmaz. Cümle eğitimi tetiklemez. Task 4 bu `metadata.csv`'yi okur.

- [ ] **Step 1: Write the failing test**

`tests/test_record_sentences.py`:

```python
# -*- coding: utf-8 -*-
import csv
import hashlib
import io
import math
import os
import struct
import sys
import wave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient

import api
import config
from src.constants import DATASET_DIRS, RECORD_TYPE_SENTENCE
from src.data.repository import UserDataRepository

client = TestClient(api.app)
_AUTH = {"X-API-Key": config.API_SECRET_KEY}
USER = "TestUser"
SENTENCE = "Uykum geldi, yatmak istiyorum."


def _wav_bytes(seconds=1.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / rate)))
            for i in range(int(seconds * rate))
        )
        w.writeframes(frames)
    return buf.getvalue()


class _FakeTraining:
    def __init__(self):
        self.started = 0

    def start_training(self, background_tasks):
        self.started += 1


@pytest.fixture
def env(tmp_path, monkeypatch):
    users = tmp_path / "users"
    sets = tmp_path / "sentence_sets"
    sets.mkdir()
    (sets / "olcum_sozluk_disi.txt").write_text(f"{SENTENCE}\n", encoding="utf-8")
    monkeypatch.setattr(config, "BASE_PATH", str(users))
    monkeypatch.setattr(api, "repo", UserDataRepository(str(users)))
    monkeypatch.setitem(DATASET_DIRS, RECORD_TYPE_SENTENCE, str(sets))
    monkeypatch.setattr(api.settings_service, "load_settings", lambda user_id: {"self_learning": True})
    training = _FakeTraining()
    monkeypatch.setattr(api, "get_training_service", lambda user_id: training)
    return users, training


def _post(word, kind=None):
    data = {"user_id": USER, "word": word}
    if kind is not None:
        data["kind"] = kind
    return client.post(
        "/record",
        data=data,
        files={"audio": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        headers=_AUTH,
    )


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sentence_recording_is_saved_under_audio_with_hash_dir(env):
    users, _ = env
    res = _post(SENTENCE, kind="sentence")
    assert res.status_code == 200
    assert res.json()["rep"] == 1

    digest = hashlib.sha1(SENTENCE.encode("utf-8")).hexdigest()[:10]
    assert (users / USER / "audio" / f"s_{digest}" / "rep1.wav").exists()

    rows = _rows(users / USER / "metadata.csv")
    assert len(rows) == 1
    assert rows[0]["transcription"] == SENTENCE
    assert rows[0]["file_path"] == f"audio/s_{digest}/rep1.wav"
    assert rows[0]["repetition"] == "1"
    assert rows[0]["set_file"] == "olcum_sozluk_disi.txt"
    assert not (users / USER / "metadata_words.csv").exists()


def test_second_sentence_recording_is_rep_two(env):
    _post(SENTENCE, kind="sentence")
    assert _post(SENTENCE, kind="sentence").json()["rep"] == 2


def test_sentence_recording_never_starts_training(env):
    _, training = env
    for _ in range(config.IDEAL_REPETITIONS):
        _post(SENTENCE, kind="sentence")
    assert training.started == 0


def test_unknown_sentence_is_rejected(env):
    users, _ = env
    res = _post("Listede olmayan cümle.", kind="sentence")
    assert res.status_code == 400
    assert not (users / USER / "audio").exists()


def test_invalid_kind_is_rejected(env):
    assert _post("elma", kind="harf").status_code == 400


def test_record_without_kind_still_saves_a_word(env):
    users, training = env
    for _ in range(config.IDEAL_REPETITIONS):
        res = _post("elma")
    assert res.status_code == 200
    assert (users / USER / "words" / "elma" / f"rep{config.IDEAL_REPETITIONS}.wav").exists()
    assert len(_rows(users / USER / "metadata_words.csv")) == config.IDEAL_REPETITIONS
    assert not (users / USER / "metadata.csv").exists()
    assert training.started == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_record_sentences.py -q -p no:cacheprovider`
Expected: `4 failed, 2 passed` (`test_second_sentence_recording_is_rep_two` ve `test_record_without_kind_still_saves_a_word` bugün de geçer).

- [ ] **Step 3: Implement**

```diff
--- a/api.py
+++ b/api.py
@@ -11,6 +11,7 @@
 from pathlib import Path
 from typing import List, Optional, Dict, Any, Set
 import csv
+import hashlib
 
 import config
 from src.constants import RECORD_TYPE_WORD, RECORD_TYPE_SENTENCE, DATASET_DIRS, METADATA_FILENAMES
@@ -195,6 +196,14 @@
     return lines
 
 
+def _find_sentence_set(sentence: str) -> Optional[str]:
+    """Cümlenin geçtiği ölçüm setinin dosya adı; hiçbirinde yoksa None."""
+    for path in sorted(_dataset_dir(RECORD_TYPE_SENTENCE).glob(f"{_SENTENCE_SET_PREFIX}*.txt")):
+        if sentence in _read_set_file(path.name, RECORD_TYPE_SENTENCE):
+            return path.name
+    return None
+
+
 def _get_next_word_and_rep(
     user_id: str,
     set_file: str,
@@ -303,14 +312,26 @@
     user_id: str = Form(...), 
     word: str = Form(...), 
     audio: UploadFile = File(...),
-    background_tasks: BackgroundTasks = None
+    background_tasks: BackgroundTasks = None,
+    kind: str = Form("word"),
 ):
     """Mobil uygulamadan gelen ses kaydını saklar ve metadata günceller."""
-    user_dir = repo.get_save_path(user_id, RECORD_TYPE_WORD) / word
+    record_type = _record_type(kind)
+    set_file = None
+    if record_type == RECORD_TYPE_SENTENCE:
+        # Ölçüm yalnızca bilinen cümlelerle anlamlı; set_file gruplama için metadata'ya yazılır
+        set_file = _find_sentence_set(word)
+        if set_file is None:
+            raise HTTPException(status_code=400, detail="Cümle hiçbir ölçüm setinde yok.")
+        # Cümle klasör adı olamaz (noktalama, uzunluk): metnin özeti kullanılır
+        item_dir = f"s_{hashlib.sha1(word.encode('utf-8')).hexdigest()[:10]}"
+    else:
+        item_dir = word
+    user_dir = repo.get_save_path(user_id, record_type) / item_dir
     user_dir.mkdir(parents=True, exist_ok=True)
     
     # Kaçıncı tekrar olduğunu bul
-    current_count = repo.get_recorded_count(user_id, RECORD_TYPE_WORD, word)
+    current_count = repo.get_recorded_count(user_id, record_type, word)
     rep = current_count + 1
     
     file_path = user_dir / f"rep{rep}.wav"
@@ -321,7 +342,13 @@
     quality_info = calculate_audio_quality(str(file_path))
     
     # Metadata kaydet
-    relative_path = normalize_path_for_cross_platform(str(file_path.absolute()), repo.get_user_path(user_id))
+    if record_type == RECORD_TYPE_SENTENCE:
+        # Kullanıcı dizinine göre göreli yol (audio/s_<özet>/repN.wav): ölçüm betiği taşınabilir okur
+        relative_path = normalize_path_for_cross_platform(
+            str(file_path.absolute()), repo.get_user_path(user_id).absolute()
+        )
+    else:
+        relative_path = normalize_path_for_cross_platform(str(file_path.absolute()), repo.get_user_path(user_id))
     
     metadata_entry = {
         "file_path": relative_path,
@@ -332,12 +359,15 @@
         "snr_db": quality_info['snr_db'],
         "duration": quality_info['duration']
     }
+    if set_file is not None:
+        metadata_entry["set_file"] = set_file
     
-    repo.save_metadata(user_id, RECORD_TYPE_WORD, [metadata_entry], append=True)
+    repo.save_metadata(user_id, record_type, [metadata_entry], append=True)
     
-    # Self-learning: yeterli yeni kayıt varsa arka planda fine-tune başlat
+    # Self-learning: yeterli yeni kayıt varsa arka planda fine-tune başlat.
+    # Cümle kayıtları yalnızca ölçüm içindir, eğitimi tetiklemez.
     settings = settings_service.load_settings(user_id)
-    if settings.get("self_learning", True):
+    if record_type == RECORD_TYPE_WORD and settings.get("self_learning", True):
         details = _get_recorded_details_words(user_id)
         total_samples = sum(details.values())
         ideal = int(config.IDEAL_REPETITIONS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_record_sentences.py tests/test_collect_sentences.py -q -p no:cacheprovider`
Expected: `16 passed`

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_record_sentences.py
git commit -m "feat: store sentence recordings separately from word training data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Ölçüm betiği

**Files:**
- Create: `src/cli/evaluate_sentences.py`
- Test: `tests/test_evaluate_sentences.py`

**Interfaces:**
- Consumes: Task 3'ün `metadata.csv` biçimi; `src.utils.utils.normalize_turkish_text`; `ASRSystem.transcribe(path, fuzzy_correct=False) -> (Optional[str], float)`; `SentenceCorrector(vocabulary, lexicon_path).correct(text) -> str`; `SettingsService().load_settings(user_id)`; `ModelService.find_personalized_model(key)`.
- Produces: `normalize(text) -> str`, `score(refs, hyps) -> {"wer","cer","exact"}`, `load_rows(user_dir) -> [{"audio_path","file_path","expected","set_file"}]`, `evaluate(rows, transcribe, correct) -> {"summary","rows","skipped"}` (`summary` anahtarları set dosya adları + `"toplam"`; her biri `{"count","raw","corrected"}`), `worst(rows, n=10)`, `write_report(report, reports_dir, now=None) -> Path` (`sentence_eval_<YYYYMMDD-HHMMSS>.json`), `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

`tests/test_evaluate_sentences.py`:

```python
# -*- coding: utf-8 -*-
import csv
import json
import os
import sys
import wave
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.cli import evaluate_sentences as ev


def _silent_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)


def _metadata(user_dir, rows):
    user_dir.mkdir(parents=True, exist_ok=True)
    with open(user_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "transcription", "repetition", "set_file"])
        writer.writeheader()
        for i, (text, set_file) in enumerate(rows, start=1):
            rel = f"audio/s_{i}/rep1.wav"
            _silent_wav(user_dir / rel)
            writer.writerow({"file_path": rel, "transcription": text, "repetition": 1, "set_file": set_file})


def test_normalize_handles_turkish_case_and_punctuation():
    assert ev.normalize("İLAÇ Iıdır, değil mi?") == "ilaç ııdır değil mi"
    assert ev.normalize(None) == ""


def test_score_reports_wer_cer_and_exact_rate():
    result = ev.score(["su yok", "hava çok soğuk"], ["su yok", "hava çok"])
    assert result["exact"] == 0.5
    assert result["wer"] == pytest.approx(1 / 5)
    assert 0 < result["cer"] < 1


def test_load_rows_resolves_paths_relative_to_user_dir(tmp_path):
    user_dir = tmp_path / "U"
    _metadata(user_dir, [("Su yok.", "olcum_a.txt")])
    rows = ev.load_rows(user_dir)
    assert rows == [{
        "audio_path": str(user_dir / "audio/s_1/rep1.wav"),
        "file_path": "audio/s_1/rep1.wav",
        "expected": "Su yok.",
        "set_file": "olcum_a.txt",
    }]


def test_load_rows_without_metadata_is_empty(tmp_path):
    assert ev.load_rows(tmp_path) == []


def test_evaluate_groups_by_set_and_compares_raw_and_corrected(tmp_path):
    user_dir = tmp_path / "U"
    _metadata(user_dir, [
        ("Su yok.", "olcum_sozluk_ici.txt"),
        ("Hava çok soğuk.", "olcum_sozluk_ici.txt"),
        ("Başım ağrıyor.", "olcum_sozluk_disi.txt"),
    ])
    raw = {"su yok": "su yok", "hava çok soğuk": "hava çok sok", "başım ağrıyor": "başım"}
    by_path = {r["audio_path"]: raw[ev.normalize(r["expected"])] for r in ev.load_rows(user_dir)}
    fixes = {"hava çok sok": "Hava çok soğuk."}

    report = ev.evaluate(
        ev.load_rows(user_dir),
        transcribe=lambda path: by_path[path],
        correct=lambda text: fixes.get(text, text),
    )

    ici = report["summary"]["olcum_sozluk_ici.txt"]
    assert ici["count"] == 2
    assert ici["raw"]["exact"] == 0.5
    assert ici["corrected"]["exact"] == 1.0
    assert report["summary"]["olcum_sozluk_disi.txt"]["raw"]["wer"] == 0.5
    assert report["summary"]["toplam"]["count"] == 3
    assert report["skipped"] == []
    assert len(report["rows"]) == 3


def test_evaluate_counts_empty_asr_output_as_all_wrong(tmp_path):
    user_dir = tmp_path / "U"
    _metadata(user_dir, [("Su yok.", "olcum_a.txt")])
    report = ev.evaluate(ev.load_rows(user_dir), transcribe=lambda p: None, correct=lambda t: t or "")
    assert report["summary"]["toplam"]["raw"]["wer"] == 1.0
    assert report["rows"][0]["raw"] == ""


def test_evaluate_skips_missing_and_unreadable_audio(tmp_path):
    user_dir = tmp_path / "U"
    _metadata(user_dir, [("Su yok.", "olcum_a.txt"), ("Hava çok soğuk.", "olcum_a.txt"), ("Kim geldi?", "olcum_a.txt")])
    (user_dir / "audio/s_2/rep1.wav").unlink()
    (user_dir / "audio/s_3/rep1.wav").write_bytes(b"bozuk")
    calls = []

    report = ev.evaluate(
        ev.load_rows(user_dir),
        transcribe=lambda p: calls.append(p) or "su yok",
        correct=lambda t: t,
    )

    assert [s["file_path"] for s in report["skipped"]] == ["audio/s_2/rep1.wav", "audio/s_3/rep1.wav"]
    assert len(calls) == 1
    assert report["summary"]["toplam"]["count"] == 1


def test_worst_orders_by_raw_error(tmp_path):
    rows = [
        {"expected": "a", "raw": "a", "corrected": "a", "raw_wer": 0.0, "corrected_wer": 0.0},
        {"expected": "b", "raw": "x", "corrected": "x", "raw_wer": 1.0, "corrected_wer": 1.0},
        {"expected": "c", "raw": "c d", "corrected": "c", "raw_wer": 0.5, "corrected_wer": 0.0},
    ]
    assert [r["expected"] for r in ev.worst(rows, 2)] == ["b", "c"]


def test_write_report_uses_timestamped_name(tmp_path):
    out = ev.write_report({"summary": {}}, tmp_path, now=datetime(2026, 10, 12, 9, 5, 7))
    assert out == tmp_path / "sentence_eval_20261012-090507.json"
    assert json.loads(out.read_text(encoding="utf-8")) == {"summary": {}}


def test_main_without_recordings_exits_with_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev.config, "BASE_PATH", str(tmp_path))
    assert ev.main(["--user", "Yok"]) == 1
    assert "Cümle kaydı yok" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_evaluate_sentences.py -q -p no:cacheprovider`
Expected: collection ERROR, `ImportError: cannot import name 'evaluate_sentences'`.

- [ ] **Step 3: Implement**

`src/cli/evaluate_sentences.py`:

```python
# -*- coding: utf-8 -*-
"""
Cümle kayıtlarında ASR doğruluğunu ölçer (alt proje B1).

Her cümle kaydı için iki çıktı karşılaştırılır:
  ham         : kullanıcının modeli, fuzzy düzeltme kapalı (canlı /translate ile aynı)
  düzeltilmiş : ham + SentenceCorrector (Furkan'ın gördüğü cümle)

Kullanım:
    python -m src.cli.evaluate_sentences --user FurkanV1 [--device cpu]
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import jiwer
import soundfile as sf

import config
from src.utils.utils import normalize_turkish_text

TOTAL = "toplam"


def normalize(text: Optional[str]) -> str:
    """Büyük/küçük harf (İ/I) ve noktalama farkı hata sayılmasın."""
    return normalize_turkish_text(text or "")


def score(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    """Normalize edilmiş metin listeleri için WER, CER ve tam isabet oranı."""
    return {
        "wer": float(jiwer.wer(refs, hyps)),
        "cer": float(jiwer.cer(refs, hyps)),
        "exact": sum(r == h for r, h in zip(refs, hyps)) / len(refs),
    }


def load_rows(user_dir: Path) -> List[Dict[str, str]]:
    path = Path(user_dir) / "metadata.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            file_path = row["file_path"]
            audio = Path(file_path)
            if not audio.is_absolute():
                audio = Path(user_dir) / audio
            rows.append({
                "audio_path": str(audio),
                "file_path": file_path,
                "expected": row["transcription"],
                "set_file": row.get("set_file") or "bilinmiyor",
            })
    return rows


def evaluate(
    rows: List[Dict[str, str]],
    transcribe: Callable[[str], Optional[str]],
    correct: Callable[[str], str],
) -> Dict:
    results, skipped = [], []
    for row in rows:
        try:
            sf.info(row["audio_path"])
        except Exception as e:
            skipped.append({"file_path": row["file_path"], "reason": str(e)})
            continue
        raw = transcribe(row["audio_path"]) or ""
        corrected = correct(raw) or ""
        expected = normalize(row["expected"])
        results.append({
            "file_path": row["file_path"],
            "set_file": row["set_file"],
            "expected": row["expected"],
            "raw": raw,
            "corrected": corrected,
            "raw_wer": float(jiwer.wer(expected, normalize(raw))),
            "corrected_wer": float(jiwer.wer(expected, normalize(corrected))),
        })

    groups: Dict[str, List[Dict]] = {}
    for r in results:
        groups.setdefault(r["set_file"], []).append(r)
    if results:
        groups[TOTAL] = results

    summary = {}
    for name, items in groups.items():
        refs = [normalize(r["expected"]) for r in items]
        summary[name] = {
            "count": len(items),
            "raw": score(refs, [normalize(r["raw"]) for r in items]),
            "corrected": score(refs, [normalize(r["corrected"]) for r in items]),
        }
    return {"summary": summary, "rows": results, "skipped": skipped}


def worst(rows: List[Dict], n: int = 10) -> List[Dict]:
    """Ham çıktıda hatası en yüksek n kayıt (B1 modelin kendisini ölçer)."""
    return sorted(rows, key=lambda r: (r["raw_wer"], r["corrected_wer"]), reverse=True)[:n]


def write_report(report: Dict, reports_dir: Path, now: Optional[datetime] = None) -> Path:
    now = now or datetime.now()
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"sentence_eval_{now.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def print_summary(report: Dict) -> None:
    print(f"\n{'Grup':<28}{'Adet':>6}  {'WER ham/düz':>14}  {'CER ham/düz':>14}  {'Tam ham/düz':>14}")
    for name, s in report["summary"].items():
        raw, cor = s["raw"], s["corrected"]
        print(
            f"{name:<28}{s['count']:>6}  "
            f"{raw['wer']:>6.1%}/{cor['wer']:<7.1%}  "
            f"{raw['cer']:>6.1%}/{cor['cer']:<7.1%}  "
            f"{raw['exact']:>6.1%}/{cor['exact']:<7.1%}"
        )
    print("\nEn kötü örnekler (ham WER'e göre):")
    for r in worst(report["rows"]):
        print(f"- [{r['raw_wer']:.0%}] beklenen: {r['expected']}")
        print(f"        ham: {r['raw']}")
        print(f"  düzeltilmiş: {r['corrected']}")
    if report["skipped"]:
        print(f"\nAtlanan kayıt: {len(report['skipped'])}")


def _load_asr(user_id: str):
    # Ağır importlar (transformers, peft) yalnızca gerçek ölçümde yüklenir
    from src.core.asr import ASRSystem
    from src.services.model_service import ModelService
    from src.services.settings_service import SettingsService

    # api.get_asr ile aynı seçim: ayar "whisper" ise Whisper adaptörü
    settings = SettingsService().load_settings(user_id)
    model_key = f"{user_id}_whisper" if settings.get("model") == "whisper" else user_id
    return ASRSystem(model_name=ModelService.find_personalized_model(model_key), user_id=user_id)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cümle kayıtlarında ASR doğruluğunu ölçer.")
    parser.add_argument("--user", default="FurkanV1")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto",
                        help="cpu: API GPU'yu kullanırken bellek yetmezse")
    args = parser.parse_args(argv)

    if args.device == "cpu":
        # CUDA henüz başlatılmadı; torch GPU'yu görmez
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    rows = load_rows(Path(config.BASE_PATH) / args.user)
    if not rows:
        print("Cümle kaydı yok.")
        return 1

    from src.core.sentence_corrector import SentenceCorrector

    asr = _load_asr(args.user)
    corrector = SentenceCorrector(asr.user_vocabulary, Path(config.LEXICON_PATH))
    report = evaluate(
        rows,
        transcribe=lambda path: asr.transcribe(path, fuzzy_correct=False)[0],
        correct=corrector.correct,
    )
    report["user_id"] = args.user
    report["model_type"] = asr.model_type

    print_summary(report)
    out = write_report(report, Path(config.REPORTS_DIR))
    print(f"\nRapor: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_evaluate_sentences.py -q -p no:cacheprovider`
Expected: `10 passed`

- [ ] **Step 5: Smoke-check the CLI entry point**

Run: `/home/mayasoft/app/venv/bin/python -m src.cli.evaluate_sentences --user FurkanV1; echo "exit=$?"`
Expected: `Cümle kaydı yok.` ve `exit=1` (henüz cümle kaydı yok; model yüklenmez).

- [ ] **Step 6: Run the full server suite**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/ --ignore=tests/test_utils.py -q -p no:cacheprovider`
Expected: `110 passed, 2 skipped` (öncesi 82 passed + bu plandaki 28 yeni test; atlanan 2 test önceden de atlanıyordu — sayı farklıysa yeni başarısızlık olmadığını kontrol et).

- [ ] **Step 7: Commit**

```bash
git add src/cli/evaluate_sentences.py tests/test_evaluate_sentences.py
git commit -m "feat: add sentence accuracy evaluation CLI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: İstemci API'sinde `kind`

Bu task'tan itibaren çalışma dizini: `/home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite` (yamalar bu kökten uygulanır, `npm`/`npx` komutları `web/` altından çalışır).

**Files:**
- Modify: `web/src/lib/api.ts`
- Test: `web/src/lib/api.test.ts` (yeni)

**Interfaces:**
- Consumes: Task 2–3 sunucu sözleşmesi (`kind` sorgu parametresi / form alanı; cümle `exclude`'u `\n` ile ayrılır).
- Produces: `export type RecordKind = 'word' | 'sentence'`; `getWordSets(kind = 'word')`, `getNextWord(setFile, exclude, kind = 'word')`, `getProgress(setFile, kind = 'word')`, `uploadRecording(word, audioBlob, kind = 'word')`. Mevcut çağıranlar değişmeden derlenir.

- [ ] **Step 1: Write the failing test**

`web/src/lib/api.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getNextWord, getProgress, getWordSets, uploadRecording } from './api';

function stubFetch(body: unknown) {
  const fetchMock = vi.fn(
    async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify(body), { status: 200 }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function sentQuery(fetchMock: ReturnType<typeof stubFetch>): URLSearchParams {
  const url = String(fetchMock.mock.calls[0][0]);
  return new URLSearchParams(url.split('?')[1] ?? '');
}

function sentForm(fetchMock: ReturnType<typeof stubFetch>): FormData {
  return fetchMock.mock.calls[0][1]?.body as FormData;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('kayıt türü (kind)', () => {
  it('getWordSets asks for sentence sets', async () => {
    const fetchMock = stubFetch({ sets: [] });
    await getWordSets('sentence');
    expect(sentQuery(fetchMock).get('kind')).toBe('sentence');
  });

  it('getWordSets defaults to word', async () => {
    const fetchMock = stubFetch({ sets: [] });
    await getWordSets();
    expect(sentQuery(fetchMock).get('kind')).toBe('word');
  });

  it('getNextWord joins sentence exclusions with newlines', async () => {
    const fetchMock = stubFetch({ done: true });
    await getNextWord('olcum_a.txt', ['Uykum geldi, yatmak istiyorum.', 'Su yok.'], 'sentence');
    const query = sentQuery(fetchMock);
    expect(query.get('kind')).toBe('sentence');
    expect(query.get('exclude')).toBe('Uykum geldi, yatmak istiyorum.\nSu yok.');
  });

  it('getNextWord keeps comma-separated word exclusions', async () => {
    const fetchMock = stubFetch({ done: true });
    await getNextWord('wordSet.txt', ['elma', 'armut']);
    const query = sentQuery(fetchMock);
    expect(query.get('kind')).toBe('word');
    expect(query.get('exclude')).toBe('elma,armut');
  });

  it('getProgress sends kind', async () => {
    const fetchMock = stubFetch({ completed_words: 0, total_words: 30, ideal_repetitions: 2 });
    await getProgress('olcum_a.txt', 'sentence');
    expect(sentQuery(fetchMock).get('kind')).toBe('sentence');
  });

  it('uploadRecording sends kind in the form', async () => {
    const fetchMock = stubFetch({});
    await uploadRecording('Su yok.', new Blob([new Uint8Array([1])]), 'sentence');
    const form = sentForm(fetchMock);
    expect(form.get('kind')).toBe('sentence');
    expect(form.get('word')).toBe('Su yok.');
  });

  it('uploadRecording defaults to word', async () => {
    const fetchMock = stubFetch({});
    await uploadRecording('elma', new Blob([new Uint8Array([1])]));
    expect(sentForm(fetchMock).get('kind')).toBe('word');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (`web/`): `npx vitest run src/lib/api.test.ts`
Expected: `7 failed` (`kind` gönderilmiyor; `get('kind')` → `null`).

- [ ] **Step 3: Implement**

```diff
diff --git a/web/src/lib/api.ts b/web/src/lib/api.ts
index e0d0b730..c009110f 100644
--- a/web/src/lib/api.ts
+++ b/web/src/lib/api.ts
@@ -76,12 +76,18 @@ export async function rebuildVoiceProfile(): Promise<{ sample_count: number }> {
   return res.json();
 }
 
+/** Kayıt sayfasının modu: kelime (eğitim verisi) ya da cümle (ölçüm). */
+export type RecordKind = 'word' | 'sentence';
+
 export interface WordSetsResponse {
   sets: string[];
 }
 
-export async function getWordSets(): Promise<WordSetsResponse> {
-  const res = await fetch(`${API_BASE_URL}/collect/word-sets`, { headers: authHeaders() });
+export async function getWordSets(kind: RecordKind = 'word'): Promise<WordSetsResponse> {
+  const params = new URLSearchParams({ kind });
+  const res = await fetch(`${API_BASE_URL}/collect/word-sets?${params.toString()}`, {
+    headers: authHeaders(),
+  });
   if (!res.ok) throw new Error(`Kelime setleri alınamadı: ${res.status}`);
   return res.json();
 }
@@ -97,11 +103,17 @@ export interface NextWordResponse {
   error?: string;
 }
 
-export async function getNextWord(setFile: string, exclude: string[]): Promise<NextWordResponse> {
+export async function getNextWord(
+  setFile: string,
+  exclude: string[],
+  kind: RecordKind = 'word',
+): Promise<NextWordResponse> {
   const params = new URLSearchParams({
     user_id: USER_ID,
     set_file: setFile,
-    exclude: exclude.join(','),
+    // Cümlelerin içinde virgül olabilir: sunucu cümleleri satır sonuyla ayırır
+    exclude: exclude.join(kind === 'sentence' ? '\n' : ','),
+    kind,
   });
   const res = await fetch(`${API_BASE_URL}/collect/next-word?${params.toString()}`, {
     headers: authHeaders(),
@@ -118,8 +130,11 @@ export interface ProgressResponse {
   ideal_repetitions: number;
 }
 
-export async function getProgress(setFile: string): Promise<ProgressResponse> {
-  const params = new URLSearchParams({ user_id: USER_ID, set_file: setFile });
+export async function getProgress(
+  setFile: string,
+  kind: RecordKind = 'word',
+): Promise<ProgressResponse> {
+  const params = new URLSearchParams({ user_id: USER_ID, set_file: setFile, kind });
   const res = await fetch(`${API_BASE_URL}/collect/progress?${params.toString()}`, {
     headers: authHeaders(),
   });
@@ -127,10 +142,15 @@ export async function getProgress(setFile: string): Promise<ProgressResponse> {
   return res.json();
 }
 
-export async function uploadRecording(word: string, audioBlob: Blob): Promise<void> {
+export async function uploadRecording(
+  word: string,
+  audioBlob: Blob,
+  kind: RecordKind = 'word',
+): Promise<void> {
   const form = new FormData();
   form.append('user_id', USER_ID);
   form.append('word', word);
+  form.append('kind', kind);
   form.append('audio', audioBlob, `${word}.wav`);
   const res = await fetch(`${API_BASE_URL}/record`, {
     method: 'POST',
```

- [ ] **Step 4: Run test to verify it passes**

Run (`web/`): `npx vitest run src/lib/api.test.ts && npx tsc -b`
Expected: `7 passed`, `tsc` çıktısız.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/api.ts web/src/lib/api.test.ts
git commit -m "feat: send the recording kind to the collect and record endpoints

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Çevrimdışı kuyrukta kayıt türü

**Files:**
- Modify: `web/src/lib/pendingRecordingsStore.ts`
- Modify: `web/src/pages/PendingUploadsPage.tsx`
- Test: `web/src/lib/pendingRecordingsStore.test.ts`

**Interfaces:**
- Consumes (Task 5): `RecordKind`, `uploadRecording(word, blob, kind)`.
- Produces (Task 7 kullanır): `PendingTake.kind?: RecordKind`; `takeKind(take) -> RecordKind` (alan yoksa `'word'`); `addTake(word, audioBlob, kind = 'word', now = new Date())`; `getPendingWords(kind = 'word') -> Set<string>`. IndexedDB sürümü 1 kalır.

- [ ] **Step 1: Write the failing tests**

```diff
diff --git a/web/src/lib/pendingRecordingsStore.test.ts b/web/src/lib/pendingRecordingsStore.test.ts
index 464e09c6..b8b93bf3 100644
--- a/web/src/lib/pendingRecordingsStore.test.ts
+++ b/web/src/lib/pendingRecordingsStore.test.ts
@@ -1,4 +1,5 @@
 import 'fake-indexeddb/auto';
+import { openDB } from 'idb';
 import { beforeEach, describe, expect, it } from 'vitest';
 import {
   _resetForTests,
@@ -7,6 +8,7 @@ import {
   getPendingWords,
   getTakesForWord,
   removeTake,
+  takeKind,
 } from './pendingRecordingsStore';
 
 beforeEach(async () => {
@@ -50,4 +52,32 @@ describe('pendingRecordingsStore', () => {
     await addTake('vişne', blob);
     expect(await getPendingWords()).toEqual(new Set(['nar', 'vişne']));
   });
+
+  it('stores the kind of a take', async () => {
+    const blob = new Blob([new Uint8Array([1])], { type: 'audio/wav' });
+    await addTake('Su yok.', blob, 'sentence');
+    const [take] = await getAllTakes();
+    expect(take.kind).toBe('sentence');
+    expect(takeKind(take)).toBe('sentence');
+  });
+
+  it('treats takes saved before kind existed as words', async () => {
+    await getAllTakes(); // veritabanini olusturur
+    const db = await openDB('furkanca-pending-recordings', 1);
+    await db.add('takes', { word: 'elma', audioBlob: new Blob([]), recordedAt: '2026-10-01T00:00:00Z' });
+    db.close();
+
+    const [take] = await getAllTakes();
+    expect(takeKind(take)).toBe('word');
+    expect(await getPendingWords('word')).toEqual(new Set(['elma']));
+    expect(await getPendingWords('sentence')).toEqual(new Set());
+  });
+
+  it('filters pending texts by kind', async () => {
+    const blob = new Blob([new Uint8Array([1])], { type: 'audio/wav' });
+    await addTake('nar', blob);
+    await addTake('Su yok.', blob, 'sentence');
+    expect(await getPendingWords('word')).toEqual(new Set(['nar']));
+    expect(await getPendingWords('sentence')).toEqual(new Set(['Su yok.']));
+  });
 });
```

- [ ] **Step 2: Run tests to verify they fail**

Run (`web/`): `npx vitest run src/lib/pendingRecordingsStore.test.ts`
Expected: `3 failed, 4 passed` (`takeKind is not a function`, `now.toISOString is not a function`).

- [ ] **Step 3: Implement**

```diff
diff --git a/web/src/lib/pendingRecordingsStore.ts b/web/src/lib/pendingRecordingsStore.ts
index 8523d161..70d12915 100644
--- a/web/src/lib/pendingRecordingsStore.ts
+++ b/web/src/lib/pendingRecordingsStore.ts
@@ -1,10 +1,19 @@
 import { openDB, type IDBPDatabase } from 'idb';
+import type { RecordKind } from './api';
 
 export interface PendingTake {
   id: number;
+  /** Kelime ya da cümle metni. */
   word: string;
   audioBlob: Blob;
   recordedAt: string;
+  /** Cümle modundan önce kaydedilenlerde yok. */
+  kind?: RecordKind;
+}
+
+/** Alanı olmayan eski kayıtlar kelimedir. */
+export function takeKind(take: Pick<PendingTake, 'kind'>): RecordKind {
+  return take.kind ?? 'word';
 }
 
 const DB_NAME = 'furkanca-pending-recordings';
@@ -31,10 +40,11 @@ function getDb(): Promise<IDBPDatabase> {
 export async function addTake(
   word: string,
   audioBlob: Blob,
+  kind: RecordKind = 'word',
   now: Date = new Date(),
 ): Promise<PendingTake> {
   const db = await getDb();
-  const record = { word, audioBlob, recordedAt: now.toISOString() };
+  const record = { word, audioBlob, recordedAt: now.toISOString(), kind };
   const id = (await db.add(STORE_NAME, record)) as number;
   return { id, ...record };
 }
@@ -54,9 +64,9 @@ export async function getTakesForWord(word: string): Promise<PendingTake[]> {
   return db.getAllFromIndex(STORE_NAME, WORD_INDEX, word);
 }
 
-export async function getPendingWords(): Promise<Set<string>> {
+export async function getPendingWords(kind: RecordKind = 'word'): Promise<Set<string>> {
   const takes = await getAllTakes();
-  return new Set(takes.map((t) => t.word));
+  return new Set(takes.filter((t) => takeKind(t) === kind).map((t) => t.word));
 }
 
 export async function _resetForTests(): Promise<void> {
diff --git a/web/src/pages/PendingUploadsPage.tsx b/web/src/pages/PendingUploadsPage.tsx
index fad02932..fa705066 100644
--- a/web/src/pages/PendingUploadsPage.tsx
+++ b/web/src/pages/PendingUploadsPage.tsx
@@ -1,6 +1,6 @@
 import { useEffect, useState } from 'react';
 import { uploadRecording } from '../lib/api';
-import { getAllTakes, removeTake, type PendingTake } from '../lib/pendingRecordingsStore';
+import { getAllTakes, removeTake, takeKind, type PendingTake } from '../lib/pendingRecordingsStore';
 
 type UploadStatus = 'uploading' | 'error';
 
@@ -26,7 +26,7 @@ export default function PendingUploadsPage() {
     for (const take of current) {
       setStatus((prev) => ({ ...prev, [take.id]: 'uploading' }));
       try {
-        await uploadRecording(take.word, take.audioBlob);
+        await uploadRecording(take.word, take.audioBlob, takeKind(take));
         await removeTake(take.id);
         setStatus((prev) => {
           const next = { ...prev };
```

- [ ] **Step 4: Run tests to verify they pass**

Run (`web/`): `npx vitest run src/lib/pendingRecordingsStore.test.ts && npx tsc -b`
Expected: `7 passed`, `tsc` çıktısız.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/pendingRecordingsStore.ts web/src/lib/pendingRecordingsStore.test.ts web/src/pages/PendingUploadsPage.tsx
git commit -m "feat: remember whether a pending take is a word or a sentence

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Kayıt sayfasında Cümle modu

**Files:**
- Modify (tam yeniden yazım): `web/src/pages/KayitPage.tsx`

**Interfaces:**
- Consumes: `RecordKind`, `getWordSets/getNextWord/getProgress(…, kind)` (Task 5); `addTake(word, blob, kind)`, `getPendingWords(kind)` (Task 6); `ManualRecorder` (`MIN_RECORDING_MS` 500, `MAX_RECORDING_MS` 30000), `startAudioCapture`, `startUnlessCancelled`, `formatElapsed`, `pcmToWav` (mevcut).
- Produces: Kullanıcı arayüzü — "Kelime | Cümle" seçimi; kelime modu bugünküyle aynı (geri sayımlı kayıt); cümle modu dokun-başlat/dokun-bitir, "Bu cümle: n / 2", "Atla" (oturumluk, `exclude`'a eklenir), "Sıradaki Cümle".

Bileşen testi yok: projede DOM test altyapısı yok (Furkanca sayfasıyla aynı karar). Mantık Task 5–6 testleri ve mevcut `manualRecorder`/`deadlines` testleriyle örtülü; sayfa `tsc` + build + elle doğrulanır.

- [ ] **Step 1: Replace the page**

`web/src/pages/KayitPage.tsx`:

```tsx
import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { getNextWord, getProgress, getWordSets, type RecordKind } from '../lib/api';
import { startAudioCapture, type AudioCaptureHandle } from '../lib/audioCapture';
import { startUnlessCancelled } from '../lib/deadlines';
import { formatElapsed } from '../lib/furkancaFlow';
import { ManualRecorder } from '../lib/manualRecorder';
import { pcmToWav } from '../lib/wav';
import { addTake, getAllTakes, getPendingWords, getTakesForWord } from '../lib/pendingRecordingsStore';

const DEFAULT_WORD_SET = 'wordSet.txt';

type SentencePhase = 'idle' | 'starting' | 'recording';

export default function KayitPage() {
  const [kind, setKind] = useState<RecordKind>('word');
  const [sets, setSets] = useState<string[]>([]);
  const [setFile, setSetFile] = useState(DEFAULT_WORD_SET);
  const [word, setWord] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [currentCount, setCurrentCount] = useState(0);
  const [idealReps, setIdealReps] = useState(0);
  const [totalWords, setTotalWords] = useState(0);
  const [completedWords, setCompletedWords] = useState(0);
  const [seconds, setSeconds] = useState(2);
  const [remaining, setRemaining] = useState(0);
  const [recording, setRecording] = useState(false);
  const [sentencePhase, setSentencePhase] = useState<SentencePhase>('idle');
  const [elapsedMs, setElapsedMs] = useState(0);
  const [status, setStatus] = useState<string | null>(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [wordPendingCount, setWordPendingCount] = useState(0);

  // Mod/set degisince onceki istegin gec gelen yaniti yeni modun metnini ezmesin
  const wordRequestRef = useRef(0);
  const captureRef = useRef<AudioCaptureHandle | null>(null);
  const recorderRef = useRef<ManualRecorder | null>(null);

  // Sayfadan cikilinca cumle kaydinin mikrofonu kapanir
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      captureRef.current?.stop();
      captureRef.current = null;
    };
  }, []);

  const unit = kind === 'word' ? 'kelime' : 'cümle';
  const busy = recording || sentencePhase !== 'idle';

  const loadSets = useCallback(async () => {
    try {
      const { sets: s } = await getWordSets(kind);
      setSets(s);
      setSetFile((current) => (s.length > 0 && !s.includes(current) ? s[0] : current));
    } catch {
      // set listesi opsiyonel; varsayılan setFile ile devam edilir
    }
  }, [kind]);

  const refreshWord = useCallback(async () => {
    if (!setFile) return; // cümle setleri henüz yüklenmedi
    const requestId = ++wordRequestRef.current;
    try {
      const pending = Array.from(await getPendingWords(kind));
      const data = await getNextWord(setFile, [...pending, ...skipped], kind);
      if (requestId !== wordRequestRef.current) return;
      setWord(data.word ?? null);
      setDone(data.done === true);
      setCurrentCount(data.current_count ?? 0);
      setIdealReps(data.ideal_repetitions ?? 0);
      setTotalWords((prev) => data.total_words ?? prev);
    } catch (e) {
      setStatus(`${kind === 'word' ? 'Kelime' : 'Cümle'} alınamadı: ${(e as Error).message}`);
    }
  }, [setFile, kind, skipped]);

  const refreshProgress = useCallback(async () => {
    if (!setFile) return;
    try {
      const data = await getProgress(setFile, kind);
      setCompletedWords(data.completed_words ?? 0);
      setTotalWords((prev) => data.total_words ?? prev);
      setIdealReps((prev) => data.ideal_repetitions ?? prev);
    } catch {
      // ilerleme opsiyonel
    }
  }, [setFile, kind]);

  const refreshPendingCounts = useCallback(async () => {
    const all = await getAllTakes();
    setPendingCount(all.length);
    if (word) {
      const forWord = await getTakesForWord(word);
      setWordPendingCount(forWord.length);
    } else {
      setWordPendingCount(0);
    }
  }, [word]);

  useEffect(() => {
    void loadSets();
  }, [loadSets]);

  useEffect(() => {
    void refreshWord();
    void refreshProgress();
  }, [setFile, refreshWord, refreshProgress]);

  useEffect(() => {
    void refreshPendingCounts();
  }, [word, refreshPendingCounts]);

  function changeKind(next: RecordKind) {
    if (next === kind || busy) return;
    wordRequestRef.current++;
    setKind(next);
    // Cümle setleri sunucudan gelince ilki seçilir
    setSetFile(next === 'word' ? DEFAULT_WORD_SET : '');
    setSets([]);
    setSkipped([]);
    setWord(null);
    setDone(false);
    setCompletedWords(0);
    setTotalWords(0);
    setStatus(null);
  }

  async function handleRecord() {
    if (recording || !word) return;
    setRecording(true);
    setStatus(null);
    setRemaining(seconds);

    let handle: Awaited<ReturnType<typeof startAudioCapture>> | null = null;
    try {
      const chunks: Int16Array[] = [];
      handle = await startAudioCapture((chunk) => chunks.push(chunk));

      for (let i = seconds; i > 0; i--) {
        setRemaining(i);
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
      handle.stop();

      const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
      const merged = new Int16Array(totalLength);
      let offset = 0;
      for (const c of chunks) {
        merged.set(c, offset);
        offset += c.length;
      }

      const wavBlob = pcmToWav(merged, handle.sampleRate, 1);
      await addTake(word, wavBlob);
      await refreshPendingCounts();
      setStatus(`Cihazda kaydedildi: ${word}`);
    } catch (e) {
      setStatus(`Hata: ${(e as Error).message}`);
      if (handle) {
        handle.stop();
      }
    } finally {
      setRecording(false);
      setRemaining(0);
    }
  }

  const finishSentence = useCallback(async () => {
    const recorder = recorderRef.current;
    const capture = captureRef.current;
    // Ikinci cagri (dugme + 30 sn siniri ayni anda) yok sayilir
    if (!recorder || !capture || !word) return;
    recorderRef.current = null;
    captureRef.current = null;
    capture.stop();
    setSentencePhase('idle');

    const result = recorder.finish();
    if (result.tooShort) {
      setStatus('Çok kısa, tekrar dene');
      return;
    }
    try {
      await addTake(word, pcmToWav(result.pcm, capture.sampleRate, 1), 'sentence');
      await refreshPendingCounts();
      setStatus('Cihazda kaydedildi');
    } catch (e) {
      setStatus(`Hata: ${(e as Error).message}`);
    }
  }, [word, refreshPendingCounts]);

  async function startSentence() {
    if (sentencePhase !== 'idle' || !word) return;
    setSentencePhase('starting');
    setStatus(null);
    setElapsedMs(0);
    try {
      const handle = await startUnlessCancelled(
        () =>
          startAudioCapture((chunk) => {
            const recorder = recorderRef.current;
            if (!recorder) return;
            const reachedMax = recorder.push(chunk);
            setElapsedMs(recorder.durationMs);
            if (reachedMax) void finishSentence();
          }),
        () => mountedRef.current,
      );
      if (!handle) return;
      captureRef.current = handle;
      recorderRef.current = new ManualRecorder(handle.sampleRate);
      // Dugme ancak mikrofon gercekten acildiktan sonra "Bitir" olur: ilk hece kacmaz
      setSentencePhase('recording');
    } catch (e) {
      setSentencePhase('idle');
      setStatus(`Mikrofon açılamadı: ${(e as Error).message}`);
    }
  }

  function skipSentence() {
    if (!word || busy) return;
    setStatus(null);
    setSkipped((prev) => [...prev, word]);
  }

  const wordFull = word !== null && currentCount + wordPendingCount >= idealReps;
  const repsDone = currentCount + wordPendingCount;

  return (
    <main>
      <h1>Kayıt</h1>
      <p>
        <Link to="/kayit/bekleyen">Bekleyen Kayıtlar ({pendingCount})</Link>
      </p>

      <div role="group" aria-label="Kayıt türü">
        <button
          type="button"
          aria-pressed={kind === 'word'}
          disabled={busy}
          onClick={() => changeKind('word')}
        >
          Kelime
        </button>
        <button
          type="button"
          aria-pressed={kind === 'sentence'}
          disabled={busy}
          onClick={() => changeKind('sentence')}
        >
          Cümle
        </button>
      </div>

      <label>
        {kind === 'word' ? 'Kelime Seti' : 'Cümle Seti'}
        <select
          value={setFile}
          onChange={(e) => {
            setSkipped([]);
            setSetFile(e.target.value);
          }}
          disabled={busy}
        >
          {sets.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>

      <p>
        İlerleme: {completedWords} / {totalWords} {unit}
      </p>

      {word === null ? (
        done && <p>Tüm {unit}ler tamamlandı!</p>
      ) : kind === 'word' ? (
        <>
          <h2>{word}</h2>
          <p>
            {repsDone} / {idealReps} tekrar
          </p>

          <label>
            Kayıt süresi: {seconds}sn
            <input
              type="range"
              min={1}
              max={5}
              value={seconds}
              disabled={recording}
              onChange={(e) => setSeconds(Number(e.target.value))}
            />
          </label>

          <button onClick={handleRecord} disabled={recording || wordFull}>
            {recording ? `Kaydediliyor... ${remaining}` : 'Kaydet'}
          </button>

          {wordFull && (
            <button
              onClick={() => {
                void refreshWord();
                void refreshProgress();
              }}
            >
              Sıradaki Kelime
            </button>
          )}
        </>
      ) : (
        <>
          <h2 style={{ fontSize: '2rem' }}>{word}</h2>
          <p>
            Bu cümle: {repsDone} / {idealReps}
          </p>

          <button
            type="button"
            disabled={sentencePhase === 'starting' || (sentencePhase === 'idle' && wordFull)}
            onClick={() =>
              sentencePhase === 'recording' ? void finishSentence() : void startSentence()
            }
            style={{ width: '100%', minHeight: '20vh', fontSize: '1.5rem' }}
          >
            {sentencePhase === 'recording'
              ? `⏹ Bitir ${formatElapsed(elapsedMs)}`
              : sentencePhase === 'starting'
                ? 'Mikrofon açılıyor...'
                : '🎤 Okumaya Başla'}
          </button>

          {wordFull ? (
            <button
              type="button"
              onClick={() => {
                void refreshWord();
                void refreshProgress();
              }}
            >
              Sıradaki Cümle
            </button>
          ) : (
            <button type="button" disabled={busy} onClick={skipSentence}>
              Atla
            </button>
          )}
        </>
      )}

      {status && <p>{status}</p>}
    </main>
  );
}
```

- [ ] **Step 2: Type-check, test, build**

Run (`web/`): `npx tsc -b && npm test 2>&1 | tail -4 && npx vite build --outDir /tmp/claude-1000/kayit-build-check`
Expected: `tsc` çıktısız; `Test Files  6 passed`, `Tests  45 passed`; `✓ built`. (Canlı `dist/`'e dokunulmaz: o dağıtımdır.)

- [ ] **Step 3: Commit**

```bash
git add web/src/pages/KayitPage.tsx
git commit -m "feat: add a sentence mode to the recording page

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Elle doğrulama ve dağıtım (plan sonrası, kullanıcı onayıyla)

1. Sunucu: `cumle-olcum` → `master` merge (onay), kullanıcı `sudo systemctl restart pronouns-api.service`.
2. Web: `cd .claude/worktrees/react-rewrite/web && npm run build` (onay). Web'de: Cümle modu → set `olcum_sozluk_ici.txt`, bir cümle 2 kez kaydet, "Atla", Bekleyen Kayıtlar → Tümünü Yükle; `data/users/FurkanV1/metadata.csv` ve `audio/s_*/` oluştu mu; Kelime moduna dönüp eski akış.
3. APK: `web/deploy-android.sh` (onay); Furkan'ın telefonuna `/apk`'dan yeni APK.
4. Furkan 60 cümleyi ikişer kez okur; `python -m src.cli.evaluate_sentences --user FurkanV1` (bellek yetmezse `--device cpu`) → sonuç kullanıcıyla değerlendirilir (B2 kararı).
