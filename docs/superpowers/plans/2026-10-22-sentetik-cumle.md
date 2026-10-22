# Sentetik Cümle Ölçümü ve Eğitimi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Furkan'ın kelime kayıtlarından sentetik cümle sesi üretip canlı Whisper modelinin cümle doğruluğunu ölçmek; fark ≥ 5 puansa modeli kelime + sentetik cümlelerle, canlı adaptörden devam ederek ayrı bir klasöre eğitip karşılaştırmak.

**Architecture:** Saf ses/cümle yardımcıları `src/training/sentence_synth.py`'de. Ölçüm aracı `src/cli/evaluate_synthetic.py`, `evaluate_sentences`'in skor/değerlendirme fonksiyonlarını yeniden kullanır. `train_whisper_adapter.py`'ye `--init-from`, `--output-name`, `--no-sentences` eklenir; cümle sesleri eğitim sırasında `train.csv` kayıtlarından anında üretilir. Son iki görev gerçek ölçüm ve (koşullu) eğitimdir.

**Tech Stack:** Python 3.12 (`/home/mayasoft/app/venv/bin/python`), numpy, librosa, soundfile, pandas, jiwer, transformers + peft (Whisper LoRA), pytest.

**Spec:** `docs/superpowers/specs/2026-10-22-sentetik-cumle-design.md`

## Global Constraints

- Çalışma yeri: `/home/mayasoft/app/Pronouns`, dal `sentetik-cumle` (master'da çalışılmaz; merge ve push kullanıcı onayıyla).
- Python her zaman `/home/mayasoft/app/venv/bin/python`; komutlar repo kökünden.
- Test komutu: `/home/mayasoft/app/venv/bin/python -m pytest tests -q -p no:warnings --ignore=tests/test_utils.py` (`tests/test_utils.py` önceden beri toplanamıyor: `No module named 'utils'`; bu işin konusu değil). Başlangıçta 116 test geçer.
- **Asla stage etme:** `data/users/FurkanV1/*` (canlı veri, commit'lenmemiş değişiklikler var), `lib/main.dart`, `backups/`. Her `git add` dosya adlarıyla yapılır; `git add -A`/`git add .` yok.
- `datasets/` `.gitignore`'da: yeni cümle dosyaları `git add -f` ile eklenir.
- Ölçüm sesleri: `sentetik_olcum.txt` + `eval.csv`; eğitim sesleri: `sentetik_egitim.txt` + `train.csv`. İkisi karışmaz.
- Karar eşiği: cümle WER − kelime WER ≥ **0,05** → `egitime_gec`, değilse `dur`.
- Devam eğitimi: en fazla 10 tur, öğrenme hızı 1e-5, sabır 3; `SYNTH_SENTENCE_REPEATS = 2`.
- Canlı model klasörü `data/models/personalized_models/FurkanV1_whisper`'a bu planda **hiçbir şey yazılmaz**. Canlıya alma bu planın dışında, kullanıcı onayıyla.
- API'yi ve servisleri kullanıcı yeniden başlatır; bu plan hiçbir servisi yeniden başlatmaz.
- Commit mesajları `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` ile biter.

## Review Focus

1. **Cümle sesi gerçek konuşmaya benziyor mu:** kırpma eşiği (30 dB) zayıf ünsüzleri (ör. sondaki "-k", "-t") kesebilir; bu, cümle WER'ini yapay olarak kötüleştirir. Task 6'da birkaç sentetik ses dosyasının süresi ve en kötü örneklerin "hep son harf eksik" deseni gösterip göstermediği raporlanır.
2. **DataLoader işçileri aynı cümle sesini üretmemeli:** `transform` her çağrıda `random.Random(random.random())` kullanır; torch işçileri Python `random`'u ayrı tohumlar. Task 5 bunu sınamaz; Task 7'nin eğitim logunda kayıp düşüşü normal mi bakılır.
3. **Self-learning eğitimi artık cümle de görür:** `TrainingService` varsayılan trainer'ı çağırır; `train.csv` yoksa (yalnızca `metadata_words.csv`) cümleler sessizce eklenmez — Task 5'teki `_with_sentences` bu koşulu taşır.
4. **Ayarlar ekranının WER'i yanlış modele kaymamalı:** `--output-name` ile yapılan eğitimin raporu `training_FurkanV1_*.json` adıyla yazılırsa `TrainingService.get_last_wer` onu canlı modelin WER'i sanar. Task 5'teki `test_side_report_is_not_picked_up_as_the_live_training_report` bunu sabitler.
5. **Yarım kalmış ölçüm seti:** `ensure_eval_set` manifest'i en son yazar; üretim yarıda kalırsa bir sonraki çalıştırma baştan üretir (manifest yok). Elle silinmiş tek bir wav ise `evaluate`'in `skipped` listesine düşer, ölçüm durmaz.

---

### Task 1: Sentetik cümle listeleri

**Files:**
- Create: `datasets/sentence_sets/sentetik_olcum.txt` (167 cümle)
- Create: `datasets/sentence_sets/sentetik_egitim.txt` (1004 cümle)
- Test: `tests/test_synthetic_sentence_sets.py`

**Interfaces:**
- Produces: iki UTF-8 dosya, satır başına bir cümle, boş satır yok. Task 4 `sentetik_olcum.txt`'yi, Task 5 `sentetik_egitim.txt`'yi okur.

- [ ] **Step 1: Write the failing test**

`tests/test_synthetic_sentence_sets.py`:

```python
# -*- coding: utf-8 -*-
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.utils.utils import normalize_turkish_text

ROOT = Path(__file__).resolve().parent.parent
SETS = ROOT / "datasets" / "sentence_sets"
EVAL_SET = SETS / "sentetik_olcum.txt"
TRAIN_SET = SETS / "sentetik_egitim.txt"
USER_DIR = ROOT / "data" / "users" / "FurkanV1"


def _sentences(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def _transcripts(path, column):
    # Kayıtlar kelime kelime ama "sosyal medya" gibi çok kelimeli girdiler
    # tek kayıttır; cümledeki kelime ancak tek başına kaydı varsa üretilebilir.
    if not path.exists():
        pytest.skip(f"{path} yok (canlı veri)")
    with open(path, encoding="utf-8") as f:
        return {normalize_turkish_text(row[column] or "") for row in csv.DictReader(f)}


def test_sets_have_expected_size_and_no_blank_lines():
    assert len(_sentences(EVAL_SET)) >= 140
    assert len(_sentences(TRAIN_SET)) >= 950
    for path in (EVAL_SET, TRAIN_SET):
        assert all(_sentences(path)), path.name


def test_sentences_have_three_to_seven_words():
    for path in (EVAL_SET, TRAIN_SET):
        for sentence in _sentences(path):
            assert 3 <= len(normalize_turkish_text(sentence).split()) <= 7, sentence


def test_eval_sentences_are_covered_by_eval_recordings():
    vocab = _transcripts(USER_DIR / "eval.csv", "transcript")
    for sentence in _sentences(EVAL_SET):
        missing = [w for w in normalize_turkish_text(sentence).split() if w not in vocab]
        assert not missing, f"{sentence}: {missing}"


def test_train_sentences_are_covered_by_train_recordings():
    vocab = _transcripts(USER_DIR / "train.csv", "transcript")
    for sentence in _sentences(TRAIN_SET):
        missing = [w for w in normalize_turkish_text(sentence).split() if w not in vocab]
        assert not missing, f"{sentence}: {missing}"


def test_no_duplicates_within_a_set():
    for path in (EVAL_SET, TRAIN_SET):
        normalized = [normalize_turkish_text(s) for s in _sentences(path)]
        assert len(normalized) == len(set(normalized)), path.name


def test_sets_are_disjoint_from_each_other_and_from_real_measurement_sets():
    eval_set = {normalize_turkish_text(s) for s in _sentences(EVAL_SET)}
    train_set = {normalize_turkish_text(s) for s in _sentences(TRAIN_SET)}
    real = {
        normalize_turkish_text(s)
        for path in SETS.glob("olcum_*.txt")
        for s in _sentences(path)
    }
    assert not eval_set & train_set
    assert not eval_set & real
    assert not train_set & real
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_synthetic_sentence_sets.py -q -p no:warnings`
Expected: FAIL — `FileNotFoundError` (`sentetik_olcum.txt` yok).

- [ ] **Step 3: Create the eval sentence file**

`datasets/sentence_sets/sentetik_olcum.txt` (dosya tam olarak aşağıdaki satırlar, sonda tek yeni satır):

```text
Hemşire ilaç verdi.
Çocuk kitap okudu.
Kedi bahçe kapı gördü.
Arkadaş kahve istedi.
O okul gitti.
Polis kaza gördü.
Biz park gitti.
Köpek kapı baktı.
Çocuk dondurma istiyor.
Doğru cevap verdi.
Ben kahve istiyor.
O film izleme sevdi.
İki kedi bahçe oyun yaptı.
Yeni bilgisayar çok pahalı.
Eski ev çok soğuk.
Mavi kapı açık.
Beyaz kedi yatak uyku.
Deniz çok sıcak.
Orman çok yeşil.
Kahve çok sıcak.
Su çok soğuk.
Makarna çok iyi oldu.
Pilav ve et yedi.
Domates ve biber aldı.
Patates ve soğan aldı.
Peynir ve ekmek yedi.
Kiraz ve üzüm tatlı.
Muz ve portakal aldı.
Karpuz ve kavun yedi.
Elektrik hemen geldi.
Hastane çok yakın.
Okul çok uzak.
Mağaza indirim yaptı.
Firma yeni ürün yaptı.
Şirket yeni plan yaptı.
Müşteri şikayet etti.
Satıcı fiyat söyledi.
Lider karar verdi.
Hükümet yeni kanun yaptı.
Devlet vergi aldı.
Çocuk resim yaptı.
Ben iyi hissetti.
Sen nerede kaldı.
Siz ne istiyor.
Hangi kitap okudu.
Kapı açık mı.
Su sıcak mı.
Ağustos çok sıcak.
Ocak çok soğuk.
Mart yağmur çok.
Pazartesi okul başladı.
Cuma park gitti.
Cumartesi pazar gitti.
Perşembe hastane gitti.
Saat sekiz oldu.
Beş kitap okudu.
Dört kedi gördü.
İki kahve istedi.
Altı gün sürdü.
Dokuz saat çalıştı.
Yedi gün gezi yaptı.
Bin kişi geldi.
Bilgisayar sistem çalıştı.
Yeni yazılım test etti.
Site güvenlik iyi.
Veri analiz yaptı.
Grafik çok basit.
Video çok kısa.
Ses çok yüksek.
Işık çok az.
Sarı çiçek gördü.
Mor çiçek sevdi.
Turuncu kedi gördü.
Gri bulut geldi.
Yeşil çim gördü.
Kuş ağaç gördü.
Kaplan çok hızlı.
Maymun muz yedi.
İnek çim yedi.
Keçi yaprak yedi.
Ayı orman gitti.
Kurt köy geldi.
Fil çok büyük.
Kaplumbağa çok yavaş.
Yılan çim gitti.
Böcek yaprak yedi.
At tarla gitti.
Çiftlik çok büyük.
Köy çok küçük.
Şehir çok büyük.
Sokak çok uzun.
Cadde çok kısa.
Apartman çok eski.
Daire çok küçük.
Oda çok sıcak.
Banyo çok küçük.
Salon çok büyük.
Bahçe çok yeşil.
Garaj kapı açık.
Merdiven çok uzun.
Tedavi uzun sürdü.
Spor yapmak iyi.
Stres çok kötü.
Hemşire hemen geldi.
Çocuk çok ağlamak istiyor.
Arkadaş hastane gitti.
Kedi süre bekledi.
Sen ne düşündü.
O çok iyi insan.
Ben yeni bilgi öğrendi.
Çocuk yeni kitap aldı.
Maymun ağaç gitti.
Kuş çok hızlı uçmak istiyor.
Köpek çok hızlı gitti.
Kedi uyku istiyor.
Herkes erken geldi.
Hemşire ilaç ve su verdi.
Polis yardım etti.
Arkadaş yardım istedi.
Müşteri yeni ürün istedi.
Firma reklam yaptı.
Şirket yeni site yaptı.
Satıcı indirim yaptı.
Banka kredi verdi.
Vergi çok yüksek.
Fiyat çok yüksek.
Su çok ucuz.
Hedef çok yüksek.
Karar çok zor.
Sorun hemen çözüm buldu.
Plan çok basit.
Fikir çok iyi.
Sonuç çok iyi oldu.
Deneme çok kötü oldu.
Başarı çok zor.
Çocuk okul başladı.
Kitap çok uzun.
Film çok uzun sürdü.
Oyun çok kısa sürdü.
Müzik çok yüksek.
Ses çok az.
Gök çok mavi.
Bulut çok gri.
Yıldız çok az.
Nehir çok uzun.
Dağ çok yüksek.
Göl çok soğuk.
Deniz çok mavi.
Ağaç çok büyük.
Yaprak çok sarı.
Bitki çok küçük.
Çilek çok tatlı.
Armut çok sert.
Ceviz ve fındık aldı.
Zeytin ve peynir yedi.
Havuç ve marul aldı.
Lahana ve karnabahar aldı.
Brokoli çok iyi.
Bisküvi ve kek yedi.
Dondurma çok soğuk.
Çorba çok sıcak.
Yemek çok iyi oldu.
Et ve pilav yedi.
Büyük fırtına geldi.
Büyük deprem oldu.
Orman yangın oldu.
Cadde kaza oldu.
```

- [ ] **Step 4: Create the training sentence file**

`datasets/sentence_sets/sentetik_egitim.txt`:

```text
Doktor muayene yaptı.
Doktor hemen geldi.
Doktor soru sordu.
Doktor tedavi başladı.
Doktor çok iyi insan.
Hemşire aşı yaptı.
Hemşire yardım etti.
Hemşire doktor gibi çalıştı.
Hemşire hastane çalıştı.
Çocuk elma yedi.
Çocuk yumurta yedi.
Çocuk çikolata istedi.
Çocuk pasta yedi.
Çocuk okul gitti.
Çocuk park gitti.
Çocuk resim çizmek istiyor.
Çocuk çok hızlı koşmak istiyor.
Çocuk erken uyumak istiyor.
Çocuk kitap okumak istiyor.
Çocuk soru sordu.
Çocuk cevap verdi.
Çocuk yeni oyun öğrendi.
Çocuk iki elma aldı.
Çocuk üç kitap okudu.
Kedi süt istedi.
Kedi fare gördü.
Kedi bahçe gitti.
Kedi kapı bekledi.
Kedi kuş baktı.
Köpek kedi gördü.
Köpek bahçe gitti.
Köpek kapı bekledi.
Köpek çok hızlı koşmak istiyor.
Köpek yemek istiyor.
Köpek su istiyor.
Kuş ağaç gitti.
Kuş çok yüksek uçmak istiyor.
Kuş tohum yedi.
Tavuk yumurta verdi.
Tavuk tohum yedi.
İnek süt verdi.
Koyun çim yedi.
Keçi çim yedi.
Ördek göl gitti.
Kaz göl gitti.
Aslan et yedi.
Kaplan et yedi.
Timsah nehir bekledi.
Fare peynir yedi.
Sıçan bodrum gitti.
Ayı balık yedi.
Fil su istedi.
Kurt koyun gördü.
Tilki tavuk gördü.
Yılan taş gitti.
Kaplumbağa çok yavaş gitti.
Arkadaş telefon etti.
Arkadaş ev geldi.
Arkadaş çay istedi.
Arkadaş yardım etti.
Arkadaş haber verdi.
Arkadaş geç geldi.
Komşu kapı geldi.
Komşu yardım etti.
Komşu çay verdi.
Komşu börek yaptı.
Komşu çok iyi insan.
Akraba ev geldi.
Akraba telefon etti.
Dost çok önemli.
Dost yardım etti.
Kadın kitap okudu.
Kadın çay istedi.
Kadın çorba yaptı.
Kadın pazar gitti.
Erkek kahve istedi.
Erkek iş gitti.
Erkek park gitti.
İnsan çok çalıştı.
Herkes çok çalıştı.
Herkes yemek yedi.
Biri kapı geldi.
Onlar park gitti.
Onlar yemek yedi.
Onlar çok çalıştı.
Biz okul gitti.
Biz yeni ev aldı.
Biz deniz gitti.
Siz nerede kaldı.
Siz ne yaptı.
Sen ne istiyor.
Sen ne dedi.
Sen çok iyi insan.
Ben su istiyor.
Ben çay istiyor.
Ben süt istiyor.
Ben ekmek istiyor.
Ben yardım istiyor.
Ben doktor gitmek istiyor.
Ben ev gitmek istiyor.
Ben uyumak istiyor.
Ben konuşmak istiyor.
Ben okumak istiyor.
Ben yemek istiyor.
Ben dinlenme istiyor.
Ben kötü hissetti.
O hemen gitti.
O soru sordu.
O cevap verdi.
O iş bitirdi.
O yeni şey öğrendi.
O telefon aldı.
Süt çok soğuk.
Süt çok sıcak.
Çay çok sıcak.
Çay çok soğuk.
Kahve çok soğuk.
Ayran çok soğuk.
Su çok sıcak.
Yoğurt çok iyi.
Yumurta çok sıcak.
Elma çok tatlı.
Elma ve armut aldı.
Kayısı çok tatlı.
Şeftali çok tatlı.
İncir çok tatlı.
Hurma çok tatlı.
Muz çok tatlı.
Karpuz çok tatlı.
Kavun çok tatlı.
Üzüm çok tatlı.
Portakal çok tatlı.
Çikolata çok tatlı.
Pasta çok tatlı.
Kek çok tatlı.
Salata çok iyi.
Ispanak çok iyi.
Sarımsak çok sert.
Soğan çok sert.
Tavuk ve pilav yedi.
Balık ve salata yedi.
Süt ve yumurta aldı.
Elma ve muz aldı.
Çay ve kek istedi.
Kahve ve su istedi.
Yoğurt ve ayran aldı.
Badem ve fıstık aldı.
Patates ve havuç aldı.
Domates ve salatalık aldı.
Marul ve ıspanak aldı.
Makarna ve salata yedi.
Çorba ve ekmek yedi.
Börek ve çay istedi.
Hava çok sıcak.
Hava çok iyi.
Hava çok kötü.
Güneş çok sıcak.
Rüzgar çok sert.
Kar çok soğuk.
Yağmur çok sert.
Fırtına çok sert.
Bulut çok siyah.
Gök çok siyah.
Güneş çok yüksek.
Ateş çok sıcak.
Toprak çok sıcak.
Taş çok sert.
Deniz çok soğuk.
Göl çok sıcak.
Nehir çok hızlı.
Orman çok büyük.
Dağ çok soğuk.
Şimdi kar başladı.
Şimdi yağmur başladı.
Sonra güneş geldi.
Sonra hava soğuk oldu.
Kasım çok soğuk.
Aralık çok soğuk.
Şubat çok soğuk.
Temmuz çok sıcak.
Haziran çok sıcak.
Mayıs çok iyi.
Nisan yağmur çok.
Eylül okul başladı.
Ekim çok iyi.
Salı okul gitti.
Pazar ev kaldı.
Pazartesi iş başladı.
Cuma erken geldi.
Cumartesi deniz gitti.
Çarşamba doktor gitti.
Perşembe pazar gitti.
Saat dokuz oldu.
Saat on oldu.
Saat yedi oldu.
Zaman çok hızlı geçmek.
Zaman çok önemli.
Telefon çok pahalı.
Telefon çok eski.
Telefon ekran büyük.
Yeni telefon aldı.
İnternet çok yavaş.
İnternet çok hızlı.
Bilgisayar çok yavaş.
Bilgisayar çok hızlı.
Bilgisayar yeni aldı.
Program çok yavaş.
Yeni program yazdı.
Algoritma çok basit.
Algoritma çok karmaşık.
Kod çok uzun.
Kod çok kısa.
Sistem çok yavaş.
Sistem çok hızlı.
Rapor çok uzun.
Metin çok uzun.
Dosya silmek istiyor.
Klasör çok büyük.
Şifre çok zor.
Şifre çok kolay.
Ekran çok büyük.
Ekran çok küçük.
Video çok uzun.
Müzik dinlemek istiyor.
Kitap okumak istiyor.
Resim çizmek istiyor.
Spor yapmak istiyor.
Egzersiz yapmak istiyor.
Yüzmek çok iyi.
Koşmak çok iyi.
Okumak çok önemli.
Uyumak çok önemli.
Uyku çok önemli.
Sağlık çok önemli.
Aile çok önemli.
Eğitim çok önemli.
Sevgi çok önemli.
Güven çok önemli.
Zaman çok önemli değer.
Sağlık en önemli.
Aile en önemli.
Çocuk yatak yatmak istiyor.
Çocuk bahçe yürümek istiyor.
Çocuk çok zıplamak istiyor.
Çocuk çok gülmek istiyor.
Çocuk müzik dinlemek istiyor.
Çocuk beş elma saydı.
Çocuk kağıt kesmek istiyor.
Çocuk resim boyamak istiyor.
Çocuk kağıt yapıştırmak istiyor.
Çocuk çiçek dikmek istiyor.
Çocuk ağaç dikmek istiyor.
Çocuk okul kalmak istiyor.
Çocuk ev kalmak istiyor.
Çocuk çok konuştu.
Çocuk az konuştu.
Çocuk çok yemek yedi.
Çocuk çok su istedi.
Çocuk soğuk su istedi.
Çocuk sıcak süt istedi.
Çocuk hastane kaldı.
Çocuk çok korku hissetti.
Çocuk çok mutluluk hissetti.
Ben çok mutluluk hissetti.
Ben çok korku hissetti.
Ben çok stres hissetti.
Ben ağrı hissetti.
Ben baş ağrı hissetti.
Ben çok çalıştı.
Ben iş bitirdi.
Ben yeni iş buldu.
Ben yeni ev buldu.
Ben telefon kaybetti.
Ben para kaybetti.
Ben şifre kaybetti.
Ben oyun kazandı.
Ben kitap aldı.
Ben ekmek aldı.
Ben su aldı.
Ben süt aldı.
Ben yeni telefon aldı.
Ben haber duydu.
Ben ses duydu.
Ben kuş gördü.
Ben deniz gördü.
Ben çok düşündü.
Ben sonra düşündü.
Ben karar verdi.
Ben cevap verdi.
Ben soru sordu.
Ben rapor yazdı.
Ben kod yazdı.
Ben program yazdı.
Ben yeni dil öğrendi.
Ben çok şey öğrendi.
Ben çok şey yaşadı.
Ben çok şey gördü.
Ben şimdi geldi.
Ben sonra geldi.
Ben erken geldi.
Ben geç kaldı.
Ben ev kaldı.
Ben doktor konuştu.
Ben arkadaş konuştu.
Ben telefon konuştu.
Ben bilgisayar kullandı.
Ben ilaç kullandı.
Ben yeni ilaç kullandı.
Ben çok iyi anladı.
Ben hiç anladı mı.
Sen ne gördü.
Sen ne duydu.
Sen ne aldı.
Sen ne yedi.
Sen ne yaptı.
Sen ne söyledi.
Sen ne öğrendi.
Sen nerede çalıştı.
Sen nerede yaşadı.
Sen nasıl hissetti.
Sen neden geldi.
Sen neden gitti.
Sen neden kaldı.
Sen kim gördü.
Sen hangi kitap okudu.
Sen hangi renk sevdi.
Sen çay istiyor mu.
Sen kahve istiyor mu.
Sen su istiyor mu.
Sen yardım istiyor mu.
O çok iyi doktor.
O çok iyi arkadaş.
O çok iyi komşu.
O çok iyi öğretti.
O çok şey öğretti.
O hemen anladı.
O çok iyi bildi.
O cevap bildi.
O çözüm buldu.
O sorun çözmek istiyor.
O çok uzak yaşadı.
O köy yaşadı.
O şehir yaşadı.
O deniz gitti.
O dağ gitti.
O orman gitti.
O hastane gitti.
O banka gitti.
O mağaza gitti.
O pazar gitti.
O ofis gitti.
O fabrika çalıştı.
O şirket çalıştı.
O banka çalıştı.
O hastane çalıştı.
O okul çalıştı.
O ofis çalıştı.
O çiftlik çalıştı.
O tarla çalıştı.
O inşaat çalıştı.
Biz çok çalıştı.
Biz yemek yedi.
Biz çay istedi.
Biz deniz gördü.
Biz yeni şehir gördü.
Biz gezi yaptı.
Biz plan yaptı.
Biz karar verdi.
Biz oyun kazandı.
Biz oyun kaybetti.
Biz erken geldi.
Biz geç kaldı.
Biz ev kaldı.
Biz köy gitti.
Biz dağ gitti.
Biz orman gitti.
Biz çiftlik gitti.
Siz nerede yaşadı.
Siz ne düşündü.
Siz ne öğrendi.
Siz çok iyi insan.
Onlar çok iyi insan.
Onlar oyun kazandı.
Onlar ev aldı.
Onlar yeni ev buldu.
Onlar köy yaşadı.
Onlar şehir gitti.
Onlar erken geldi.
Onlar geç kaldı.
Onlar soru sordu.
Onlar cevap verdi.
Onlar karar verdi.
Herkes karar verdi.
Herkes cevap bildi.
Herkes haber duydu.
Herkes yardım etti.
Herkes çay istedi.
Kimse cevap bildi mi.
Kim kapı geldi.
Kim süt aldı.
Kim kitap aldı.
Kim pasta yedi.
Kim oyun kazandı.
Ne zaman geldi.
Ne zaman gitti.
Ne zaman başladı.
Neden ağlamak istiyor.
Neden geç kaldı.
Neden erken gitti.
Hangi gün geldi.
Hangi saat geldi.
Hangi renk sevdi.
Hangi oda kaldı.
Doktor ne dedi.
Doktor ilaç yazdı.
Doktor ameliyat yaptı.
Doktor çok yardım etti.
Doktor test yaptı.
Doktor sonuç söyledi.
Doktor diyet verdi.
Doktor dinlenme dedi.
Doktor egzersiz dedi.
Hemşire su verdi.
Hemşire ateş baktı.
Ağrı çok az.
Ağrı çok fazla.
Ateş çok yüksek.
Ateş çok düşük.
Basınç çok yüksek.
Basınç çok düşük.
Tedavi çok iyi oldu.
Ameliyat çok uzun sürdü.
Ameliyat çok kısa sürdü.
Muayene çok kısa sürdü.
Hastane çok büyük.
Enfeksiyon çok kötü.
Virüs çok hızlı.
Bakteri çok küçük.
Bağışıklık çok önemli.
Hastalık çok uzun sürdü.
İlaç çok pahalı.
İlaç çok iyi.
Aşı çok önemli.
Beslenme çok önemli.
Spor çok önemli.
Egzersiz çok önemli.
Dinlenme çok önemli.
Uyku çok az.
Uyku çok fazla.
Stres çok fazla.
Stres çok az.
Ev çok büyük.
Ev çok küçük.
Ev çok sıcak.
Ev çok eski.
Ev çok yeni.
Ev çok uzak.
Ev çok yakın.
Ev çok pahalı.
Ev çok ucuz.
Yeni ev çok büyük.
Eski ev çok küçük.
Oda çok soğuk.
Oda çok büyük.
Oda çok küçük.
Mutfak çok küçük.
Mutfak çok büyük.
Mutfak çok sıcak.
Banyo çok soğuk.
Banyo çok büyük.
Salon çok küçük.
Salon çok sıcak.
Balkon çok küçük.
Balkon çok büyük.
Bahçe çok büyük.
Bahçe çok küçük.
Bahçe çiçek çok.
Garaj çok büyük.
Bodrum çok soğuk.
Çatı çok yüksek.
Tavan çok yüksek.
Duvar çok eski.
Zemin çok soğuk.
Zemin çok sıcak.
Kapı çok eski.
Kapı çok büyük.
Kapı açık kaldı.
Pencere açık kaldı.
Pencere çok büyük.
Pencere çok küçük.
Merdiven çok yüksek.
Asansör çok yavaş.
Asansör çok hızlı.
Asansör çalıştı mı.
Yatak çok büyük.
Yatak çok küçük.
Yatak çok sert.
Yatak çok yumuşak.
Temizlik çok önemli.
Temizlik yapmak istiyor.
Ev temizlik yaptı.
Oda temizlik yaptı.
Mutfak temizlik yaptı.
Apartman çok yüksek.
Apartman çok yeni.
Daire çok büyük.
Daire çok pahalı.
Villa çok pahalı.
Villa çok büyük.
Mahalle çok eski.
Mahalle çok yeni.
Sokak çok kısa.
Cadde çok uzun.
Şehir çok küçük.
Şehir çok eski.
Köy çok büyük.
Köy çok uzak.
Park çok büyük.
Park çok küçük.
Park çok yakın.
Park çok uzak.
Okul çok yakın.
Okul çok büyük.
Okul çok küçük.
Okul çok eski.
Okul çok yeni.
Banka çok yakın.
Banka çok uzak.
Mağaza çok yakın.
Mağaza çok büyük.
Mağaza çok pahalı.
Mağaza çok ucuz.
Pazar çok ucuz.
Fabrika çok büyük.
Fabrika çok eski.
Ofis çok büyük.
Ofis çok küçük.
Ofis çok soğuk.
Ofis çok sıcak.
Ofis çok yakın.
Ofis çok uzak.
İş çok zor.
İş çok kolay.
İş çok fazla.
İş çok az.
İş çok önemli.
Görev çok zor.
Görev çok kolay.
Plan çok iyi.
Plan çok kötü.
Plan çok zor.
Plan çok kolay.
Strateji çok iyi.
Strateji çok kötü.
Taktik çok iyi.
Hedef çok zor.
Hedef çok önemli.
Karar çok önemli.
Karar çok kolay.
Fikir çok kötü.
Fikir çok yeni.
Fikir çok basit.
Öneri çok iyi.
Öneri çok kötü.
Yorum çok uzun.
Yorum çok kısa.
Yorum çok iyi.
Şikayet çok fazla.
Şikayet çok az.
Sorun çok büyük.
Sorun çok küçük.
Sorun çok zor.
Sorun çok basit.
Çözüm çok basit.
Çözüm çok zor.
Çözüm çok iyi.
Sonuç çok kötü oldu.
Sonuç çok önemli.
Başarı çok önemli.
Başarı çok iyi.
Hata çok büyük.
Hata çok küçük.
Hata çok önemli.
Risk çok yüksek.
Risk çok düşük.
Risk çok büyük.
Fırsat çok iyi.
Fırsat çok büyük.
Avantaj çok büyük.
Dezavantaj çok büyük.
Fayda çok büyük.
Zarar çok büyük.
Zarar çok az.
Fark çok büyük.
Fark çok az.
Değer çok yüksek.
Değer çok düşük.
Kalite çok iyi.
Kalite çok kötü.
Kalite çok yüksek.
Kalite çok düşük.
Fiyat çok düşük.
Fiyat çok iyi.
Fiyat çok pahalı gibi.
Para çok az.
Para çok fazla.
Para çok önemli.
Bütçe çok az.
Bütçe çok büyük.
Borç çok fazla.
Borç çok az.
Kredi çok pahalı.
Vergi çok fazla.
Vergi çok az.
Zam çok az.
Zam çok fazla.
İndirim çok iyi.
İndirim çok az.
Kampanya çok iyi.
Reklam çok uzun.
Reklam çok fazla.
Marka çok pahalı.
Marka çok iyi.
Ürün çok iyi.
Ürün çok kötü.
Ürün çok pahalı.
Ürün çok ucuz.
Müşteri çok fazla.
Müşteri çok az.
Satıcı çok iyi insan.
Satıcı yardım etti.
Firma çok büyük.
Firma çok küçük.
Şirket çok büyük.
Şirket çok küçük.
Şirket çok eski.
Şirket çok yeni.
Şirket para kazandı.
Şirket para kaybetti.
Şirket yeni ürün yaptı.
Şirket yeni ofis aldı.
Ekonomi çok iyi.
Ekonomi çok kötü.
Piyasa çok iyi.
Piyasa çok kötü.
Ticaret çok önemli.
Tarım çok önemli.
Üretim çok fazla.
Üretim çok az.
Tüketim çok fazla.
Tüketim çok az.
Yatırım çok önemli.
Yatırım çok iyi.
Sigorta çok pahalı.
Sigorta çok önemli.
Gelir çok az.
Gelir çok fazla.
Gider çok fazla.
Gider çok az.
Hesap çok kolay.
Hesap çok zor.
Nakit para yok.
Yeni hesap açmak istiyor.
Banka hesap açmak istiyor.
Ben su istiyor çünkü hava çok sıcak.
Ben çay istiyor çünkü hava çok soğuk.
Çocuk uyumak istiyor çünkü çok geç.
Ben doktor gitti çünkü ağrı çok fazla.
O erken gitti çünkü iş çok fazla.
Biz ev kaldı çünkü yağmur çok sert.
Biz park gitti çünkü hava çok iyi.
O çok çalıştı ama sonuç kötü oldu.
Ben çok düşündü ama karar vermek zor.
Ev çok büyük ama çok pahalı.
Telefon çok yeni ama çok yavaş.
Kahve çok sıcak ama çok iyi.
Yemek çok iyi ama çok az.
Okul çok uzak ama çok iyi.
İş çok zor ama çok önemli.
Ben ve arkadaş park gitti.
Ben ve komşu çay istedi.
Doktor ve hemşire hastane çalıştı.
Ben kahve ve kek istiyor.
Ben ekmek ve süt aldı.
Ben elma ve armut sevdi.
Ben kedi ve köpek sevdi.
Ben müzik ve film sevdi.
Ben yeni telefon ve bilgisayar aldı.
Çocuk için süt aldı.
Çocuk için yeni kitap aldı.
Çocuk için oyun aldı.
Aile için yemek yaptı.
Aile için çok çalıştı.
Sağlık için spor yapmak önemli.
Sağlık için uyku çok önemli.
Sağlık için su çok önemli.
Sağlık için egzersiz yapmak iyi.
Başarı için çok çalışmak önemli.
Okul için erken uyanmak önemli.
İş için yeni bilgisayar aldı.
Ben şimdi uyumak istiyor.
Ben şimdi yemek istiyor.
Ben şimdi ev gitmek istiyor.
Ben şimdi doktor gitmek istiyor.
Ben şimdi su istiyor.
Ben şimdi çay istiyor.
Ben şimdi oturmak istiyor.
Ben şimdi dinlenme istiyor.
Ben sonra konuşmak istiyor.
Ben sonra okumak istiyor.
Şimdi ne yapmak istiyor.
Şimdi nerede kalmak istiyor.
Sonra ne oldu.
Sonra ne yaptı.
Sonra nerede kaldı.
Önce kahve sonra iş.
Önce yemek sonra spor.
Önce doktor sonra ev.
Önce düşünmek sonra konuşmak.
Önce okumak sonra yazmak.
Ben biraz su istiyor.
Ben biraz çay istiyor.
Ben biraz ekmek istiyor.
Ben biraz dinlenme istiyor.
Ben biraz yardım istiyor.
Ben biraz zaman istiyor.
Biraz daha su istiyor.
Biraz daha çay istiyor.
Biraz daha zaman istiyor.
Biraz daha yemek istiyor.
Çok daha iyi oldu.
Çok daha kötü oldu.
Daha çok çalışmak istiyor.
Daha çok okumak istiyor.
Daha çok öğrenmek istiyor.
Daha az stres istiyor.
Ben hiç kahve sevdi mi.
Ben hep çay istiyor.
O hep geç kaldı.
O hep çok çalıştı.
O hep yardım etti.
Köpek hep bahçe kaldı.
Ben hemen geldi.
Ben hemen gitti.
Ben hemen anladı.
Doktor hemen ilaç verdi.
Polis hemen geldi.
Hemşire hemen yardım etti.
Arkadaş hemen telefon etti.
Bir kedi bahçe geldi.
Bir köpek kapı bekledi.
Bir kuş ağaç gitti.
İki çocuk park gitti.
İki kedi süt istedi.
İki arkadaş kahve istedi.
Üç kuş ağaç gitti.
Üç çocuk okul gitti.
Dört kişi geldi.
Beş kişi kaldı.
Altı kişi gitti.
Yedi gün yağmur sürdü.
Dokuz kişi yemek yedi.
On kişi soru sordu.
Saat sekiz okul başladı.
Saat dokuz iş başladı.
Saat on doktor geldi.
Saat beş iş bitirdi.
Saat altı ev geldi.
Saat yedi yemek yedi.
İki saat sürdü.
Üç saat sürdü.
Bir saat bekledi.
İki saat bekledi.
Uzun süre bekledi.
Kısa süre bekledi.
Kırmızı elma aldı.
Mavi gök gördü.
Mavi deniz gördü.
Yeşil yaprak gördü.
Sarı yaprak gördü.
Sarı kedi gördü.
Beyaz kedi gördü.
Siyah köpek gördü.
Kahverengi köpek gördü.
Gri kedi gördü.
Mor çiçek aldı.
Pembe çiçek aldı.
Turuncu kedi sevdi.
Lacivert kumaş aldı.
Turkuaz deniz gördü.
Beyaz bulut gördü.
Siyah bulut geldi.
Kırmızı renk sevdi.
Mavi renk sevdi.
Yeşil renk sevdi.
Sarı renk sevdi.
Mor renk sevdi.
Siyah renk sevdi.
Beyaz renk sevdi.
Altın çok pahalı.
Gümüş çok pahalı.
Bakır çok ucuz.
Demir çok sert.
Çelik çok sert.
Ahşap çok sıcak.
Plastik çok ucuz.
Kumaş çok yumuşak.
Pamuk çok yumuşak.
Yün çok sıcak.
İpek çok pahalı.
Deri çok pahalı.
Taş çok soğuk.
Teknoloji çok hızlı gelişmek.
Teknoloji çok önemli.
Yazılım çok önemli.
Yazılım test yaptı.
Donanım çok pahalı.
Donanım çok eski.
Sistem çalıştı mı.
Sistem çok eski.
Sistem çok yeni.
Güvenlik çok önemli.
Erişim çok kolay.
Erişim çok zor.
Bağlantı çok yavaş.
Bağlantı çok hızlı.
Site çok yavaş.
Site çok hızlı.
Platform çok büyük.
Forum çok aktif.
Blog yazmak istiyor.
Blog çok iyi.
Web site yaptı.
Eposta çok uzun.
Link çok uzun.
Sayfa çok uzun.
Liste çok uzun.
Liste çok kısa.
Tablo çok büyük.
Grafik çok iyi.
Grafik çok karmaşık.
Model çok büyük.
Model çok iyi.
Analiz çok uzun sürdü.
Analiz çok önemli.
Test çok uzun sürdü.
Test çok kısa sürdü.
Test çok kolay.
Test çok zor.
Veri çok fazla.
Veri çok az.
Veri çok önemli.
İstatistik çok önemli.
Rapor çok kısa.
Raporlama çok önemli.
Ölçüm çok önemli.
Değerlendirme çok uzun sürdü.
Performans çok iyi.
Performans çok kötü.
Hız çok önemli.
Hız çok yüksek.
Hız çok düşük.
Otomasyon çok önemli.
Optimizasyon çok zor.
Entegrasyon çok uzun sürdü.
Kullanıcı çok fazla.
Kullanıcı çok az.
Kullanıcı yorum yazdı.
Kullanıcı şikayet etti.
Kullanıcı yardım istedi.
Abonelik çok pahalı.
Abonelik çok ucuz.
Beğeni çok fazla.
Paylaşım çok fazla.
Medya çok önemli.
Sosyal medya çok önemli.
Haber çok kötü.
Haber çok iyi.
Haber çok önemli.
İletişim çok önemli.
Haberleşme çok önemli.
Toplum çok önemli.
Kültür çok önemli.
Dil çok önemli.
Din çok önemli.
Hukuk çok önemli.
Kanun çok önemli.
Kural çok önemli.
Kural çok basit.
Yasa çok yeni.
Adalet çok önemli değer.
Eşitlik çok önemli.
Özgürlük çok önemli.
Bağımsızlık çok önemli.
Barış çok iyi.
Savaş çok kötü.
Kötülük çok kötü.
İyilik çok iyi.
Merhamet çok önemli.
Şefkat çok önemli.
Ahlak çok önemli.
Erdem çok önemli.
Sorumluluk çok önemli.
Sorumluluk çok büyük.
Emek çok önemli.
Çaba çok önemli.
Azim çok önemli.
Dürüst insan çok önemli.
Güven çok zor.
Sevgi çok büyük.
Nefret çok kötü.
Öfke çok kötü.
Korku çok kötü.
Üzüntü çok kötü.
Şüphe çok kötü.
Duygu çok önemli.
İnanç çok önemli.
Mutluluk çok önemli.
Doğa çok önemli.
Doğa çok büyük.
Çevre çok önemli.
Orman çok önemli.
Ağaç çok önemli.
Toprak çok önemli.
Tohum çok küçük.
Bitki su istiyor.
Çiçek su istiyor.
Ağaç su istiyor.
Bahçe su istiyor.
Tarla su istiyor.
Çim su istiyor.
Ben çiçek dikmek istiyor.
Ben ağaç dikmek istiyor.
Ben çim biçmek istiyor.
Ben yün örmek istiyor.
Ben duvar boyamak istiyor.
Ben kapı boyamak istiyor.
Ben resim yapmak istiyor.
Ben yemek yapmak istiyor.
Ben çorba yapmak istiyor.
Ben pasta yapmak istiyor.
Ben kek yapmak istiyor.
Ben börek yapmak istiyor.
Ben salata yapmak istiyor.
Ben makarna yapmak istiyor.
Ben pilav yapmak istiyor.
Ben çay yapmak istiyor.
Ben kahve yapmak istiyor.
Ben yeni şey öğrenmek istiyor.
Ben yeni dil öğrenmek istiyor.
Ben çocuk öğretmek istiyor.
Ben kitap yazmak istiyor.
Ben blog yazmak istiyor.
Ben kod yazmak istiyor.
Ben program yazmak istiyor.
Ben dosya silmek istiyor.
Ben yeni ev almak istiyor.
Ben yeni telefon almak istiyor.
Ben deniz görmek istiyor.
Ben dağ görmek istiyor.
Ben orman görmek istiyor.
Ben arkadaş görmek istiyor.
Ben komşu görmek istiyor.
Ben doktor görmek istiyor.
Ben haber duymak istiyor.
Ben müzik duymak istiyor.
Ben çok yaşamak istiyor.
Ben köy yaşamak istiyor.
Ben şehir yaşamak istiyor.
Ben deniz yüzmek istiyor.
Ben göl yüzmek istiyor.
Ben park yürümek istiyor.
Ben orman yürümek istiyor.
Ben park koşmak istiyor.
Ben çok uçmak istiyor gibi.
Ben oyun kazanmak istiyor.
Ben para kazanmak istiyor.
Ben iş bitirmek istiyor.
Ben iş başlamak istiyor.
Ben iş bulmak istiyor.
Ben çözüm bulmak istiyor.
Ben anlatmak istiyor.
Ben anlamak istiyor.
Ben düşünmek istiyor.
Ben söylemek istiyor.
Ben sormak istiyor.
Ben bilmek istiyor.
Ben beklemek istiyor.
Ben kalmak istiyor.
Ben gitmek istiyor.
Ben bakmak istiyor.
Ben seçmek istiyor.
Ben kullanmak istiyor.
Ben çıkmak istiyor.
Ben girmek istiyor.
Ben kalkmak istiyor.
Ben yatmak istiyor.
Ben dinlemek istiyor.
Ben çalışmak istiyor.
Ben korumak istiyor.
Ben taşımak istiyor.
Ben bırakmak istiyor.
Ben vermek istiyor.
Ben almak istiyor.
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_synthetic_sentence_sets.py -q -p no:warnings`
Expected: `6 passed` (hiçbiri atlanmamış: `train.csv` ve `eval.csv` `data/users/FurkanV1/` altında var).

- [ ] **Step 6: Commit**

```bash
git add -f datasets/sentence_sets/sentetik_olcum.txt datasets/sentence_sets/sentetik_egitim.txt
git add tests/test_synthetic_sentence_sets.py
git commit -m "feat: add synthetic sentence lists for measuring and training sentence accuracy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Sentetik cümle sesi (`sentence_synth`)

**Files:**
- Create: `src/training/sentence_synth.py`
- Test: `tests/test_sentence_synth.py`

**Interfaces:**
- Produces (Task 4 ve 5 kullanır):
  - `sentence_words(sentence: str) -> List[str]`
  - `read_sentences(path) -> List[str]`
  - `load_takes(csv_path, user_dir) -> Dict[str, List[str]]` (anahtar normalize transkript, değer mutlak wav yolları)
  - `usable_sentences(sentences, takes_by_word) -> Tuple[List[str], List[str]]` (kullanılabilir, atlanan)
  - `trim_word(audio, top_db=30) -> np.ndarray`
  - `build_sentence(words, takes_by_word, rng, sr=16000, load=None) -> np.ndarray`
  - `MissingWordError(KeyError)`
  - `read_manifest(user_dir) -> List[Dict[str, str]]` (`file_path` kullanıcı dizinine göreli, `transcription`)
  - `ensure_eval_set(user_dir, sentences_path, seed=config.SEED, per_sentence=2, sr=16000) -> Tuple[List[Dict], List[str]]`
  - Sabitler: `EDGE_S=0.2`, `GAP_MIN_S=0.08`, `GAP_MAX_S=0.35`, `PEAK=0.9`, `SYNTH_EVAL_DIR="synthetic_eval"`

- [ ] **Step 1: Write the failing tests**

`tests/test_sentence_synth.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_sentence_synth.py -q -p no:warnings`
Expected: FAIL — `ImportError: cannot import name 'sentence_synth' from 'src.training'`.

- [ ] **Step 3: Write the implementation**

`src/training/sentence_synth.py`:

```python
# -*- coding: utf-8 -*-
"""
Kelime kayıtlarından sentetik cümle sesi üretir (alt proje B2).

Her kelime için o kelimenin kayıtlarından biri seçilir, baş/son sessizliği
kırpılır, tepe genliği eşitlenir ve kelimeler kısa, rastgele duraklarla
birleştirilir. Ölçüm sesleri eval.csv'den, eğitim sesleri train.csv'den
üretilir; böylece ölçülen ses eğitimde hiç duyulmaz.
"""
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


def read_manifest(user_dir) -> List[Dict[str, str]]:
    """synthetic_eval/manifest.csv satırları (file_path kullanıcı dizinine göre göreli); yoksa []."""
    path = Path(user_dir) / SYNTH_EVAL_DIR / MANIFEST
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8")
    return df.to_dict("records")


def ensure_eval_set(
    user_dir,
    sentences_path,
    seed: int = config.SEED,
    per_sentence: int = 2,
    sr: int = config.ORNEKLEME_ORANI,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Sabit sentetik ölçüm seslerini döndürür; yoksa eval.csv kayıtlarından bir kez üretir.
    Manifest en son yazılır: üretim yarıda kalırsa bir sonraki çağrı baştan üretir.
    """
    import soundfile as sf

    user_dir = Path(user_dir)
    out_dir = user_dir / SYNTH_EVAL_DIR
    rows = read_manifest(user_dir)
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
    pd.DataFrame(rows, columns=["file_path", "transcription"]).to_csv(
        out_dir / MANIFEST, index=False, encoding="utf-8"
    )
    return rows, skipped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_sentence_synth.py -q -p no:warnings`
Expected: `15 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/training/sentence_synth.py tests/test_sentence_synth.py
git commit -m "feat: build synthetic sentence audio from trimmed word recordings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `evaluate_sentences`'e `--model` ve rapor öneki

**Files:**
- Modify: `src/cli/evaluate_sentences.py` (`write_report`, `_load_asr`, `main`)
- Test: `tests/test_evaluate_sentences.py` (sona ekleme)

**Interfaces:**
- Produces (Task 4 kullanır):
  - `model_key(user_id: str, settings_model: Optional[str], model_name: Optional[str] = None) -> str`
  - `_load_asr(user_id: str, model_name: Optional[str] = None)` → `ASRSystem`; model bulunamazsa `SystemExit("Model bulunamadı: <ad>")`
  - `write_report(report, reports_dir, now=None, prefix="sentence_eval") -> Path`
  - Mevcut: `evaluate(rows, transcribe, correct)`, `score`, `normalize`, `print_summary`, `worst`

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluate_sentences.py` dosyasının sonuna ekle:

```python


def test_model_key_defaults_to_the_live_model_choice():
    assert ev.model_key("FurkanV1", "whisper") == "FurkanV1_whisper"
    assert ev.model_key("FurkanV1", "wav2vec2") == "FurkanV1"
    assert ev.model_key("FurkanV1", None) == "FurkanV1"


def test_model_key_uses_the_given_model_folder():
    assert ev.model_key("FurkanV1", "whisper", "FurkanV1_whisper_cumle") == "FurkanV1_whisper_cumle"


def test_write_report_accepts_a_prefix(tmp_path):
    out = ev.write_report({}, tmp_path, now=datetime(2026, 10, 22, 1, 2, 3), prefix="synthetic_eval")
    assert out == tmp_path / "synthetic_eval_20261022-010203.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_evaluate_sentences.py -q -p no:warnings`
Expected: `3 failed, 10 passed` — `AttributeError: ... has no attribute 'model_key'` (2) ve `TypeError: write_report() got an unexpected keyword argument 'prefix'` (1).

- [ ] **Step 3: Apply the implementation**

Aşağıdaki yamayı `git apply` ile uygula (repo kökünden):

```bash
git apply <<'PATCH'
diff --git a/src/cli/evaluate_sentences.py b/src/cli/evaluate_sentences.py
index f310acd7..a3b2e798 100644
--- a/src/cli/evaluate_sentences.py
+++ b/src/cli/evaluate_sentences.py
@@ -7,7 +7,7 @@ Her cümle kaydı için iki çıktı karşılaştırılır:
   düzeltilmiş : ham + SentenceCorrector (Furkan'ın gördüğü cümle)
 
 Kullanım:
-    python -m src.cli.evaluate_sentences --user FurkanV1 [--device cpu]
+    python -m src.cli.evaluate_sentences --user FurkanV1 [--model <klasör adı>] [--device cpu]
 """
 import argparse
 import csv
@@ -112,11 +112,13 @@ def worst(rows: List[Dict], n: int = 10) -> List[Dict]:
     return sorted(rows, key=lambda r: (r["raw_wer"], r["corrected_wer"]), reverse=True)[:n]
 
 
-def write_report(report: Dict, reports_dir: Path, now: Optional[datetime] = None) -> Path:
+def write_report(
+    report: Dict, reports_dir: Path, now: Optional[datetime] = None, prefix: str = "sentence_eval"
+) -> Path:
     now = now or datetime.now()
     reports_dir = Path(reports_dir)
     reports_dir.mkdir(parents=True, exist_ok=True)
-    out = reports_dir / f"sentence_eval_{now.strftime('%Y%m%d-%H%M%S')}.json"
+    out = reports_dir / f"{prefix}_{now.strftime('%Y%m%d-%H%M%S')}.json"
     out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
     return out
 
@@ -140,21 +142,33 @@ def print_summary(report: Dict) -> None:
         print(f"\nAtlanan kayıt: {len(report['skipped'])}")
 
 
-def _load_asr(user_id: str):
+def model_key(user_id: str, settings_model: Optional[str], model_name: Optional[str] = None) -> str:
+    """Ölçülecek model klasörünün adı; --model verilmezse api.get_asr ile aynı seçim."""
+    if model_name:
+        return model_name
+    return f"{user_id}_whisper" if settings_model == "whisper" else user_id
+
+
+def _load_asr(user_id: str, model_name: Optional[str] = None):
     # Ağır importlar (transformers, peft) yalnızca gerçek ölçümde yüklenir
     from src.core.asr import ASRSystem
     from src.services.model_service import ModelService
     from src.services.settings_service import SettingsService
 
-    # api.get_asr ile aynı seçim: ayar "whisper" ise Whisper adaptörü
     settings = SettingsService().load_settings(user_id)
-    model_key = f"{user_id}_whisper" if settings.get("model") == "whisper" else user_id
-    return ASRSystem(model_name=ModelService.find_personalized_model(model_key), user_id=user_id)
+    key = model_key(user_id, settings.get("model"), model_name)
+    model_path = ModelService.find_personalized_model(key)
+    if model_path is None:
+        raise SystemExit(f"Model bulunamadı: {key}")
+    # user_id sözlüğü (metadata_words.csv) seçer; model klasöründen bağımsız
+    return ASRSystem(model_name=model_path, user_id=user_id)
 
 
 def main(argv: Optional[List[str]] = None) -> int:
     parser = argparse.ArgumentParser(description="Cümle kayıtlarında ASR doğruluğunu ölçer.")
     parser.add_argument("--user", default="FurkanV1")
+    parser.add_argument("--model", default=None,
+                        help="data/models/personalized_models altındaki klasör adı (varsayılan: canlı model)")
     parser.add_argument("--device", choices=["auto", "cpu"], default="auto",
                         help="cpu: API GPU'yu kullanırken bellek yetmezse")
     args = parser.parse_args(argv)
@@ -170,7 +184,7 @@ def main(argv: Optional[List[str]] = None) -> int:
 
     from src.core.sentence_corrector import SentenceCorrector
 
-    asr = _load_asr(args.user)
+    asr = _load_asr(args.user, args.model)
     corrector = SentenceCorrector(asr.user_vocabulary, Path(config.LEXICON_PATH))
     report = evaluate(
         rows,
@@ -179,6 +193,7 @@ def main(argv: Optional[List[str]] = None) -> int:
     )
     report["user_id"] = args.user
     report["model_type"] = asr.model_type
+    report["model"] = args.model
 
     print_summary(report)
     out = write_report(report, Path(config.REPORTS_DIR))
PATCH
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_evaluate_sentences.py -q -p no:warnings`
Expected: `13 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cli/evaluate_sentences.py tests/test_evaluate_sentences.py
git commit -m "feat: let evaluate_sentences measure a chosen model folder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Ölçüm aracı `evaluate_synthetic`

**Files:**
- Create: `src/cli/evaluate_synthetic.py`
- Test: `tests/test_evaluate_synthetic.py`

**Interfaces:**
- Consumes: Task 2 `ensure_eval_set`, `load_takes`, `sentence_words`; Task 3 `_load_asr(user_id, model_name)`, `write_report(..., prefix=)`, `evaluate`, `print_summary`.
- Produces (Task 6 ve 7 çalıştırır):
  - CLI `python -m src.cli.evaluate_synthetic --user FurkanV1 [--model <ad>] [--full-words] [--device cpu]`
  - `decide(word_wer, sentence_wer, threshold=0.05) -> "egitime_gec" | "dur"`
  - Rapor `reports/synthetic_eval_<YYYYmmdd-HHMMSS>.json`: `model`, `word_wer`, `sentence_wer`, `difference`, `decision`, `skipped_sentences`, `summary` (`kelime`, `sentetik_cumle`, `toplam`), `rows`, `skipped`, `full_words`, `model_type`, `user_id`

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluate_synthetic.py`:

```python
# -*- coding: utf-8 -*-
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import soundfile as sf

from src.cli import evaluate_synthetic as es

SR = 16000


def test_decide_moves_to_training_at_five_points():
    assert es.decide(0.15, 0.20) == "egitime_gec"
    assert es.decide(0.15, 0.30) == "egitime_gec"


def test_decide_stops_below_five_points():
    assert es.decide(0.15, 0.199) == "dur"
    assert es.decide(0.20, 0.15) == "dur"


def test_sentence_rows_point_into_the_user_dir(tmp_path):
    rows = es.sentence_rows([{"file_path": "synthetic_eval/s001_1.wav", "transcription": "Kedi süt istedi."}], tmp_path)
    assert rows == [{
        "audio_path": str(tmp_path / "synthetic_eval/s001_1.wav"),
        "file_path": "synthetic_eval/s001_1.wav",
        "expected": "Kedi süt istedi.",
        "set_file": "sentetik_cumle",
    }]


def test_word_rows_filters_to_the_given_words():
    takes = {"kedi": ["/a/kedi1.wav", "/a/kedi2.wav"], "süt": ["/a/süt1.wav"], "uçak": ["/a/uçak1.wav"]}
    rows = es.word_rows(takes, {"kedi", "süt"})
    assert [(r["expected"], r["audio_path"]) for r in rows] == [
        ("kedi", "/a/kedi1.wav"), ("kedi", "/a/kedi2.wav"), ("süt", "/a/süt1.wav"),
    ]
    assert {r["set_file"] for r in rows} == {"kelime"}


def test_word_rows_without_filter_returns_all():
    takes = {"kedi": ["/a/kedi1.wav"], "uçak": ["/a/uçak1.wav"]}
    assert len(es.word_rows(takes)) == 2


def test_build_report_adds_difference_and_decision():
    result = {
        "summary": {
            "kelime": {"raw": {"wer": 0.1}},
            "sentetik_cumle": {"raw": {"wer": 0.3}},
        },
        "rows": [],
        "skipped": [],
    }
    report = es.build_report(result, ["Kedi uçak istedi."], "FurkanV1_whisper_cumle")
    assert report["difference"] == pytest.approx(0.2)
    assert report["decision"] == "egitime_gec"
    assert report["model"] == "FurkanV1_whisper_cumle"
    assert report["skipped_sentences"] == ["Kedi uçak istedi."]
    assert report["summary"] is result["summary"]


def test_main_without_eval_csv_exits_with_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(es.config, "BASE_PATH", str(tmp_path))
    assert es.main(["--user", "Yok"]) == 1
    assert "eval.csv yok" in capsys.readouterr().out


def _tone(path, seconds=0.3):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SR)) / SR
    pad = np.zeros(SR // 4, dtype=np.float32)
    sf.write(str(path), np.concatenate([pad, 0.3 * np.sin(2 * np.pi * 220 * t), pad]), SR)


class _FakeAsr:
    user_vocabulary = []
    model_type = "whisper"

    def transcribe(self, path, fuzzy_correct=True):
        # Kelimeleri doğru, cümleleri hep yarım tanıyan bir model
        if "synthetic_eval" in path:
            return ("kedi", 0.9)
        return (path.split(os.sep)[-2], 0.9)


def test_main_measures_words_and_sentences_and_writes_report(tmp_path, monkeypatch, capsys):
    user = tmp_path / "users" / "U"
    for word in ("kedi", "süt", "istedi", "uçak"):
        _tone(user / "words" / word / "rep1.wav")
    (user / "eval.csv").write_text(
        "file_path,transcript\n" + "".join(f"words/{w}/rep1.wav,{w}\n" for w in ("kedi", "süt", "istedi", "uçak")),
        encoding="utf-8",
    )
    sentences = tmp_path / "sentetik_olcum.txt"
    sentences.write_text("Kedi süt istedi.\n", encoding="utf-8")
    reports = tmp_path / "reports"

    monkeypatch.setattr(es.config, "BASE_PATH", str(tmp_path / "users"))
    monkeypatch.setattr(es.config, "REPORTS_DIR", str(reports))
    monkeypatch.setattr(es, "EVAL_SENTENCES", sentences)
    monkeypatch.setattr(es.ev, "_load_asr", lambda user_id, model_name=None: _FakeAsr())

    class _Corrector:
        def __init__(self, vocabulary, lexicon_path):
            pass

        def correct(self, text):
            return text

    import src.core.sentence_corrector as sc
    monkeypatch.setattr(sc, "SentenceCorrector", _Corrector)

    assert es.main(["--user", "U", "--model", "U_whisper_cumle"]) == 0

    out = capsys.readouterr().out
    assert "Karar: egitime_gec" in out
    report = json.loads(next(reports.glob("synthetic_eval_*.json")).read_text(encoding="utf-8"))
    assert report["summary"]["kelime"]["count"] == 3  # uçak cümlede yok
    assert report["summary"]["sentetik_cumle"]["count"] == 2
    assert report["word_wer"] == 0.0
    assert report["sentence_wer"] == pytest.approx(2 / 3)
    assert report["model"] == "U_whisper_cumle"


def test_main_full_words_measures_every_eval_word(tmp_path, monkeypatch):
    user = tmp_path / "users" / "U"
    for word in ("kedi", "süt", "istedi", "uçak"):
        _tone(user / "words" / word / "rep1.wav")
    (user / "eval.csv").write_text(
        "file_path,transcript\n" + "".join(f"words/{w}/rep1.wav,{w}\n" for w in ("kedi", "süt", "istedi", "uçak")),
        encoding="utf-8",
    )
    sentences = tmp_path / "sentetik_olcum.txt"
    sentences.write_text("Kedi süt istedi.\n", encoding="utf-8")
    reports = tmp_path / "reports"
    monkeypatch.setattr(es.config, "BASE_PATH", str(tmp_path / "users"))
    monkeypatch.setattr(es.config, "REPORTS_DIR", str(reports))
    monkeypatch.setattr(es, "EVAL_SENTENCES", sentences)
    monkeypatch.setattr(es.ev, "_load_asr", lambda user_id, model_name=None: _FakeAsr())
    import src.core.sentence_corrector as sc
    monkeypatch.setattr(sc, "SentenceCorrector", lambda vocabulary, lexicon_path: type("C", (), {"correct": staticmethod(lambda t: t)})())

    assert es.main(["--user", "U", "--full-words"]) == 0

    report = json.loads(next(reports.glob("synthetic_eval_*.json")).read_text(encoding="utf-8"))
    assert report["summary"]["kelime"]["count"] == 4
    assert report["full_words"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_evaluate_synthetic.py -q -p no:warnings`
Expected: FAIL — `ImportError: cannot import name 'evaluate_synthetic' from 'src.cli'`.

- [ ] **Step 3: Write the implementation**

`src/cli/evaluate_synthetic.py`:

```python
# -*- coding: utf-8 -*-
"""
Sentetik cümlelerde ve aynı kelimelerin tekil kayıtlarında ASR doğruluğunu
karşılaştırır (alt proje B2). Cümle WER'i kelime WER'inden en az 5 puan
kötüyse karar "egitime_gec", değilse "dur".

Kullanım:
    python -m src.cli.evaluate_synthetic --user FurkanV1 [--model <klasör adı>] [--full-words] [--device cpu]
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import config
from src.cli import evaluate_sentences as ev
from src.training import sentence_synth as ss

EVAL_SENTENCES = Path("datasets") / "sentence_sets" / "sentetik_olcum.txt"
WORDS = "kelime"
SENTENCES = "sentetik_cumle"
THRESHOLD = 0.05


def decide(word_wer: float, sentence_wer: float, threshold: float = THRESHOLD) -> str:
    # Yuvarlama: 0.20 - 0.15 kayan noktada 0.04999... çıkabilir
    return "egitime_gec" if round(sentence_wer - word_wer, 6) >= threshold else "dur"


def sentence_rows(manifest_rows: List[Dict[str, str]], user_dir: Path) -> List[Dict[str, str]]:
    return [
        {
            "audio_path": str(Path(user_dir) / r["file_path"]),
            "file_path": r["file_path"],
            "expected": r["transcription"],
            "set_file": SENTENCES,
        }
        for r in manifest_rows
    ]


def word_rows(takes_by_word: Dict[str, List[str]], words: Optional[set] = None) -> List[Dict[str, str]]:
    """eval.csv'nin tekil kelime kayıtları; words verilirse yalnızca o kelimeler."""
    return [
        {"audio_path": path, "file_path": path, "expected": word, "set_file": WORDS}
        for word, paths in sorted(takes_by_word.items())
        if words is None or word in words
        for path in paths
    ]


def build_report(result: Dict, skipped: List[str], model: Optional[str]) -> Dict:
    word_wer = result["summary"][WORDS]["raw"]["wer"]
    sentence_wer = result["summary"][SENTENCES]["raw"]["wer"]
    return {
        "model": model,
        "word_wer": word_wer,
        "sentence_wer": sentence_wer,
        "difference": sentence_wer - word_wer,
        "decision": decide(word_wer, sentence_wer),
        "skipped_sentences": skipped,
        **result,
    }


def print_decision(report: Dict) -> None:
    print(
        f"\nKelime WER: {report['word_wer']:.1%}  Cümle WER: {report['sentence_wer']:.1%}  "
        f"Fark: {report['difference']:+.1%}"
    )
    print(f"Karar: {report['decision']}")
    if report["skipped_sentences"]:
        print(f"Atlanan cümle (kaydı olmayan kelime): {len(report['skipped_sentences'])}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sentetik cümlelerde ASR doğruluğunu ölçer.")
    parser.add_argument("--user", default="FurkanV1")
    parser.add_argument("--model", default=None,
                        help="data/models/personalized_models altındaki klasör adı (varsayılan: canlı model)")
    parser.add_argument("--full-words", action="store_true",
                        help="kelime kıyası için eval.csv'nin tamamı")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = parser.parse_args(argv)

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    user_dir = Path(config.BASE_PATH) / args.user
    if not (user_dir / "eval.csv").exists():
        print(f"eval.csv yok: {user_dir}")
        return 1

    manifest, skipped = ss.ensure_eval_set(user_dir, EVAL_SENTENCES)
    if not manifest:
        print("Üretilebilen sentetik cümle yok.")
        return 1
    takes = ss.load_takes(user_dir / "eval.csv", user_dir)
    words = None if args.full_words else {
        w for r in manifest for w in ss.sentence_words(r["transcription"])
    }
    rows = word_rows(takes, words) + sentence_rows(manifest, user_dir)

    from src.core.sentence_corrector import SentenceCorrector

    asr = ev._load_asr(args.user, args.model)
    corrector = SentenceCorrector(asr.user_vocabulary, Path(config.LEXICON_PATH))
    result = ev.evaluate(
        rows,
        transcribe=lambda path: asr.transcribe(path, fuzzy_correct=False)[0],
        correct=corrector.correct,
    )
    report = build_report(result, skipped, args.model)
    report["user_id"] = args.user
    report["model_type"] = asr.model_type
    report["full_words"] = args.full_words

    ev.print_summary(report)
    print_decision(report)
    out = ev.write_report(report, Path(config.REPORTS_DIR), prefix="synthetic_eval")
    print(f"\nRapor: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_evaluate_synthetic.py -q -p no:warnings`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cli/evaluate_synthetic.py tests/test_evaluate_synthetic.py
git commit -m "feat: add synthetic sentence accuracy measurement with a training decision

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Eğitime sentetik cümleler ve devam eğitimi

**Files:**
- Modify: `config.py` (yeni sabitler, `GRADIENT_ACCUMULATION_STEPS` satırının altına)
- Modify: `src/cli/train_whisper_adapter.py`
- Test: `tests/test_whisper_sentence_training.py`

**Interfaces:**
- Consumes: Task 2 `load_takes`, `read_sentences`, `usable_sentences`, `build_sentence`, `sentence_words`, `read_manifest`.
- Produces (Task 7 çalıştırır):
  - `config.SYNTH_SENTENCE_REPEATS = 2`, `CONTINUE_EPOCHS = 10`, `CONTINUE_LEARNING_RATE = 1e-5`, `CONTINUE_EARLY_STOPPING_PATIENCE = 3`
  - `resolve_output_dir(user_id, output_name=None, init_from=None) -> Path`
  - `training_schedule(continuing: bool) -> (epochs, lr, patience)`
  - `build_train_frame(words_df, sentences, takes_by_word, repeats=None) -> (DataFrame[file_path, transcript, kind], skipped)`
  - `build_eval_frame(words_df, manifest_rows, user_dir) -> DataFrame`
  - `parse_args(argv=None)` (`user_id`, `--base_model`, `--init-from`, `--output-name`, `--no-sentences`)
  - `write_side_report(training_data, output_name, reports_dir=None, now=None) -> Path`
  - `WhisperPersonalizedTrainer(user_id, base_model_path=None, init_from=None, output_name=None, use_sentences=True)`; `TrainingService`'in `WhisperPersonalizedTrainer(user_id=...)` çağrısı değişmeden çalışır.

- [ ] **Step 1: Write the failing tests**

`tests/test_whisper_sentence_training.py`:

```python
# -*- coding: utf-8 -*-
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest

import config
from src.cli import train_whisper_adapter as tw

LIVE = tw.PERSONALIZED_ROOT / "FurkanV1_whisper"


def test_default_output_is_the_live_whisper_folder():
    # Self-learning (TrainingService) bugünkü gibi canlı klasöre eğitir
    assert tw.resolve_output_dir("FurkanV1") == LIVE


def test_output_name_writes_to_a_separate_folder():
    out = tw.resolve_output_dir("FurkanV1", "FurkanV1_whisper_cumle", str(LIVE))
    assert out == tw.PERSONALIZED_ROOT / "FurkanV1_whisper_cumle"


def test_output_name_cannot_be_the_live_folder():
    with pytest.raises(ValueError):
        tw.resolve_output_dir("FurkanV1", "FurkanV1_whisper")


def test_continuing_requires_an_output_name():
    with pytest.raises(ValueError):
        tw.resolve_output_dir("FurkanV1", None, str(LIVE))


def test_output_cannot_be_the_init_folder():
    init = tw.PERSONALIZED_ROOT / "FurkanV1_whisper_cumle"
    with pytest.raises(ValueError):
        tw.resolve_output_dir("FurkanV1", "FurkanV1_whisper_cumle", str(init))


def test_training_schedule_continuing_uses_short_low_lr_settings():
    assert tw.training_schedule(True) == (10, 1e-5, 3)
    assert tw.training_schedule(False) == (
        config.NUM_FINETUNE_EPOCHS, config.FINETUNE_LEARNING_RATE, config.EARLY_STOPPING_PATIENCE,
    )


def test_build_train_frame_adds_each_usable_sentence_repeats_times():
    words = pd.DataFrame({"file_path": ["/w/kedi1.wav"], "transcript": ["kedi"], "quality_score": [90]})
    takes = {"kedi": ["/w/kedi1.wav"], "süt": ["/w/süt1.wav"], "istedi": ["/w/istedi1.wav"]}
    df, skipped = tw.build_train_frame(words, ["Kedi süt istedi.", "Kedi uçak istedi."], takes, repeats=2)

    assert list(df.columns) == ["file_path", "transcript", "kind"]
    assert df["kind"].tolist() == ["word", "sentence", "sentence"]
    assert df[df["kind"] == "sentence"]["transcript"].tolist() == ["Kedi süt istedi."] * 2
    assert df[df["kind"] == "sentence"]["file_path"].tolist() == ["", ""]
    assert skipped == ["Kedi uçak istedi."]


def test_build_train_frame_default_repeats_from_config():
    words = pd.DataFrame({"file_path": [], "transcript": []})
    takes = {"kedi": ["a"], "süt": ["b"], "istedi": ["c"]}
    df, _ = tw.build_train_frame(words, ["Kedi süt istedi."], takes)
    assert len(df) == config.SYNTH_SENTENCE_REPEATS


def test_build_eval_frame_appends_synthetic_eval_audio(tmp_path):
    words = pd.DataFrame({"file_path": ["/w/kedi1.wav"], "transcript": ["kedi"]})
    manifest = [{"file_path": "synthetic_eval/s001_1.wav", "transcription": "Kedi süt istedi."}]
    df = tw.build_eval_frame(words, manifest, tmp_path)
    assert df["file_path"].tolist() == ["/w/kedi1.wav", str(tmp_path / "synthetic_eval/s001_1.wav")]
    assert df["transcript"].tolist() == ["kedi", "Kedi süt istedi."]
    assert set(df["kind"]) == {"word"}


def test_build_eval_frame_without_manifest_is_words_only():
    words = pd.DataFrame({"file_path": ["/w/kedi1.wav"], "transcript": ["kedi"]})
    assert len(tw.build_eval_frame(words, [], Path("."))) == 1


def test_parse_args_defaults_match_todays_cli():
    args = tw.parse_args(["FurkanV1"])
    assert args.init_from is None and args.output_name is None and args.no_sentences is False


def test_parse_args_continue_options():
    args = tw.parse_args([
        "FurkanV1", "--init-from", str(LIVE), "--output-name", "FurkanV1_whisper_cumle", "--no-sentences",
    ])
    assert args.init_from == str(LIVE)
    assert args.output_name == "FurkanV1_whisper_cumle"
    assert args.no_sentences is True


def test_side_report_is_not_picked_up_as_the_live_training_report(tmp_path):
    out = tw.write_side_report({"final_wer": 0.1}, "FurkanV1_whisper_cumle", tmp_path,
                               now=datetime(2026, 10, 22, 1, 2, 3))
    assert out.name == "whisper_side_FurkanV1_whisper_cumle_20261022_010203.json"
    assert json.loads(out.read_text(encoding="utf-8")) == {"final_wer": 0.1}
    assert not list(tmp_path.glob("training_FurkanV1_*.json"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_whisper_sentence_training.py -q -p no:warnings`
Expected: FAIL — `AttributeError: module 'src.cli.train_whisper_adapter' has no attribute 'PERSONALIZED_ROOT'` (modül yüklenir, ilk erişimde düşer; torch/transformers importu birkaç saniye sürer).

- [ ] **Step 3: Apply the implementation**

```bash
git apply <<'PATCH'
diff --git a/config.py b/config.py
index 621b159e..f2167d9b 100755
--- a/config.py
+++ b/config.py
@@ -76,6 +76,12 @@ NUM_FINETUNE_EPOCHS = 100          # Maksimum limit
 FINETUNE_BATCH_SIZE = 8            # 4'ten 8'e çıkarıldı
 GRADIENT_ACCUMULATION_STEPS = 4    # Toplam batch size 32 (8 * 4) korunuyor
 
+# --- Sentetik cümle eğitimi (B2) ---
+SYNTH_SENTENCE_REPEATS = 2              # Her eğitim cümlesi turda kaç kez (her seferinde farklı ses)
+CONTINUE_EPOCHS = 10                    # --init-from ile devam eğitiminde en fazla tur
+CONTINUE_LEARNING_RATE = 1e-5           # Devam eğitiminde mevcut bilgiyi bozmamak için düşük
+CONTINUE_EARLY_STOPPING_PATIENCE = 3
+
 # --- Anlamli Cumle Kurma Ayarlari ---
 # Kelime tamamlama ve duzeltme esigi (0-1)
 # 0.6: %60 benzerlik gosteren kelimeyi kullanicinin sozrugunden "en yakin" olanla degistirir.
diff --git a/src/cli/train_whisper_adapter.py b/src/cli/train_whisper_adapter.py
index cc74420c..19e0fccb 100644
--- a/src/cli/train_whisper_adapter.py
+++ b/src/cli/train_whisper_adapter.py
@@ -12,8 +12,10 @@ if str(root_path) not in sys.path:
 
 import argparse
 import dataclasses
+import json
 import logging
 import platform
+import random
 import time
 from datetime import datetime
 
@@ -23,7 +25,7 @@ import soundfile as sf
 import torch
 import evaluate
 from datasets import Dataset
-from peft import LoraConfig, get_peft_model
+from peft import LoraConfig, PeftModel, get_peft_model
 from transformers import (
     EarlyStoppingCallback,
     Seq2SeqTrainer,
@@ -34,6 +36,7 @@ from transformers import (
 
 import config
 from src.services.reporting_service import ReportingService
+from src.training import sentence_synth as ss
 from src.utils.utils import (
     build_augment_pipeline,
     normalize_turkish_text,
@@ -42,6 +45,80 @@ from src.utils.utils import (
 )
 
 
+PERSONALIZED_ROOT = Path("data/models/personalized_models")
+TRAIN_SENTENCES = Path("datasets") / "sentence_sets" / "sentetik_egitim.txt"
+
+
+def resolve_output_dir(user_id, output_name=None, init_from=None) -> Path:
+    """
+    Çıktı klasörü. Varsayılan canlı Whisper klasörü (self-learning bugünkü gibi).
+    Devam eğitimi (--init-from) canlı klasöre ya da başladığı klasöre asla yazmaz.
+    """
+    live = PERSONALIZED_ROOT / f"{user_id}_whisper"
+    if output_name is None:
+        if init_from is not None:
+            raise ValueError("--init-from ile --output-name verilmeli (canlı klasöre yazılmaz)")
+        return live
+    out = PERSONALIZED_ROOT / output_name
+    if out.resolve() == live.resolve():
+        raise ValueError(f"--output-name canlı klasör olamaz: {live}")
+    if init_from is not None and out.resolve() == Path(init_from).resolve():
+        raise ValueError("--output-name, --init-from ile aynı klasör olamaz")
+    return out
+
+
+def training_schedule(continuing: bool):
+    """(en fazla tur, öğrenme hızı, erken durdurma sabrı)."""
+    if continuing:
+        return config.CONTINUE_EPOCHS, config.CONTINUE_LEARNING_RATE, config.CONTINUE_EARLY_STOPPING_PATIENCE
+    return config.NUM_FINETUNE_EPOCHS, config.FINETUNE_LEARNING_RATE, config.EARLY_STOPPING_PATIENCE
+
+
+def build_train_frame(words_df, sentences, takes_by_word, repeats=None):
+    """
+    Kelime satırları + her kullanılabilir cümleden `repeats` satır. Cümle
+    satırının file_path'i boştur; sesi transform sırasında üretilir.
+    Döner: (DataFrame[file_path, transcript, kind], atlanan cümleler)
+    """
+    repeats = config.SYNTH_SENTENCE_REPEATS if repeats is None else repeats
+    words = words_df[["file_path", "transcript"]].copy()
+    words["kind"] = "word"
+    usable, skipped = ss.usable_sentences(sentences, takes_by_word)
+    sentence_rows = pd.DataFrame(
+        [{"file_path": "", "transcript": s, "kind": "sentence"} for s in usable for _ in range(repeats)],
+        columns=["file_path", "transcript", "kind"],
+    )
+    return pd.concat([words, sentence_rows], ignore_index=True), skipped
+
+
+def build_eval_frame(words_df, manifest_rows, user_dir):
+    """eval.csv kelimeleri + sabit sentetik ölçüm sesleri (varsa)."""
+    words = words_df[["file_path", "transcript"]].copy()
+    words["kind"] = "word"
+    synthetic = pd.DataFrame(
+        [{"file_path": str(Path(user_dir) / r["file_path"]), "transcript": r["transcription"], "kind": "word"}
+         for r in manifest_rows],
+        columns=["file_path", "transcript", "kind"],
+    )
+    return pd.concat([words, synthetic], ignore_index=True)
+
+
+def parse_args(argv=None):
+    parser = argparse.ArgumentParser(description="Kullanıcıya özel Whisper ASR modelini LoRA ile eğitir.")
+    parser.add_argument("user_id", type=str, help="Verisi kullanılacak ve modeli kişiselleştirilecek kullanıcının kimliği.")
+    parser.add_argument(
+        "--base_model", type=str, default=None,
+        help="İnce ayar için kullanılacak temel Whisper modelinin yolu. Varsayılan: config.WHISPER_MODEL_NAME.",
+    )
+    parser.add_argument("--init-from", type=str, default=None,
+                        help="LoRA ağırlıklarının yükleneceği adaptör klasörü (devam eğitimi).")
+    parser.add_argument("--output-name", type=str, default=None,
+                        help="data/models/personalized_models altında çıktı klasörünün adı.")
+    parser.add_argument("--no-sentences", action="store_true",
+                        help="Sentetik cümleleri eğitime ekleme.")
+    return parser.parse_args(argv)
+
+
 def setup_logging(user_id):
     """Whisper eğitimi için log dosyası oluşturur (train_adapter.py'nin CTC logging'iyle aynı desen)."""
     log_dir = Path(config.LOG_DIR)
@@ -93,12 +170,19 @@ class DataCollatorSpeechSeq2SeqWithPadding:
 
 
 class WhisperPersonalizedTrainer:
-    def __init__(self, user_id, base_model_path=None):
+    def __init__(self, user_id, base_model_path=None, init_from=None, output_name=None, use_sentences=True):
         self.user_id = user_id
         self.base_model_path = base_model_path or config.WHISPER_MODEL_NAME
         self.user_data_path = Path(config.BASE_PATH) / self.user_id
-        # Her zaman _whisper son ekiyle - canli Wav2Vec2 modeliyle asla ayni yolu paylasmaz.
-        self.output_dir = Path("data/models/personalized_models") / f"{self.user_id}_whisper"
+        self.init_from = Path(init_from) if init_from else None
+        self.output_name = output_name
+        # Varsayilan _whisper son ekli canli klasor; canli Wav2Vec2 modeliyle asla ayni yolu paylasmaz.
+        self.output_dir = resolve_output_dir(user_id, output_name, init_from)
+        self.use_sentences = use_sentences
+        self.num_epochs, self.learning_rate, self.patience = training_schedule(self.init_from is not None)
+        self.train_takes = {}
+        self.sentence_train_samples = 0
+        self.synthetic_eval_samples = 0
 
         self.device = "cuda" if torch.cuda.is_available() else "cpu"
         self.processor = None
@@ -119,14 +203,19 @@ class WhisperPersonalizedTrainer:
         self.model.generation_config.language = "tr"
         self.model.generation_config.task = "transcribe"
 
-        peft_config = LoraConfig(
-            r=config.ADAPTER_REDUCTION_FACTOR,
-            lora_alpha=config.ADAPTER_REDUCTION_FACTOR * 2,
-            target_modules=["q_proj", "v_proj", "k_proj"],
-            lora_dropout=0.1,
-            bias="none",
-        )
-        self.model = get_peft_model(self.model, peft_config)
+        if self.init_from is not None:
+            # Devam egitimi: canli adaptorun LoRA agirliklariyla baslar
+            print(f"📥 LoRA adaptörü yükleniyor: {self.init_from}")
+            self.model = PeftModel.from_pretrained(self.model, str(self.init_from), is_trainable=True)
+        else:
+            peft_config = LoraConfig(
+                r=config.ADAPTER_REDUCTION_FACTOR,
+                lora_alpha=config.ADAPTER_REDUCTION_FACTOR * 2,
+                target_modules=["q_proj", "v_proj", "k_proj"],
+                lora_dropout=0.1,
+                bias="none",
+            )
+            self.model = get_peft_model(self.model, peft_config)
 
         if config.GRADIENT_CHECKPOINTING:
             self.model.gradient_checkpointing_enable()
@@ -205,16 +294,31 @@ class WhisperPersonalizedTrainer:
         df = df[df["file_path"].apply(is_readable)]
 
         is_train = split == 'train'
+        if is_train:
+            df = self._with_sentences(df)
+        else:
+            manifest = ss.read_manifest(self.user_data_path)
+            df = build_eval_frame(df, manifest, self.user_data_path)
+            self.synthetic_eval_samples = len(manifest)
+            if manifest:
+                print(f"   🗣️  Sentetik ölçüm cümlesi: {len(manifest)}")
         augmenter = self.augmenter if is_train else None
         sample_rate = config.ORNEKLEME_ORANI
 
         # Mel ozellikleri on-the-fly hesaplanir: tum veri setini onceden map() ile
         # hesaplamak ~1.5 MB/ornek RAM tutar ve 15 GB'lik sunucuda OOM'a yol acar.
         # Bonus: augmentation her epoch'ta yeniden uygulanir.
+        train_takes = self.train_takes
+
         def transform(batch):
             input_features, labels = [], []
-            for path, transcript in zip(batch["file_path"], batch["transcript"]):
-                audio, _ = librosa.load(path, sr=sample_rate)
+            # DataLoader her isciye ayri random tohumu verir; cumleler isciler arasi farkli uretilir
+            rng = random.Random(random.random())
+            for path, transcript, kind in zip(batch["file_path"], batch["transcript"], batch["kind"]):
+                if kind == "sentence":
+                    audio = ss.build_sentence(ss.sentence_words(transcript), train_takes, rng, sr=sample_rate)
+                else:
+                    audio, _ = librosa.load(path, sr=sample_rate)
                 if augmenter:
                     audio = augmenter(samples=audio, sample_rate=sample_rate)
                 input_features.append(
@@ -223,12 +327,25 @@ class WhisperPersonalizedTrainer:
                 labels.append(self.processor.tokenizer(normalize_turkish_text(transcript)).input_ids)
             return {"input_features": input_features, "labels": labels}
 
-        dataset = Dataset.from_pandas(df[["file_path", "transcript"]], preserve_index=False)
+        dataset = Dataset.from_pandas(df[["file_path", "transcript", "kind"]], preserve_index=False)
         dataset.set_transform(transform)
 
         print(f"   📈 {split.upper()} veri seti boyutu: {len(dataset)} kayıt")
         return dataset
 
+    def _with_sentences(self, words_df):
+        """Eğitim kelimelerine sentetik cümleleri ekler (train.csv ve cümle dosyası varsa)."""
+        train_csv = self.user_data_path / "train.csv"
+        if not (self.use_sentences and TRAIN_SENTENCES.exists() and train_csv.exists()):
+            words = words_df[["file_path", "transcript"]].copy()
+            words["kind"] = "word"
+            return words
+        self.train_takes = ss.load_takes(train_csv, self.user_data_path)
+        df, skipped = build_train_frame(words_df, ss.read_sentences(TRAIN_SENTENCES), self.train_takes)
+        self.sentence_train_samples = int((df["kind"] == "sentence").sum())
+        print(f"   🗣️  Sentetik eğitim cümlesi satırı: {self.sentence_train_samples} (atlanan cümle: {len(skipped)})")
+        return df
+
     def compute_metrics(self, pred):
         pred_ids = pred.predictions
         label_ids = pred.label_ids
@@ -302,9 +419,9 @@ class WhisperPersonalizedTrainer:
             per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
             per_device_eval_batch_size=config.FINETUNE_BATCH_SIZE,
             gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
-            learning_rate=config.FINETUNE_LEARNING_RATE,
+            learning_rate=self.learning_rate,
             warmup_steps=config.WARMUP_STEPS,
-            num_train_epochs=config.NUM_FINETUNE_EPOCHS,
+            num_train_epochs=self.num_epochs,
             eval_strategy="epoch" if has_eval else "no",
             save_strategy="epoch",
             logging_strategy="epoch",
@@ -325,7 +442,7 @@ class WhisperPersonalizedTrainer:
 
         callbacks = []
         if has_eval:
-            callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE))
+            callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.patience))
 
         trainer = Seq2SeqTrainer(
             model=self.model,
@@ -359,19 +476,18 @@ class WhisperPersonalizedTrainer:
         # HF Trainer, early-stop tetiklendiginde kontrol bilgisini kolayca disari
         # vermiyor; NUM_FINETUNE_EPOCHS'a ulasmadan durmus olmasi early-stop'un
         # isaretidir (CTC trainer'daki acik early_stopped flag'inin esdegeri).
-        early_stopped = has_eval and epochs_completed < config.NUM_FINETUNE_EPOCHS
+        early_stopped = has_eval and epochs_completed < self.num_epochs
 
-        reporting_service = ReportingService()
         training_data = {
             "base_model": self.base_model_path,
-            "num_epochs": config.NUM_FINETUNE_EPOCHS,
+            "num_epochs": self.num_epochs,
             "batch_size": config.FINETUNE_BATCH_SIZE,
-            "learning_rate": config.FINETUNE_LEARNING_RATE,
+            "learning_rate": self.learning_rate,
             "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
             "adapter_reduction_factor": config.ADAPTER_REDUCTION_FACTOR,
             "warmup_steps": config.WARMUP_STEPS,
             "weight_decay": config.WEIGHT_DECAY,
-            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
+            "early_stopping_patience": self.patience,
             "use_augmentation": config.USE_AUGMENTATION,
             "mixed_precision": config.MIXED_PRECISION,
             "train_samples": self.train_samples,
@@ -395,12 +511,30 @@ class WhisperPersonalizedTrainer:
             "end_time": datetime.now().isoformat(),
             "duration_seconds": duration,
             "duration_formatted": f"{duration/60:.2f} dk",
+            "init_from": str(self.init_from) if self.init_from else None,
+            "output_name": self.output_dir.name,
+            "sentence_train_samples": self.sentence_train_samples,
+            "synthetic_eval_samples": self.synthetic_eval_samples,
         }
-        report_file = reporting_service.log_training_session(self.user_id, training_data)
+        if self.output_name is None:
+            report_file = ReportingService().log_training_session(self.user_id, training_data)
+        else:
+            # Ayri klasore egitilen model canli degil: training_<user>_*.json'a yazilirsa
+            # TrainingService.get_last_wer onu canli modelin WER'i sanar.
+            report_file = write_side_report(training_data, self.output_dir.name)
         print(f"📊 Training report saved: {report_file}")
         logger.info(f"Whisper training completed for user: {self.user_id}")
 
 
+def write_side_report(training_data, output_name, reports_dir=None, now=None) -> Path:
+    reports_dir = Path(reports_dir or config.REPORTS_DIR)
+    reports_dir.mkdir(parents=True, exist_ok=True)
+    now = now or datetime.now()
+    out = reports_dir / f"whisper_side_{output_name}_{now.strftime('%Y%m%d_%H%M%S')}.json"
+    out.write_text(json.dumps(training_data, ensure_ascii=False, indent=2), encoding="utf-8")
+    return out
+
+
 if __name__ == "__main__":
     if platform.system() == "Linux":
         import multiprocessing as mp
@@ -409,15 +543,15 @@ if __name__ == "__main__":
         except RuntimeError:
             pass
 
-    parser = argparse.ArgumentParser(description="Kullanıcıya özel Whisper ASR modelini LoRA ile eğitir.")
-    parser.add_argument("user_id", type=str, help="Verisi kullanılacak ve modeli kişiselleştirilecek kullanıcının kimliği.")
-    parser.add_argument(
-        "--base_model", type=str, default=None,
-        help="İnce ayar için kullanılacak temel Whisper modelinin yolu. Varsayılan: config.WHISPER_MODEL_NAME.",
-    )
-    args = parser.parse_args()
+    args = parse_args()
 
     set_seed(config.SEED)
 
-    trainer = WhisperPersonalizedTrainer(user_id=args.user_id, base_model_path=args.base_model)
+    trainer = WhisperPersonalizedTrainer(
+        user_id=args.user_id,
+        base_model_path=args.base_model,
+        init_from=args.init_from,
+        output_name=args.output_name,
+        use_sentences=not args.no_sentences,
+    )
     trainer.run()
PATCH
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_whisper_sentence_training.py tests/test_model_selection.py -q -p no:warnings`
Expected: `13 passed` + `test_model_selection.py`'nin tümü geçer (self-learning'in trainer çağrısı bozulmadı).

- [ ] **Step 5: Smoke-check the real trainer without training**

Model ve veri setini kurup bir cümle örneğinin özelliklerini üretir; eğitim başlatmaz. API GPU'yu kullanıyorsa bellek yetmeyebilir; o zaman `CUDA_VISIBLE_DEVICES=""` ile tekrar dene.

```bash
/home/mayasoft/app/venv/bin/python - <<'EOF'
from src.cli import train_whisper_adapter as tw
t = tw.WhisperPersonalizedTrainer(
    "FurkanV1",
    init_from="data/models/personalized_models/FurkanV1_whisper",
    output_name="FurkanV1_whisper_cumle_smoke",
)
t.load_model_and_processor()
trainable = sum(p.numel() for p in t.model.parameters() if p.requires_grad)
assert trainable > 0, "LoRA agirliklari egitilebilir degil"
train = t.prepare_dataset("train")
kinds = train.data.column("kind").to_pylist()  # set_transform'u atlayarak ham sutun
sample = train[kinds.index("sentence")]
print("trainable", trainable, "train rows", len(train), "sentence rows", t.sentence_train_samples)
print("sentence features", len(sample["input_features"]), "labels", len(sample["labels"]))
print("decoded", t.processor.tokenizer.decode(sample["labels"], skip_special_tokens=True))
evalset = t.prepare_dataset("eval")
print("eval rows", len(evalset), "synthetic eval", t.synthetic_eval_samples)
print("schedule", t.num_epochs, t.learning_rate, t.patience, "out", t.output_dir)
EOF
```

Expected: `trainable 39321600`; `train rows 8095` (6087 kelime + 2008 cümle); `sentence rows` = 2008 (1004 × 2; atlanan 0); `sentence features 128` (Whisper large-v3 mel kanalı); `decoded` küçük harfli bir eğitim cümlesi (ör. `doktor muayene yaptı`); `schedule 10 1e-05 3 out data/models/personalized_models/FurkanV1_whisper_cumle_smoke`; `synthetic eval` Task 6'dan önce 0. Bu adım klasör oluşturmaz (kayıt yalnızca `run()` sonunda); `ls data/models/personalized_models` çıktısında `_smoke` olmamalı.

- [ ] **Step 6: Run the whole suite**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests -q -p no:warnings --ignore=tests/test_utils.py`
Expected: `162 passed` (116 + 6 + 15 + 3 + 9 + 13).

- [ ] **Step 7: Commit**

```bash
git add config.py src/cli/train_whisper_adapter.py tests/test_whisper_sentence_training.py
git commit -m "feat: train Whisper on synthetic sentences and continue from an existing adapter

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Canlı modelin cümle doğruluğunu ölç (gerçek çalıştırma)

**Files:**
- Üretilir (git'e girmez, `data/` ignore): `data/users/FurkanV1/synthetic_eval/` (334 wav + `manifest.csv` + `skipped.txt`)
- Üretilir (git'e girmez): `reports/synthetic_eval_<zaman>.json`

**Interfaces:**
- Consumes: Task 4 CLI.
- Produces: karar (`egitime_gec` | `dur`) ve canlı modelin kelime/cümle WER'i; Task 7 bu karara bağlı.

- [ ] **Step 1: Check that no training is running**

Run: `ps aux | grep -E "train_(whisper_)?adapter" | grep -v grep; nvidia-smi --query-gpu=memory.used,memory.total --format=csv`
Expected: eğitim süreci yok. Varsa bitmesini bekle (self-learning eğitimi saatler sürer; kullanıcıya bildir).

- [ ] **Step 2: Run the measurement on the live model**

Run: `/home/mayasoft/app/venv/bin/python -m src.cli.evaluate_synthetic --user FurkanV1 > /tmp/synthetic_eval_live.log 2>&1; tail -40 /tmp/synthetic_eval_live.log`
Expected: tablo `kelime`, `sentetik_cumle`, `toplam` satırlarıyla; ardından `Kelime WER: … Cümle WER: … Fark: …` ve `Karar: egitime_gec` ya da `Karar: dur`; `Rapor: reports/synthetic_eval_….json`. Süre GPU'da ~2–5 dk. CUDA bellek hatasında `--device cpu` ile tekrar (çok daha yavaş). Referans: plan yazılırken ayrı bir kopyada (commit'li eski kelime wav'larıyla) aynı ölçüm kelime WER %18,1, cümle WER %88,9, karar `egitime_gec` verdi; model 334 cümlenin 289'unda tek kelime yazıyordu. Canlı veriyle sayılar birkaç puan oynayabilir.

- [ ] **Step 3: Check the synthetic audio and the worst examples (Review Focus 1)**

Run:

```bash
ls data/users/FurkanV1/synthetic_eval/*.wav | wc -l
/home/mayasoft/app/venv/bin/python -c "
import soundfile as sf, glob, numpy as np
d=[sf.info(f).duration for f in glob.glob('data/users/FurkanV1/synthetic_eval/*.wav')]
print(len(d), round(np.mean(d),2), round(min(d),2), round(max(d),2))"
```

Expected: 334 dosya, ortalama 3–6 sn. En kötü 10 örneği oku: hatalar çoğunlukla kelime sonu harf kaybıysa (kırpma) bunu rapora not et; kırpma eşiğini değiştirmek bu planın dışında, kullanıcıya getirilir.

- [ ] **Step 4: Record the decision**

Ledger'a yaz: `Task 6: live word WER <x>, sentence WER <y>, diff <d>, decision <karar>, report <dosya>`.

- **Karar `dur` ise:** Task 7'yi çalıştırma. Planın geri kalanı yapılmadan durulur; sonuçlar ve en kötü örnekler kullanıcıya getirilir (spec: "değilse durulur ve sonuç kullanıcıya getirilir").
- **Karar `egitime_gec` ise:** Task 7.

---

### Task 7: Devam eğitimi ve karşılaştırma (yalnızca karar `egitime_gec` ise)

**Files:**
- Üretilir: `data/models/personalized_models/FurkanV1_whisper_cumle/` (adaptör + `checkpoints/`)
- Üretilir: `reports/whisper_side_FurkanV1_whisper_cumle_<zaman>.json`, iki `reports/synthetic_eval_*.json`, `logs/training_whisper_FurkanV1_*.log`

**Interfaces:**
- Consumes: Task 5 CLI, Task 4 CLI, Task 6 kararı.
- Produces: canlıya alma önerisi (iki koşul) — kullanıcıya getirilir; bu görev canlı klasöre yazmaz.

- [ ] **Step 1: Check again that no training is running**

Run: `ps aux | grep -E "train_(whisper_)?adapter" | grep -v grep`
Expected: boş.

- [ ] **Step 2: Start the continued training in the background**

```bash
nohup /home/mayasoft/app/venv/bin/python -m src.cli.train_whisper_adapter FurkanV1 \
  --init-from data/models/personalized_models/FurkanV1_whisper \
  --output-name FurkanV1_whisper_cumle > /tmp/train_cumle.log 2>&1 &
echo $!
```

Expected: süreç numarası. İlk dakikalarda logda `LoRA adaptörü yükleniyor`, `Sentetik eğitim cümlesi satırı: 2008`, `Sentetik ölçüm cümlesi: 334` görünür. Tur başına ~15–20 dk; en fazla 10 tur (~3 saat). Bekleme sırasında logu `Traceback|Error|Killed|OutOfMemory|eval_wer` için izle; hata olursa dur ve kullanıcıya getir.

- [ ] **Step 3: Confirm the training finished and saved**

Run: `tail -5 /tmp/train_cumle.log; ls data/models/personalized_models/FurkanV1_whisper_cumle; ls reports/whisper_side_FurkanV1_whisper_cumle_*.json`
Expected: `✅ Model kaydedildi: data/models/personalized_models/FurkanV1_whisper_cumle`, `adapter_model.safetensors` var, yan rapor var. `ls -la data/models/personalized_models/FurkanV1_whisper/adapter_model.safetensors` tarihi değişmemiş (14 Ekim).

- [ ] **Step 4: Measure both models on the full word set and synthetic sentences**

```bash
/home/mayasoft/app/venv/bin/python -m src.cli.evaluate_synthetic --user FurkanV1 --full-words > /tmp/se_live_full.log 2>&1
/home/mayasoft/app/venv/bin/python -m src.cli.evaluate_synthetic --user FurkanV1 --full-words --model FurkanV1_whisper_cumle > /tmp/se_cumle_full.log 2>&1
grep -E "Kelime WER|Karar|Rapor" /tmp/se_live_full.log /tmp/se_cumle_full.log
```

Expected: her iki logda `Kelime WER`/`Cümle WER` satırı ve rapor yolu.

- [ ] **Step 5: Apply the promotion rule and record it**

Koşullar: yeni modelin `sentence_wer` < canlınınki **ve** yeni modelin tam kelime `word_wer` ≤ canlınınki + 0,01. Ledger'a yaz: `Task 7: live word <a> sent <b>; cumle word <c> sent <d>; promotable <evet|hayır>`.

Canlıya alma bu planın dışındadır: sonuçlar kullanıcıya getirilir; onay gelirse spec'teki adımlar (yedek kopya → adaptör kopyası → kullanıcı API'yi yeniden başlatır) ayrıca yapılır.
