# Sentetik Cümlelerle Cümle Doğruluğunu Ölçme ve İyileştirme — Tasarım

Tarih: 2026-10-22
Durum: Onaylandı (sohbette iki bölüm), yazılı spec incelemede
Alt proje: **B2** (cümle doğruluğu). B1'in gerçek cümle kayıtları henüz yok
(`docs/superpowers/specs/2026-10-12-cumle-olcum-design.md`); bu iş telefon
gerektirmeden Furkan'ın kelime kayıtlarından sentetik cümle üretir.

## Sorun

Canlı model (Whisper + LoRA, `data/models/personalized_models/FurkanV1_whisper`)
yalnızca tek kelimelik kayıtlarla eğitildi: kelime WER %14,9, CER %9,8
(14 Ekim eğitimi, 21 saat). Furkan uygulamada cümle kuruyor; cümle
doğruluğu bilinmiyor ve şu an telefondan yeni kayıt alınamıyor.

Veriden gözlemler:

- Kelime kayıtları ortalama 3,5 sn; konuşma ortalama 1,3 sn, gerisi sessizlik.
- `train.csv`/`eval.csv` kayıt düzeyinde rastgele bölündü; ölçümdeki
  kelimelerin hemen hepsi eğitimde başka tekrarıyla var. %14,9 bu yüzden
  "duyulmuş kelimenin yeni tekrarı" başarısıdır.
- Sözlük 878 kelime, çoğu yalın isim, bir kısmı çekimli fiil
  (`gitti`, `sordu`, `sevdi`) ya da mastar.
- `src/training/augment_from_words.py` (ilk commit'ten) rastgele kelime
  dizer, sessizliği kırpmaz, yolları çözmez; bu işte kullanılmaz ve
  değiştirilmez.

## Kararlar (kullanıcıyla)

- **Yaklaşım A:** Claude'un yazdığı anlamlı cümleler, kırpılmış kelime
  kayıtlarından sentetik ses, canlı adaptörden devam eden kısa eğitim.
- **Önce ölçüm, koşullu eğitim:** cümle WER'i kelime WER'inden en az
  **5 puan** kötüyse eğitime geçilir; değilse durulur ve sonuç kullanıcıya
  getirilir.
- **Sızıntı yok:** ölçüm sesleri yalnızca `eval.csv`, eğitim sesleri
  yalnızca `train.csv` kayıtlarından; ölçüm ve eğitim cümle metinleri ayrık.
- **Self-learning:** `sentetik_egitim.txt` varsa self-learning eğitimi de
  cümleleri otomatik ekler (seçenek 1).
- Canlı model klasörüne yalnızca kullanıcı onayıyla, yedek alınarak
  kopyalanır; API'yi kullanıcı yeniden başlatır.

## Başarı ölçütü

- `evaluate_synthetic` canlı modelin kelime ve sentetik cümle WER'ini aynı
  kayıtlar üzerinde verir ve "eğitime geç / dur" kararını yazar.
- Eğitim yapılırsa yeni model: sentetik cümle WER'i canlı modelden düşük
  **ve** `eval.csv`'nin tamamında kelime WER'i canlı modelden en fazla
  1 puan kötü. Bu iki koşul sağlanmazsa canlıya alınmaz.

## Tasarım (`/home/mayasoft/app/Pronouns`, dal `sentetik-cumle`)

### Bölüm 1: Cümleler, sentetik ses, ölçüm

**Cümle listeleri** (`datasets/sentence_sets/`, satır başına bir cümle,
UTF-8, boş satır yok):

| Dosya | İçerik |
|---|---|
| `sentetik_olcum.txt` | ~150 cümle |
| `sentetik_egitim.txt` | ~1000 cümle |

- Her cümle 3–7 kelime. Ölçüm cümlelerinin her kelimesinin `eval.csv`'de,
  eğitim cümlelerinin her kelimesinin `train.csv`'de **tek başına** kaydı
  var (bu kayıtların hepsi `metadata_words.csv`'den bölündüğü için
  sözlükte de var; `metadata_words.csv` canlıda değiştiği için test ona
  değil, sesin gerçekten alındığı bölüme bakar). Karşılaştırma
  `normalize_turkish_text` ile (Türkçe küçük harf, noktalama silinmiş).
- İki dosya arasında ortak cümle yok; ikisinde de `olcum_*.txt`
  cümlelerinden hiçbiri yok (o cümleler gerçek kayıt doğrulamasına ayrılır).
- Dosya içinde tekrar eden cümle yok.
- Bu kurallar `tests/test_synthetic_sentence_sets.py` ile denetlenir;
  `train.csv`/`eval.csv` yoksa kapsama testleri atlanır.
- Dosya adı `olcum_` ile başlamadığı için `/collect/word-sets?kind=sentence`
  bunları listelemez (Kayıt ekranı değişmez).

**Sentetik ses** (yeni `src/training/sentence_synth.py`, saf fonksiyonlar):

- `load_takes(csv_path, user_dir) -> Dict[str, List[str]]`: `train.csv` ya da
  `eval.csv`'yi okur, `quality_score >= config.QUALITY_THRESHOLD` filtresini
  uygular (sütun yoksa filtre yok), yolu `resolve_audio_path` ile çözer,
  okunamayan dosyayı atar; anahtar `normalize_turkish_text(transkript)`.
- `trim_word(audio, top_db=30) -> np.ndarray`: baş/son sessizliği kırpar;
  sonuç boşsa orijinali döndürür.
- `build_sentence(words, takes_by_word, rng, sr=16000) -> np.ndarray`:
  her kelime için `rng` ile bir kayıt seçer, kırpar, tepe genliğini 0,9'a
  eşitler; kelimeler arasına `rng` ile 80–350 ms sessizlik, başa ve sona
  200 ms sessizlik koyar. Bir kelimenin kaydı yoksa `MissingWordError`.
- `sentence_words(sentence) -> List[str]`: `normalize_turkish_text` ile
  bölünmüş kelimeler.
- Aynı `rng` tohumu aynı sesi verir (testle denetlenir).

**Ölçüm aracı** (yeni `src/cli/evaluate_synthetic.py`):

```
python -m src.cli.evaluate_synthetic --user FurkanV1 [--model <klasör adı>] [--device cpu]
```

- **Sabit ölçüm sesleri:** `data/users/<id>/synthetic_eval/` yoksa
  `sentetik_olcum.txt`'deki her cümle için `random.Random(config.SEED)` ile
  2 ses üretir (`eval.csv` kayıtlarından), `s<cümle no>_<n>.wav` olarak yazar
  ve `manifest.csv` (`file_path, transcription`) oluşturur. Klasör varsa
  yeniden üretmez; böylece her model aynı seslerle ölçülür. Kaydı olmayan
  kelime içeren cümle atlanır, sayısı rapora yazılır.
- **Kelime kıyası:** aynı model `eval.csv`'de, sentetik ölçüm cümlelerinde
  geçen kelimelerin kayıtları üzerinde tek tek ölçülür (kelime WER'i).
  `--full-words` ile `eval.csv`'nin tamamı ölçülür (Bölüm 2 kararı için).
- Ölçüm canlıyla aynı: `asr.transcribe(path, fuzzy_correct=False)`; ayrıca
  `SentenceCorrector` sonrası sonuç. Skorlar `evaluate_sentences.score`,
  `normalize` ve `evaluate` ile hesaplanır (kod tekrarı yok).
- `--model` verilmezse canlı model (`evaluate_sentences._load_asr` ile aynı
  seçim); verilirse `data/models/personalized_models/<ad>`.
- Karar: `decide(word_wer, sentence_wer) -> "egitime_gec" | "dur"`; fark
  (cümle − kelime, ham WER) **≥ 0,05** ise `egitime_gec`.
- Rapor `reports/synthetic_eval_<YYYYmmdd-HHMMSS>.json`: model, kelime ve
  cümle özetleri, fark, karar, atlanan cümleler, satırlar. Ekranda özet,
  karar ve en kötü 10 örnek.

**`evaluate_sentences`'e `--model`:** gerçek cümle kayıtları gelince iki
modeli karşılaştırmak için; verilmezse bugünkü davranış.

### Bölüm 2: Eğitim (yalnızca karar `egitime_gec` ise çalıştırılır)

**`src/cli/train_whisper_adapter.py`'ye eklemeler:**

- `WhisperPersonalizedTrainer(user_id, base_model_path=None, init_from=None,
  output_name=None, use_sentences=True)`.
- `--init-from <klasör>`: LoRA adaptörü `PeftModel.from_pretrained(...,
  is_trainable=True)` ile o klasörden yüklenir (yeni LoRA yerine). Verilince
  devam ayarları geçerli olur: en fazla `config.CONTINUE_EPOCHS = 10` tur,
  öğrenme hızı `config.CONTINUE_LEARNING_RATE = 1e-5`, erken durdurma sabrı
  `config.CONTINUE_EARLY_STOPPING_PATIENCE = 3`. Verilmezse bugünkü
  `NUM_FINETUNE_EPOCHS`, `FINETUNE_LEARNING_RATE`, `EARLY_STOPPING_PATIENCE`.
- `--output-name <ad>`: çıktı `data/models/personalized_models/<ad>`.
  `<user_id>_whisper` (canlı klasör) ile `init_from` aynı yere yazmak
  istenirse `ValueError` (testle denetlenir). Verilmezse bugünkü
  `<user_id>_whisper`.
- `--no-sentences`: cümleleri eklemez.

**Eğitim verisi:**

- Kelimeler: `train.csv`, bugünkü filtrelerle.
- Cümleler (`use_sentences` ve `sentetik_egitim.txt` varsa): her cümle
  `config.SYNTH_SENTENCE_REPEATS = 2` satır. Satır yalnızca metni taşır;
  `transform` sırasında `build_sentence` ile `train.csv` kayıtlarından
  **anında** üretilir (her çağrıda farklı tekrar ve duraklama, diske yazılmaz).
  Kaydı olmayan kelime içeren cümle veri seti kurulurken atılır, sayısı
  loglanır. Augmentation cümlelere de uygulanır.
- Satır ayrımı: `file_path` boş, yeni `kind` sütunu `word|sentence`.

**Tur sonu ölçüm verisi:** `eval.csv` kelimeleri + varsa
`synthetic_eval/manifest.csv` sesleri; en iyi tur bunların birleşik WER'ine
göre seçilir (`metric_for_best_model="wer"` aynı).

**Self-learning:** `TrainingService` değişmez; `WhisperPersonalizedTrainer
(user_id=...)` varsayılanı `use_sentences=True` olduğundan self-learning
eğitimi (sıfırdan, canlı klasöre) cümleleri otomatik ekler. Bu eğitim
yaklaşık %30 uzar.

**Rapor:** eğitim verisine `init_from`, `output_name`,
`sentence_train_samples`, `synthetic_eval_samples` eklenir. `--output-name`
verilen eğitimin raporu `reports/whisper_side_<ad>_<zaman>.json`'a (tüm
alanlarla) yazılır, `training_<user>_*.json`'a yazılmaz: aksi halde
`TrainingService.get_last_wer` (Ayarlar'daki model kartı) canlı olmayan
modelin WER'ini gösterirdi. Varsayılan (self-learning) eğitim raporu bugünkü
gibi `ReportingService` ile yazılır.

**Çalıştırma (Claude):**

```
python -m src.cli.train_whisper_adapter FurkanV1 \
  --init-from data/models/personalized_models/FurkanV1_whisper \
  --output-name FurkanV1_whisper_cumle
```

Başlamadan önce başka bir eğitim sürecinin çalışmadığı (`ps`, `nvidia-smi`)
denetlenir. Arka planda çalışır; tahmini süre en fazla ~3 saat.

**Karar ve canlıya alma:**

1. `evaluate_synthetic --model FurkanV1_whisper_cumle --full-words` ve aynısı
   canlı model için.
2. Koşullar: yeni modelin sentetik cümle WER'i < canlınınki **ve** tam
   `eval.csv` kelime WER'i ≤ canlınınki + 0,01.
3. Sonuç kullanıcıya getirilir. Onay gelirse: `FurkanV1_whisper` →
   `FurkanV1_whisper_yedek_<YYYYmmdd>` (kopya), yeni adaptör dosyaları
   (`adapter_config.json`, `adapter_model.safetensors`, işlemci/tokenizer
   dosyaları) `FurkanV1_whisper`'a kopyalanır; API'yi kullanıcı yeniden
   başlatır. Geri alma: yedek klasörü geri kopyalamak.

## Hata Yönetimi

| Durum | Davranış |
|---|---|
| Cümlede kaydı olmayan kelime | Ölçümde ve eğitimde o cümle atlanır, sayısı raporlanır |
| Okunamayan ses dosyası | `load_takes` atar |
| `synthetic_eval/` yok | Ölçüm üretir; eğitim tur sonu ölçümü yalnızca kelimelerle yapar |
| `sentetik_egitim.txt` yok | Eğitim bugünkü gibi yalnızca kelimelerle |
| Çıktı canlı klasör ya da `init_from` ile aynı | `ValueError`, eğitim başlamaz |
| GPU belleği yetmez (API de GPU'da) | Ölçüm `--device cpu`; eğitim için kullanıcıya danışılır |

## Test Planı

- **pytest** (sentetik sesler küçük numpy dizileriyle, model yüklenmez):
  - Cümle listeleri: 3–7 kelime, sözlükte birebir, ayrıklık (iki dosya ve
    `olcum_*`), dosya içi tekrar yok, beklenen boyut (≥140 / ≥950).
  - `trim_word`: sessizlik kırpılır; tamamen sessiz giriş orijinal döner.
  - `build_sentence`: uzunluk = kırpılmış kelimeler + duraklar (80–350 ms
    aralığında) + 2×200 ms; aynı tohum aynı çıktı; eksik kelime
    `MissingWordError`; tepe ≤ 0,9.
  - `load_takes`: kalite filtresi, okunamayan dosya atılır, anahtar
    normalize.
  - `decide`: fark 0,05 → `egitime_gec`, 0,049 → `dur`.
  - Ölçüm sesleri: yoksa üretilir, varsa yeniden üretilmez; manifest doğru.
  - Trainer: çıktı yolu kuralları (`ValueError`), devam ayarlarının seçimi,
    cümle satırlarının sayısı (`REPEATS` katı) ve eksik kelimeli cümlenin
    atılması, `--no-sentences`.
  - `evaluate_sentences --model`: model yolu seçimi.
  - Mevcut 116 test geçmeye devam eder.
- **Gerçek çalıştırma (Claude):** `evaluate_synthetic` canlı model üzerinde;
  karar `egitime_gec` ise eğitim ve iki modelin karşılaştırması.
- **Elle (sonra):** Furkan `olcum_*` cümlelerini okuyunca
  `evaluate_sentences` iki modelle.

## Kapsam Dışı

- `train.csv`/`eval.csv` bölünmesini kelime düzeyine çevirmek; yeni kelime
  (sözlük dışı) doğruluğu.
- `augment_from_words.py`'yi düzeltmek ya da silmek.
- Uygulama arayüzü, API uç noktaları, `TrainingService` kodu.
- Canlıya almayı otomatik yapmak.
