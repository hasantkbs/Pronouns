# Furkanca Tercüman Çıktısı: Düzeltilmiş Cümle + Yerel TTS — Tasarım

Tarih: 2026-10-11
Durum: Onaylandı (sohbette bölüm bölüm), yazılı spec incelemede

## Amaç

Furkanca sayfası bir **tercüman** gibi çalışmalı: Furkan cümle halinde konuşur,
uygulama onun söylediği cümlenin **düzeltilmiş, düzgün Türkçe** halini ekranda
gösterir ve **anlaşılır bir sesle** okur.

Bugün bu davranış yok:

- `/translate` ASR metnini anahtar kelime tabanlı bir asistana veriyor
  (`src/core/nlu.py` + `src/core/actions.py`); `response_text` Furkan'ın
  cümlesi değil, asistanın cevabı ("merhaba" → selam cevabı; kalıp dışı her
  cümle → "Üzgünüm, ne istediğinizi anlayamadım...").
- `ASRSystem._fuzzy_correct` her kelimeyi Furkan'ın 842 kelimelik sözlüğüne
  zorla yaklaştırıyor. Tek kelimede işe yarıyor, cümlede anlamı bozuyor
  (gerçek fonksiyonla doğrulandı):

  | ASR çıktısı | `_fuzzy_correct` sonrası |
  |---|---|
  | ben bugün okula geldim | ben bugün okul geldi |
  | yarın arkadaşımla sinemaya gideceğiz | yakın arkadaş sinemaya gideceğiz |
  | su istiyorum | su istiyor |

- Ses, Furkan'ın kendi kelime kayıtları birleştirilerek (`WordSynthesizer`)
  üretiliyor; bu kayıtlar konuşma bozukluğunu taşıdığı için dinleyene
  anlaşılırlık kazandırmıyor, sözlükte olmayan kelimeler sessiz geçiyor.

Bu spec, üç alt projeden **C**'dir (A: ses yakalama / dış ses, B: cümle
tanıma doğruluğu — ayrı spec'ler).

## Kararlar (kullanıcıyla)

- Çıktı: Furkan'ın cümlesinin düzeltilmiş hali (tercüman), asistan cevabı değil.
- Düzeltme: **yerel, sözlük korumalı** (yaklaşım 1). Dışarıya veri gönderilmez,
  LLM yok. Bağlamsal n-gram düzeltme (yaklaşım 3) ancak gerçek cümle verisinde
  ihtiyaç görülürse, ayrı iş olarak.
- Ses: **Piper** ile yerel Türkçe TTS (`tr_TR-dfki-medium` — Piper'daki tek
  Türkçe ses, erkek sesi).

## Mimari

```
WAV ─► konuşmacı doğrulama ─► ASR (ham metin) ─► SentenceCorrector ─► PiperTTS ─► yanıt
           (değişmez)          (düzeltmesiz)      (sözlük korumalı)    (yerel, CPU)
```

### 1. `src/core/sentence_corrector.py` → `SentenceCorrector`

`SentenceCorrector(user_vocabulary: list[str], lexicon_path: Path)`
`correct(raw_text: str) -> str` — ham ASR metni → biçimlendirilmiş cümle
(boş girdi → boş string).

**Sözlük verisi:** `data/lexicon/tr_50k.txt` — hermitdave/FrequencyWords
(OpenSubtitles 2018) Türkçe 50k listesi, `kelime frekans` satırları, sıklığa
göre azalan; repoya eklenir (~700 KB, bir kez indirilir, statik).

İki ayrı küme kurulur:

- **Tanıma kümesi (`known`)**: genel listenin tamamı ∪ Furkan'ın sözlüğü
  (çok kelimeli girdiler — örn. "geri bildirim" — kelimelerine bölünür). Bu
  kümedeki bir kelimeye **asla dokunulmaz**.
- **Aday listesi (`candidates`)**: düzeltmede önerilebilecek genel liste
  kelimeleri; **ASCII varyantları hariç** — kelime Türkçe'ye özgü harf
  (ı ş ğ ç ö ü) içermiyorsa ve aynı katlanmış (ı→i, ş→s, ğ→g, ç→c, ö→o, ü→u)
  biçime sahip daha sık bir yazım listede varsa (örn. "aldim" ↔ "aldım"),
  aday olamaz. Tanıma kümesinde kalır ("su", "kir", "isim" korunsun diye —
  bunları listeden tamamen atmak "kir"i "bir"e çeviriyordu, prototipte
  doğrulandı). Liste sırası = sıklık sırası.

**Algoritma:**

1. `normalize_turkish_text(raw_text)` (mevcut, `src/utils/utils.py`) → boşlukla
   kelimelere böl.
2. Her kelime `w` için:
   - `w ∈ known` → olduğu gibi.
   - `len(w) ≤ 2` → olduğu gibi (kısa kelimede en yakın eşleşme rastgeledir).
   - Rakam içeren kelime (`3,5`, `14:30`, `%50`) olduğu gibi korunur; yalnızca
     cümle sonundaki noktalama atılır. Cümle ortasında büyük harfle başlayan
     kelime (`Ahmet`) özel isim sayılır, olduğu gibi korunur.
   - Aksi halde **yalnızca harf ekleyen** bir aday aranır (*son inceleme
     sonrası daraltıldı*: harf değiştirme/silme, sözlükte olmayan geçerli
     kelimelerin anlamını çeviriyordu — `ağrımıyor→ağrıtıyor`,
     `ilacımı→ilacını`, `Furkan→kurban`). Aday geçerli sayılmak için:
     `w` adayın alt dizisi olmalı, ilk harfler aynı olmalı
     (`emine→zemine` olmaz), olumsuzluk eki sayısı (`m[aeıiouöü]`) değişmemeli
     (`kaşınıyor→kaşınmıyor` olmaz). Mesafe = eklenen harf sayısı; eşik
     Furkan sözlüğü için `≤ (1 - config.FUZZY_MATCH_THRESHOLD) × len(w)`,
     genel liste için **en fazla 1 harf** (`üşümüyorum→düşünmüyorum` olmaz):
     - `f` = Furkan sözlüğündeki en iyi aday, `g` = aday listesindeki en iyi
       aday. Eşit mesafede **daha sık** kelime kazanır (liste sırası; Furkan
       sözlüğünde eşitlikte genel listedeki sırası, yoksa alfabetik).
     - `f` eşiğe giriyor ve `mesafe(f) ≤ mesafe(g)` → `f` (Furkan önceliği).
     - değilse `g` eşiğe giriyorsa → `g`.
     - hiçbiri eşiğe girmiyorsa → `w` olduğu gibi (uydurmak yerine ham hali).
3. Biçimlendirme:
   - İlk harf Türkçe kurallarıyla büyütülür (`i` → `İ`, `ı` → `I`).
   - Herhangi bir kelime soru eki ise (`mı mi mu mü` ve kişi/zaman ekli
     halleri: `^(mı|mi|mu|mü)(yım|yim|yum|yüm|sın|sin|sun|sün|yız|yiz|yuz|yüz|sınız|siniz|sunuz|sünüz|ydı|ydi|ydu|ydü|ymış|ymiş|ymuş|ymüş)?$`)
     ya da soru kelimesi ise (`ne nerede nereye nereden kim kime kimi nasıl neden niye niçin hangi kaç`)
     → sona `?`, aksi halde `.`.

Prototipte (atılacak kod) doğrulanan örnekler: `gldim→geldim`,
`kampnya→kampanya`, `tşekkür→teşekkür`, `yardm→yardım`, `bulv→bul`;
`su`, `kir`, `isim`, `okula`, `geldim`, `yarın`, `arkadaşımla`, `istiyorum`
korunur.

**Bilinen sınırlar** (yaklaşımın doğası, kabul edildi):

- ASR yanlış ama geçerli bir kelime duyarsa (`sevdi` → `sevgi`) düzeltilemez.
- Yalnızca harf ekleyen düzeltme yapıldığı için harf *değiştirilmiş* bozuk
  çıktılar düzeltilmez (`istiyom`, `geliyom`, `sinemya` olduğu gibi kalır) —
  anlam değiştirmekten güvenli.
- Eşit mesafede sıklık kazanır (ilk harf kuralından sonra `gdiyorum →
  gidiyorum` doğru düzelir; `bulv` ise artık düzeltilmez).
- ASR ASCII yazım üretirse (`aldim`) tanıma kümesinde olduğu için düzeltilmez
  (Whisper normalde Türkçe karakterli yazar).

**Performans:** sözlük dışı kelime başına ~50k Levenshtein karşılaştırması
(C uzantısı `python-Levenshtein`, mevcut bağımlılık) ≈ 10 ms; ihmal edilebilir.
Sözlük API açılışında bir kez yüklenir; `SentenceCorrector` kullanıcı başına
bir kez oluşturulup önbelleklenir.

### 2. `src/core/tts.py` → `PiperTTS`

`PiperTTS(voice_path: Path)`; `synthesize(text: str, output_path: str) -> str | None`
— WAV yazar, yolunu döner; her hatada (model yok, sentez hatası) loglar ve
`None` döner, istisna fırlatmaz.

- Ses modeli: `data/models/tts/tr_TR-dfki-medium.onnx` (+ `.onnx.json`), ~60 MB,
  repoya **eklenmez**; `scripts/download_piper_voice.sh` Hugging Face
  `rhasspy/piper-voices` deposundan (`tr/tr_TR/dfki/medium/`) indirir.
- Model API açılışında bir kez, CPU'da yüklenir (GPU'ya dokunmaz; ASR'ın
  ~9 GB'lık GPU bütçesiyle çakışmaz). Medium model cümle başına ~0,2–0,5 sn.
- Bağımlılık: `piper-tts` → `requirements.txt`.

### 3. `ASRSystem.transcribe` (`src/core/asr.py`)

Yeni parametre `transcribe(audio_path, fuzzy_correct: bool = True)`.
`/translate` `fuzzy_correct=False` ile çağırır (düzeltmeyi
`SentenceCorrector` yapar). Varsayılan `True` kalır → değerlendirme
scriptleri ve diğer çağıranlar etkilenmez.

### 4. `/translate` (`api.py`)

```
doğrulama (aynı) → asr.transcribe(..., fuzzy_correct=False)
→ boşsa {"error": "Konuşma anlaşılamadı."} (aynı)
→ corrected = get_corrector(user_id).correct(text)
→ boşsa {"error": "Konuşma anlaşılamadı."}
→ audio = tts.synthesize(corrected, out_path)
```

Yanıt sözleşmesi (alanlar korunur, anlamları netleşir):

| Alan | Değer |
|---|---|
| `recognized_text` | ham ASR metni |
| `response_text` | düzeltilmiş cümle |
| `audio_url` | `/download/<dosya>` ya da TTS başarısızsa `null` |
| `intent` | sabit `"translate"` (geri uyumluluk) |
| `missing_words` | `[]` (geri uyumluluk) |

`nlu.process_text`, `run_action` ve `WordSynthesizer` `/translate`'ten
çıkarılır; dosyalar silinmez (başka yerlerde kullanılıyor olabilir).

`get_corrector(user_id)`: `get_asr` ile aynı modül seviyesi önbellek deseni;
sözlük `data/users/{user_id}/metadata_words.csv`'den (`ASRSystem`'in
kullandığı aynı kaynak).

### İstemci

**Değişiklik yok.** React `FurkancaPage.tsx` yalnızca `response_text`,
`audio_url` ve `rejected` alanlarını kullanıyor; `audio_url` yoksa sadece metni
gösteriyor. Yeni APK gerekmez; API yeniden başlatılınca devreye girer.

## Hata Yönetimi

İlke: bir parça bozulursa Furkan'ın cümlesi kaybolmamalı.

| Durum | Davranış |
|---|---|
| `tr_50k.txt` yok/okunamıyor | uyarı loglanır; kelime düzeltmesi **tamamen atlanır**, yalnızca biçimlendirme (Furkan sözlüğüne zorla yaklaştırma = bugünkü hatalı davranış, geri dönülmez) |
| Piper modeli yok / sentez hatası | log; `audio_url: null`, metin döner |
| ASR boş | `{"error": "Konuşma anlaşılamadı."}` (bugünkü gibi) |
| Konuşmacı doğrulama | değişmez (alt proje A) |

## Test Planı (TDD)

`tests/test_sentence_corrector.py` (küçük sahte sözlük dosyası `tmp_path`'te
+ gerçek `tr_50k.txt` ile birkaç regresyon):

- Regresyon: tablodaki üç cümle **değişmeden** korunur (biçimlendirme hariç).
- Sözlük dışı bozuk kelime düzeltilir (`gldim→geldim`, `kampnya→kampanya`).
- Furkan sözlüğü önceliği: eşit mesafede Furkan'ın kelimesi seçilir.
- Eşikte aday yoksa kelime aynen kalır; ≤2 harfli kelimeye dokunulmaz.
- ASCII varyantı aday olmaz ama tanınır (`su`, `kir` korunur; düzeltme
  sonucu asla `aldim` olmaz).
- Biçim: `?` (ek ve soru kelimesi), `.`; `i` ile başlayan cümle `İ` ile başlar.
- Sözlük dosyası yoksa: kelimelere dokunulmaz, yalnızca biçimlendirme.

`tests/test_tts.py`:

- Gerçek model varsa geçerli WAV (22050 Hz, mono, süre > 0); yoksa `skip`.
- Sentez hatası (bozuk model yolu) → `None`, istisna yok.

`tests/test_translate_speaker_verification.py` (mevcut sahte nesne deseniyle):

- `response_text` = düzeltilmiş cümle, `recognized_text` = ham metin,
  `intent == "translate"`.
- NLU / `run_action` çağrılmaz.
- TTS `None` dönerse `audio_url` `null`, metin döner.
- `transcribe` `fuzzy_correct=False` ile çağrılır.

Canlı doğrulama: eval setinden birkaç kayıt `/translate`'e; ardından Furkan
uygulamadan birkaç cümle söyler, kullanıcı sonucu kontrol eder.

## Dağıtım

- Dal: `whisper-model-selection` (Task 6, henüz master'da değil) üstünde yeni
  dal; bitince merge, API'yi kullanıcı yeniden başlatır (`sudo`).
- `scripts/download_piper_voice.sh` dağıtımdan önce bir kez çalıştırılır.
- Geri alma: `git revert` + API yeniden başlatma.

## Kapsam Dışı

- A: ses yakalama / dış ses sorunu, konuşmacı doğrulama değişiklikleri.
- B: cümle düzeyinde ASR doğruluğu ölçümü, Whisper'a geçiş.
- Bağlamsal (n-gram / LLM) düzeltme.
- Flutter uygulaması.
- `nlu.py`, `actions.py`, `synthesizer.py` dosyalarını silmek.
