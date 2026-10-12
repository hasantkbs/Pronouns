# Cümle Kaydı ve Cümle Doğruluğu Ölçümü — Tasarım

Tarih: 2026-10-12
Durum: Onaylandı (sohbette bölüm bölüm), yazılı spec incelemede
Alt proje: **B1** (cümle düzeyinde ASR doğruluğunu ölçmek). B2 (iyileştirme)
bu ölçümün sonucuna göre ayrıca tasarlanır. Önceki alt projeler:
C `docs/superpowers/specs/2026-10-11-tercuman-cikti-design.md`,
A `docs/superpowers/specs/2026-10-11-dokun-konus-design.md`.

## Sorun

Whisper + LoRA adaptörü Furkan'ın **tek kelimelik** kayıtlarıyla eğitildi
(eval WER %16,1, CER %10,5). Furkan ise uygulamada **cümle** kuruyor. Cümle
akışında (kelime geçişleri, ritim, duraklama) doğruluğun ne olduğunu
bilmiyoruz; Furkan'ın hiç cümle kaydı yok. Ölçmeden iyileştirme yapılamaz.

## Kararlar (kullanıcıyla)

- **Okunan cümle (prompted read speech):** Furkan ekranda gösterilen cümleyi
  okur; beklenen metin bilindiği için WER/CER hesaplanabilir.
- **Cümle listesini Claude hazırlar**, kullanıcı inceleyip onaylar:
  60 cümle, iki dosya, her biri 30 cümle.
- **Yaklaşım 1:** Kayıt sayfasına "Cümle" modu; her cümle **2 kez** okunur
  (toplam 120 kayıt).
- Cümle kayıtları **eğitime karışmaz** (B1 yalnızca ölçüm).

## Cümle listesi

`datasets/sentence_sets/` altında iki yeni dosya (satır başına bir cümle,
UTF-8, boş satır yok):

| Dosya | İçerik |
|---|---|
| `olcum_sozluk_ici.txt` | 30 cümle; her kelimesi FurkanV1'in sözlüğünde (`metadata_words.csv` transkripsiyonları) **birebir** var; 3–7 kelime; doğal günlük cümleler |
| `olcum_sozluk_disi.txt` | 30 günlük cümle; her birinde **en az bir** sözlük dışı kelime |

Kelime karşılaştırması: Türkçe küçük harf (İ→i, I→ı) + noktalama silinmiş
haliyle. Mevcut `sentence_set_*.txt` dosyalarına dokunulmaz.

## Tasarım

### Sunucu (`/home/mayasoft/app/Pronouns`, dal `cumle-olcum`)

- Uç noktalara isteğe bağlı **`kind`** parametresi: `word` (varsayılan) |
  `sentence`. Geçersiz değer → **400**. `kind` gönderilmezse bugünkü davranış
  aynen sürer (eski APK uyumlu).

| Uç nokta | `kind=sentence` davranışı |
|---|---|
| `GET /collect/word-sets` | `datasets/sentence_sets/` altındaki yalnızca `olcum_*.txt` dosyalarını listeler |
| `GET /collect/next-word` | Seçili setten sıradaki cümleyi ve tekrar numarasını döndürür; hedef tekrar `config.SENTENCE_REPETITIONS` (2); `exclude` cümle metinleri (virgül içerebileceği için **`\n` ile ayrılır**) |
| `GET /collect/progress` | Tamamlanan cümle sayısı (≥ 2 kayıt), toplam, `ideal_repetitions: 2` |
| `POST /record` | Form alanı `word` cümle metnini taşır; kayıt `data/users/<id>/audio/s_<sha1(cümle)[:10]>/rep<N>.wav`; satır `metadata.csv`'ye (`RECORD_TYPE_SENTENCE`) |

- `metadata.csv` satırı kelime satırıyla aynı alanlar + **`set_file`**:
  `file_path, transcription, repetition, quality_score, rms, snr_db, duration, set_file`.
  `transcription` cümlenin dosyadaki hali.
- Cümle yüklemesi **self-learning eğitimini tetiklemez** (bugün `/record`
  kelime sayısı `IDEAL_REPETITIONS`'ın katı olunca eğitim başlatıyor; bu
  yalnızca `kind=word` için kalır).
- Eğitim ve sözlük yalnızca `metadata_words.csv` okuduğu için cümleler
  eğitime ve sözlüğe karışmaz (değişiklik gerekmez).
- `config.SENTENCE_REPETITIONS = 2`.
- Kelime/cümle dalları ortak yardımcılarla yazılır (set dosyası okuma,
  kayıt sayıları); kelime yolunun davranışı değişmez.

### İstemci (`.claude/worktrees/react-rewrite`, dal `worktree-react-rewrite`, `web/src/`)

- Kayıt sayfasının üstünde **"Kelime | Cümle"** mod seçimi. Varsayılan
  "Kelime"; bugünkü akış aynen. Set listesi ve ilerleme seçili moda göre
  (`kind`) alınır.
- **Cümle modu:**
  - Cümle büyük yazıyla; altında "12 / 30 cümle · bu cümle 1/2".
  - **Dokun-başlat / dokun-bitir**, `ManualRecorder` ile (0,5 sn altı
    gönderilmez → "Çok kısa, tekrar dene"; 30 sn'de otomatik biter).
    Kelime modundaki saniye ayarı bu modda gizlenir.
  - Kayıttan sonra dinleme yok (kelime modunda da yok).
  - "Atla" bugünkü gibi çalışır.
- **Çevrimdışı kuyruk:** `PendingTake`'e `kind?: 'word' | 'sentence'`;
  alanı olmayan eski kayıtlar `'word'` sayılır. IndexedDB sürümü değişmez
  (yeni index yok). `addTake(word, blob, kind)`; bekleyenler türe göre
  filtrelenir (`getPendingWords(kind)`); Bekleyen Kayıtlar sayfası
  `uploadRecording(take.word, take.audioBlob, take.kind ?? 'word')`.

| Dosya | Değişiklik |
|---|---|
| `lib/api.ts` | `getWordSets`, `getNextWord`, `getProgress`, `uploadRecording`'e `kind` (varsayılan `'word'`) |
| `lib/pendingRecordingsStore.ts` | `kind` alanı, `addTake(..., kind)`, `getPendingWords(kind)` |
| `pages/KayitPage.tsx` | Mod seçimi + cümle modu kayıt akışı |
| `pages/PendingUploadsPage.tsx` | Türe göre yükleme |

### Ölçüm betiği (`src/cli/evaluate_sentences.py`)

```
python -m src.cli.evaluate_sentences --user FurkanV1 [--device cpu]
```

- `data/users/<id>/metadata.csv`'deki her kayıt için:
  1. **Ham:** kullanıcının ayarındaki modelle (`whisper`) `asr.transcribe(path, fuzzy_correct=False)`.
  2. **Düzeltilmiş:** ham + `SentenceCorrector` (canlı `/translate` ile aynı).
- **Normalizasyon** (iki taraf da): Türkçe küçük harf (İ→i, I→ı), noktalama
  silinir, boşluklar tekleştirilir.
- **Metrikler** (`jiwer`, venv'de kurulu): WER, CER, tam isabet oranı;
  gruplar `olcum_sozluk_ici.txt`, `olcum_sozluk_disi.txt` (`set_file`'a göre)
  ve toplam; her biri ham ve düzeltilmiş için.
- Terminalde özet tablo + en kötü 10 örnek (beklenen / ham / düzeltilmiş).
- Tam rapor `reports/sentence_eval_<YYYYMMDD-HHMMSS>.json`: özet + kayıt başına
  satırlar.
- Yalnızca okur; API çalışırken GPU'da çalışır. Bellek yetmezse `--device cpu`.
- Kayıt yoksa "Cümle kaydı yok" yazar ve 1 ile çıkar; okunamayan ses dosyası
  atlanır ve raporda `skipped` listesine yazılır.

**B2 kararı (ölçüm sonrası, kullanıcıyla):** sözlük içi cümlelerde ham WER
yüksekse sorun cümle akışındadır → cümle verisiyle eğitim (B2). Hata
yalnızca sözlük dışı cümlelerdeyse → kelime toplamaya devam.

## Hata Yönetimi

| Durum | Davranış |
|---|---|
| Geçersiz `kind` | 400 |
| `kind` yok | `word` (bugünkü davranış) |
| Set dosyası yok / `olcum_` değil (`kind=sentence`) | 404 |
| Setteki tüm cümleler tamam | `next-word` `done: true` (kelime yoluyla aynı biçim) |
| Kayıt < 0,5 sn | gönderilmez, "Çok kısa, tekrar dene" |
| Ağ yok | kuyruğa (`kind` ile), sonra Bekleyen Kayıtlar'dan yüklenir |
| Ölçümde okunamayan ses | atlanır, `skipped`'e yazılır |

## Test Planı

- **pytest** (`master`'dan dal):
  - Cümle listeleri: sözlük içi dosyadaki her kelime FurkanV1 sözlüğünde;
    sözlük dışı her cümlede en az bir sözlük dışı kelime; tekrar eden cümle
    yok; her dosyada 30 satır.
  - Uç noktalar: `kind` yokken kelime davranışı; `kind=sentence` set listesi
    yalnızca `olcum_*`; `next-word` sıradaki cümle + tekrar, 2 kayıttan sonra
    sonrakine geçer, `\n` ile ayrılmış `exclude`; `progress`; `/record`
    dosyayı `audio/s_<hash>/repN.wav`'a yazar ve `metadata.csv`'ye `set_file`
    ile ekler; `metadata_words.csv` değişmez; cümle yüklemesi eğitimi
    tetiklemez; geçersiz `kind` → 400.
  - Ölçüm: normalizasyon (İ/I, noktalama), grup metrikleri, en kötü örnek
    sıralaması, rapor yapısı, `skipped` — sahte ASR ve sahte düzelticiyle
    (model yüklenmez).
- **vitest**: kuyrukta `kind` saklanır, eski kayıt `'word'` sayılır,
  `getPendingWords(kind)` filtreler; `api.ts` isteğe `kind` koyar (sahte
  `fetch`).
- **Elle:** web'de cümle modu (kayıt, atla, kuyruk, ilerleme), sonra
  Furkan'ın telefonunda yeni APK.

## Dağıtım

1. Sunucu: `cumle-olcum` → `master` merge; API'yi kullanıcı yeniden başlatır.
2. Web: `cd .claude/worktrees/react-rewrite/web && npm run build`
   (**`deploy-web.sh` kullanılmaz**).
3. APK: `web/deploy-android.sh`; Furkan'ın telefonuna yeni APK kurulur.
4. Furkan 60 cümleyi ikişer kez kaydeder; Claude ölçümü çalıştırıp sonucu
   kullanıcıyla değerlendirir.

## Kapsam Dışı

- Cümle verisiyle eğitim (B2), dil modeli, `SentenceCorrector` değişikliği.
- Mevcut `sentence_set_*.txt` dosyaları; Flutter uygulaması.
- `worktree-react-rewrite` dalını `master`'a merge etmek.
