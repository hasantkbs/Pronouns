# Furkanca Sayfası: Dokun-Başlat / Dokun-Bitir Kayıt — Tasarım

Tarih: 2026-10-11
Durum: Onaylandı (sohbette bölüm bölüm), yazılı spec incelemede
Alt proje: **A** (dış ses sorunu). C (tercüman çıktısı) tamamlandı:
`docs/superpowers/specs/2026-10-11-tercuman-cikti-design.md`.

## Sorun (kullanıcının gözlemi: a + c)

- **(a) Furkan'ın cümlesi kayboluyor:** Furkan konuşurken/hemen sonra dış ses
  gelince sonuç gelmeden "Dinleniyor"a dönülüyor.
- **(c) Durum titriyor:** kısa gürültülerde (kapı, öksürük, tıkırtı) durum
  "Kaydediliyor" ↔ "Dinleniyor" arasında gidip geliyor.

## Kök nedenler (kod ve ölçümle doğrulandı)

1. `web/src/lib/vad.ts` `VadSegmenter` yalnızca **ses şiddetine** (RMS ≥ 0,012)
   bakıyor, konuşmayı gürültüden ayırmıyor: her ses kayıt başlatır, 0,3 sn'den
   kısa olanlar atılıp "Dinleniyor"a dönülür → (c).
2. Furkan konuşurken gelen dış ses aynı segmente karışır. Konuşmacı doğrulama
   bu karışık segmenti reddederse istemci (`FurkancaPage.tsx`,
   `result.rejected`) **hiçbir şey göstermeden** dinlemeye döner → (a).
   Cümle ortasında 1 sn'den uzun duraklama da cümleyi böler.
3. **Konuşmacı doğrulama canlıda hiç çalışmıyor:** kayıtlı profil CPU'da,
   embedding GPU'da; `torch.dot` her seferinde hata verir ve fail-open ile
   her ses kabul edilir (canlı logda 5/5 istekte "Benzerlik hesaplanamadı").
   Kararlar loglanmadığı için bu fark edilmemiş
   (`logging.getLogger("speaker_verification")` handler'sız).

## Kararlar (kullanıcıyla)

- Furkan ekrandaki düğmeye rahatça basabiliyor → **dokun-başlat / dokun-bitir**
  (yaklaşım 1). Basılı tut-konuş ve iyileştirilmiş eller serbest (Silero VAD)
  reddedildi.
- **Eller serbest mod tamamen kaldırılır** (seçenek olarak da kalmaz).

## Tasarım

### Kullanıcı akışı

```
[ 🎤 Konuşmaya Başla ] ─dokun─► [ ⏹ Bitir ] ─dokun─► İşleniyor… ─► Yanıt seslendiriliyor… ─► [ 🎤 Konuşmaya Başla ]
   (mikrofon kapalı)       (süre 0:03 + ses seviyesi)                      (sonuç listenin başına)
```

- Tek, geniş, kolay basılır düğme.
- Mikrofon **yalnızca kayıt sırasında** açık (arka planda dinleme yok).
- Düğme, mikrofon **gerçekten açıldıktan sonra** "Bitir"e döner ve süre o
  anda başlar (`getUserMedia` 0,1–0,3 sn sürebilir; ilk hece kaçmasın).
- Kayıt sırasında: geçen süre (`0:03`) + canlı ses seviyesi çubuğu.
- Sınırlar: **30 sn**'de otomatik biter ve gönderilir; **0,5 sn**'den kısa
  kayıt gönderilmez → "Çok kısa, tekrar dene".
- İşleniyor / seslendiriliyor sırasında düğme kilitli (yeni kayıt yok; TTS
  sesi kayda girmez).
- Her hata görünür (sunucu hatası, "Konuşma anlaşılamadı", beklenmedik
  `rejected` → "Ses tanınmadı, tekrar dene"); mesaj bir sonraki kayıtta temizlenir.
- Geçmiş listesi aynen (en yeni üstte).

### İstemci (`.claude/worktrees/react-rewrite`, dal `worktree-react-rewrite`, `web/src/`)

| Dosya | Sorumluluk |
|---|---|
| `lib/manualRecorder.ts` (yeni) | Saf mantık: `push(chunk)` ile PCM biriktirir; `durationMs`, son chunk'ın seviyesi (`calculateRms`); `finish()` → birleşik `Int16Array` ya da `tooShort`; `maxDurationMs` (30 000) aşılınca `push` "otomatik bitir" sinyali döner. Mikrofona dokunmaz. |
| `lib/furkancaFlow.ts` (yeni) | Saf reducer: durumlar `idle → starting → recording → uploading → speaking → idle`; olaylar `START`, `MIC_READY`, `MIC_FAILED`, `STOP`, `TOO_SHORT`, `RESULT`, `AUDIO_DONE`, `FAILED`; durum `error?: string`, `history`; `buttonEnabled` türetilir (`idle`, `recording`'de açık; `starting`, `uploading`, `speaking`'de kilitli). |
| `pages/FurkancaPage.tsx` | Yeniden yazılır: reducer + `startAudioCapture`/`stop` + `manualRecorder` + `pcmToWav` + `translateSpeech` + `Audio` oynatma. VAD, duraklat/devam, sürekli dinleme kaldırılır. |
| `lib/vad.ts` → `lib/wav.ts` | `VadSegmenter` + testleri silinir; `pcmToWav`, `calculateRms`, `pcmChunkDurationMs` testleriyle `wav.ts`/`wav.test.ts`'e taşınır. |
| `pages/KayitPage.tsx` | Yalnızca import yolu (`../lib/wav`). |

`api.ts` `translateSpeech` değişmez (zaten `error`'u fırlatıyor).

### Sunucu (`/home/mayasoft/app/Pronouns`, `master` üstünde dal)

- **Doğrulama reddetmez, loglar:** `config.SPEAKER_VERIFY_BLOCKING = False`.
  `/translate` doğrulamayı çalıştırıp skoru loglar; `BLOCKING` `False` iken
  `is_match` ne olursa olsun reddetmez. `True` yapılırsa bugünkü ret davranışı
  geri gelir. `SPEAKER_VERIFY_ENABLED = False` doğrulamayı tamamen atlar (değişmez).
- **Cihaz hatası:** `SpeakerVerificationService.verify` skordan önce profili
  embedding'in cihazına taşır (`profile.to(embedding.device)`).
- **Log:** `logs/speaker_verification.log` (APK indirme logundaki
  `_apk_logger` deseniyle, `config.LOG_DIR` altında dosya handler'ı); satır:
  `user=<id> match=<bool> score=<float|None> blocking=<bool>`.
- `/translate` yanıt sözleşmesi değişmez.

## Hata Yönetimi

| Durum | Davranış |
|---|---|
| Mikrofon izni reddedildi / açılamadı | `MIC_FAILED` → `idle`, "Mikrofon açılamadı: …" mesajı |
| Kayıt < 0,5 sn | `TOO_SHORT` → `idle`, "Çok kısa, tekrar dene" (gönderilmez) |
| Kayıt 30 sn'ye ulaştı | otomatik `STOP` → normal gönderim |
| Sunucu/ağ hatası, "Konuşma anlaşılamadı" | `FAILED(msg)` → `idle`, mesaj görünür |
| `rejected: true` (yalnızca `BLOCKING=True` ise) | "Ses tanınmadı, tekrar dene" → `idle` |
| TTS sesi yok (`audio_url: null`) | metin listeye eklenir, `speaking` atlanır |
| Ses oynatma hatası | `AUDIO_DONE` sayılır → `idle` |
| Doğrulama hata verir | fail-open (değişmez), skor `None` loglanır |

## Test Planı

- **vitest** (`npm test`, `web/`):
  - `manualRecorder.test.ts`: biriktirme + birleşik çıktı, süre hesabı, seviye,
    0,5 sn altı `tooShort`, 30 sn'de otomatik bitir sinyali.
  - `furkancaFlow.test.ts`: mutlu yol (idle→starting→recording→uploading→speaking→idle),
    `audio_url` yokken speaking atlanır, her hata → idle + mesaj, yeni `START`
    mesajı temizler, `buttonEnabled` kuralları, `uploading`/`speaking`'de `START` yok sayılır.
  - `wav.test.ts`: `vad.test.ts`'teki `pcmToWav`/`calculateRms` testleri taşınır.
- **pytest** (`master`):
  - `BLOCKING=False` + `match=False` → `/translate` reddetmez, çeviri döner.
  - `BLOCKING=True` + `match=False` → bugünkü `{"rejected": true, ...}`.
  - `verify`: profil ve embedding farklı cihazdayken skor hesaplanır (cihaz
    farkı sahte tensörle ya da CUDA varsa gerçek tensörle).
  - Log dosyasına karar satırı yazılır.
- **Elle:** önce web (`https://app.furkanca.algorixstudio.com`), sonra
  Furkan'ın telefonunda yeni APK: kayıt sırasında arkadan ses, cümle ortasında
  uzun duraklama, kısa dokunuş, 30 sn sınırı, mikrofon izni reddi.

## Dağıtım

- **Sunucu:** `master`'a merge; API'yi kullanıcı yeniden başlatır (`sudo`).
- **Web:** `cd .claude/worktrees/react-rewrite/web && npm run build`.
  `pronouns-web.service` (`serve -s dist`) dosyaları diskten okuduğu için
  yeniden başlatma gerekmez. **`deploy-web.sh` kullanılmaz**: systemd servisinden
  önce yazılmış; `pkill` + kendi `serve`'ünü başlatması `Restart=always` ile
  port çakışması yaratır.
- **APK:** `web/deploy-android.sh` (derler, eski APK'yı yedekler,
  `data/apk/pronouns.apk`'ya koyar). Furkan'ın telefonuna `/apk`'dan **yeni APK
  kurulmalı**; eski APK'da eller serbest mod kalır.
- Geri alma: web/APK için önceki `dist`/APK yedeği; sunucu `git revert` +
  yeniden başlatma; `SPEAKER_VERIFY_BLOCKING=True` eski ret davranışını açar.

## Kapsam Dışı

- Eller serbest mod, VAD, Silero.
- Kayıt içindeki arka plan sesini ayırma/temizleme.
- `worktree-react-rewrite` dalını `master`'a merge etmek.
- Flutter uygulaması; `deploy-web.sh`'yi düzeltmek.
