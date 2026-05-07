# Sunucuda Yapılacaklar (91.241.50.187)

## Durum Açıklaması
- Backend: FastAPI, `api.py` → port 8000
- Flutter APK: debug build hazır, telefona kurulacak
- Kullanıcı: `Furkan` (kUserId)

---

## 1. Güncel Kodu Sunucuya Yükle

```bash
# Yerel makineden sunucuya kopyala (SSH ile)
scp -r "/path/to/Pronouns/" user@91.241.50.187:/home/user/pronouns/

# veya git kullanıyorsanız
git pull origin master
```

---

## 2. Python Bağımlılıklarını Güncelle

```bash
cd /home/user/pronouns
pip install -r requirements.txt
```

---

## 3. Yeni API Endpoint'lerini Test Et

Sunucu ayağa kalktıktan sonra aşağıdaki komutlarla test et:

```bash
BASE="http://localhost:8000"
USER="Furkan"

# Ayarları getir
curl "$BASE/settings?user_id=$USER"

# Ayarları kaydet
curl -X POST "$BASE/settings" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"Furkan","model":"Furkan","algorithm":"lora","self_learning":true,"learning_rate":0.00003,"epochs":3,"batch_size":8}'

# Model bilgisi
curl "$BASE/model/info?user_id=$USER"

# Fine-tune başlat
curl -X POST "$BASE/fine-tune" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"Furkan","algorithm":"lora","learning_rate":0.00003,"epochs":3,"batch_size":8}'

# Kelime seti listesi
curl "$BASE/collect/word-sets"

# Sıradaki kelime
curl "$BASE/collect/next-word?user_id=$USER&set_file=wordSet.txt"

# İlerleme
curl "$BASE/collect/progress?user_id=$USER&set_file=wordSet.txt"
```

---

## 4. Docker ile Çalıştırma (önerilir)

```bash
cd /home/user/pronouns

# Yeniden derle ve başlat
docker-compose down
docker-compose build
docker-compose up -d

# Logları izle
docker-compose logs -f
```

### docker-compose.yml kontrol noktaları:
- Port 8000 dışarıya açık mı?
- `data/users/Furkan/` klasörü volume olarak bağlı mı?
- `datasets/words_set/` klasörü container içinde erişilebilir mi?

---

## 5. Furkan Kullanıcı Klasörünü Oluştur

```bash
mkdir -p /home/user/pronouns/data/users/Furkan/words
# metadata_words.csv otomatik oluşturulur (ilk kayıtta)
```

---

## 6. Self-Learning Yapısının Çalışması

`api.py` → `POST /record` endpoint'i:
- Her kayıt sonrası `metadata_words.csv`'ye satır ekler
- `total_samples % IDEAL_REPETITIONS == 0` olduğunda (yani her 10 kayıtta bir)
  arka planda `PersonalizedTrainer.run()` otomatik başlatır
- Bu davranışı kapatmak için `POST /settings` ile `"self_learning": false` gönder

---

## 7. Model Güncellemesi (Admin olarak)

Sunucudan manuel eğitim:
```bash
cd /home/user/pronouns
python personalize_model.py --user_id Furkan
# veya
python train_adapter.py
```

Veya API üzerinden:
```bash
curl -X POST "http://localhost:8000/fine-tune" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"Furkan","algorithm":"lora","epochs":10,"batch_size":8,"learning_rate":0.00003}'
```

---

## 8. APK'yı Telefona Yükle

**APK yolu (yerel):**
```
build/app/outputs/flutter-apk/app-debug.apk   (149 MB, debug build)
```

**Yükleme yöntemleri:**

### A) ADB ile (USB bağlı Samsung S26):
```bash
adb install build/app/outputs/flutter-apk/app-debug.apk
```

### B) Dosya transferi ile:
1. APK'yı telefona kopyala (USB / Drive / Telegram)
2. Samsung S26'da "Bilinmeyen kaynaklardan yükle" aktif et:
   - Ayarlar → Uygulamalar → Özel erişim → Bilinmeyen uygulamalar yükle → Dosyalar (veya Telegram)
3. APK'ya dokun, "Yükle"ye bas

### C) Sunucu üzerinden:
```bash
# APK'yı sunucuya kopyala
scp build/app/outputs/flutter-apk/app-debug.apk user@91.241.50.187:/home/user/pronouns/apk/pronouns.apk

# Telefondan tarayıcıyla indir:
# http://91.241.50.187:8000/apk
```

---

## 9. Test Akışı (Telefon + Sunucu)

1. Sunucu çalışıyor mu kontrol et: `curl http://91.241.50.187:8000/collect/word-sets`
2. APK'yı telefona yükle
3. Uygulamayı aç → "Kayıt" butonuna bas
4. Kelime setinden ilk kelimeyi söyle ve kaydet
5. `data/users/Furkan/metadata_words.csv` dosyasının güncellendiğini kontrol et
6. "Furkanca" butonuna bas → konuş → düzeltilmiş metni gör
7. "Ayarlar" → Fine-Tune Başlat → sunucu loglarında eğitim başladığını kontrol et

---

## 10. Sıradaki Geliştirmeler (Sonraki Aşama)

- [ ] Release APK (imzalı) build: `flutter build apk --release`
- [ ] Ayarları kalıcı hale getir (JSON/SQLite yerine in-memory şu an)
- [ ] `/settings` endpoint'ini JSON dosyasına persist et
- [ ] WER (Word Error Rate) hesapla ve `/model/info`'ya ekle
- [ ] Eğitim durumunu WebSocket/polling ile mobilden izle
- [ ] Push notification: eğitim tamamlandığında bildir
