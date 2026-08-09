# Pronouns AI - Mimari Komut Notları

Bu dosya, sistemin yönetimi için gerekli olan temel komutları ve mimari yapıları içerir.

## 🚀 API Servis Yönetimi (Systemd)

API artık bir systemd servisi olarak çalışmaktadır. Bu sayede sunucu başladığında otomatik çalışır ve çökerse kendini yeniler.

- **Servis Durumu:** `sudo systemctl status pronouns-api.service`
- **Servisi Başlat:** `sudo systemctl start pronouns-api.service`
- **Servisi Durdur:** `sudo systemctl stop pronouns-api.service`
- **Servisi Yeniden Başlat:** `sudo systemctl restart pronouns-api.service`
- **Logları İzle (Canlı):** `sudo journalctl -u pronouns-api.service -f`

## 📊 Veri ve Kayıt Yapısı

Kayıtlar `/home/mayasoft/app/Pronouns/data/users/<user_id>/` altında saklanır.

- **Ses Dosyaları:** `words/<kelime>/repX.wav`
- **Metadata:** `metadata_words.csv` (Dosya yolları, kalite skorları ve **timestamp** bilgisini içerir)
- **Yeni Kayıt Ayrımı:** 10 Haziran 2026'dan itibaren tüm yeni kayıtlara `timestamp` sütunu eklenmiştir. Bu sütun üzerinden eski ve yeni veriler kolayca filtrelenebilir.

## 🤖 Model Eğitimi (Finetuning)

Model eğitimi manuel olarak veya API üzerinden başlatılabilir.

- **Eğitim Scripti:** `./train_asr_service.sh <user_id>`
- **Arka Planda Eğitim:** `nohup python3 train_adapter.py <user_id> > logs/training.log 2>&1 &`

## 🌐 Ağ ve Port Bilgileri

- **API Port:** `8001`
- **Dış Erişim:** `http://88.255.236.143:8001` (VPN/Public)
- **Yerel Erişim:** `http://10.10.108.10:8001`

---
*Not: Bu notlar Gemini CLI tarafından sistem kurulumu sırasında oluşturulmuştur.*
