# Linux Sunucu Kurulum ve Optimizasyon Kılavuzu

## 🐧 Sistem Gereksinimleri

- **OS**: Linux (Ubuntu 20.04+ / CentOS 8+ / Debian 11+)
- **CPU**: Intel Xeon E5-2670 v3 (48 çekirdek) @ 3.100GHz
- **GPU**: NVIDIA RTX A5000 (24GB VRAM)
- **CUDA**: 11.8+ (RTX A5000 için)
- **Python**: 3.9+
- **RAM**: 64GB+ (önerilen)

## 📦 Kurulum

### 1. Sistem Güncellemeleri

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# CentOS/RHEL
sudo yum update -y
```

### 2. CUDA ve cuDNN Kurulumu

```bash
# NVIDIA driver kontrolü
nvidia-smi

# CUDA toolkit (eğer yoksa)
# Ubuntu için:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install -y cuda-toolkit-11-8
```

### 3. Python Ortamı

```bash
# Python 3.9+ kontrolü
python3 --version

# Virtual environment oluştur
python3 -m venv Pronouns
source Pronouns/bin/activate

# Gerekli paketleri yükle
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Proje Yapılandırması

```bash
# Proje dizinine git
cd /path/to/Pronouns

# Log dizini oluştur
mkdir -p logs runs

# İzinleri ayarla
chmod +x train_asr_service.sh
```

## ⚙️ Konfigürasyon

### config.py Ayarları

Linux sunucu için özel ayarlar:

```python
# Linux sunucu ayarları
MULTIPROCESSING_START_METHOD = "fork"  # Linux'ta daha hızlı
CUDA_VISIBLE_DEVICES = None  # Tüm GPU'ları kullan
LOG_DIR = "logs"
LOG_LEVEL = "INFO"
ENABLE_TENSORBOARD = True
```

### CUDA Device Seçimi

Birden fazla GPU varsa:

```python
# Sadece GPU 0 kullan
CUDA_VISIBLE_DEVICES = "0"

# GPU 0 ve 1 kullan
CUDA_VISIBLE_DEVICES = "0,1"
```

## 🚀 Kullanım

### Manuel Eğitim

```bash
# Aktif environment
source Pronouns/bin/activate

# Veri hazırlama
python3 prepare_training_data.py Furkan

# Model eğitimi
python3 train_adapter.py Furkan
```

### Servis Script ile Eğitim

```bash
# Script'i çalıştırılabilir yap
chmod +x train_asr_service.sh

# Eğitimi başlat
./train_asr_service.sh Furkan
```

### Arka Planda Eğitim (nohup)

```bash
# nohup ile arka planda çalıştır
nohup python3 train_adapter.py Furkan > logs/training.log 2>&1 &

# Process ID'yi kaydet
echo $! > training.pid

# Eğitimi kontrol et
tail -f logs/training.log

# Eğitimi durdur
kill $(cat training.pid)
```

### Systemd Service (Opsiyonel)

1. **Service dosyasını düzenle**:
   ```bash
   sudo nano /etc/systemd/system/asr-training.service
   ```
   
   `asr-training.service` dosyasındaki şunları güncelle:
   - `User=YOUR_USERNAME` → Gerçek kullanıcı adı
   - `WorkingDirectory=/path/to/Pronouns` → Gerçek proje yolu
   - `ExecStart=/usr/bin/python3 /path/to/Pronouns/train_adapter.py Furkan` → Gerçek yollar

2. **Service'i etkinleştir**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable asr-training.service
   sudo systemctl start asr-training.service
   ```

3. **Durum kontrolü**:
   ```bash
   sudo systemctl status asr-training.service
   sudo journalctl -u asr-training.service -f
   ```

## 📊 Monitoring

### GPU İzleme

```bash
# Sürekli GPU izleme
watch -n 1 nvidia-smi

# Veya
nvidia-smi -l 1

# Detaylı GPU bilgisi
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv -l 1
```

### CPU ve Memory İzleme

```bash
# htop (eğer yoksa: sudo apt install htop)
htop

# Veya top
top

# Memory kullanımı
free -h

# Disk kullanımı
df -h
```

### Log İzleme

```bash
# Son logları izle
tail -f logs/training_*.log

# Hata loglarını filtrele
grep -i error logs/training_*.log

# Belirli bir kullanıcı için loglar
ls -lt logs/training_Furkan_*.log | head -1 | xargs tail -f
```

## 🔧 Optimizasyonlar

### 1. Multiprocessing (Linux Fork)

Linux'ta `fork` methodu `spawn`'dan daha hızlıdır:
- Daha az overhead
- Daha hızlı process başlatma
- Daha iyi memory sharing

### 2. CUDA Optimizasyonları

```bash
# CUDA cache temizleme (gerekirse)
rm -rf ~/.nv/

# CUDA device seçimi
export CUDA_VISIBLE_DEVICES=0
```

### 3. System Limits

```bash
# File descriptor limit artır
ulimit -n 65536

# Process limit kontrolü
ulimit -u
```

### 4. I/O Scheduler (SSD için)

```bash
# SSD için noop scheduler (opsiyonel)
echo noop | sudo tee /sys/block/nvme0n1/queue/scheduler
```

## 🐛 Sorun Giderme

### CUDA Out of Memory

```bash
# GPU memory kullanımını kontrol et
nvidia-smi

# Batch size'ı azalt (config.py)
FINETUNE_BATCH_SIZE = 12  # 16'dan 12'ye
```

### Multiprocessing Hataları

```bash
# Fork method kontrolü
python3 -c "import multiprocessing as mp; print(mp.get_start_method())"

# Gerekirse spawn'a geç (config.py)
MULTIPROCESSING_START_METHOD = "spawn"
```

### Permission Hataları

```bash
# Log dizini izinleri
chmod 755 logs
chown -R $USER:$USER logs/

# Model dizini izinleri
chmod 755 data/models/personalized_models
```

### Process Çökmesi

```bash
# Core dumps kontrolü
ulimit -c unlimited

# Crash logları
dmesg | tail -50
journalctl -xe
```

## 📈 Performans İpuçları

### 1. Veri Ön İşleme

```python
# config.py'de artırılabilir (48 çekirdek için)
DATA_PREPROCESSING_NUM_PROC = 24  # 16'dan 24'e
```

### 2. DataLoader Workers

```python
# CPU'ya göre ayarlanabilir
DATALOADER_NUM_WORKERS = 12  # 8'den 12'ye
```

### 3. Batch Size

RTX A5000 için:
- Minimum: 8
- Optimal: 16
- Maksimum: 32 (VRAM'e bağlı)

## 🔐 Güvenlik

### Firewall

```bash
# Gerekli portları aç (TensorBoard için)
sudo ufw allow 6006/tcp
```

### User Permissions

```bash
# Sadece gerekli dizinlere erişim
chmod 750 data/models/personalized_models
chmod 750 logs
```

## 📝 Log Yönetimi

### Log Rotation

```bash
# logrotate yapılandırması
sudo nano /etc/logrotate.d/asr-training

# İçerik:
/path/to/Pronouns/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### Log Temizleme

```bash
# 7 günden eski logları sil
find logs/ -name "*.log" -mtime +7 -delete
```

## 🎯 Sonuç

Linux sunucu için sistem optimize edildi:

✅ **Fork multiprocessing** (daha hızlı)  
✅ **Detaylı logging** (Linux log dosyaları)  
✅ **Systemd service** desteği  
✅ **Resource monitoring** araçları  
✅ **CUDA optimizasyonları**  
✅ **Process management**  

Sistem artık Linux sunucu ortamında production-ready durumda!

