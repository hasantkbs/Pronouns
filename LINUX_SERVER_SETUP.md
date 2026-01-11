# Linux Server Setup and Optimization Guide

## 🐧 System Requirements

- **OS**: Linux (Ubuntu 20.04+ / CentOS 8+ / Debian 11+)
- **CPU**: Intel Xeon E5-2670 v3 (48 cores) @ 3.100GHz
- **GPU**: NVIDIA RTX A5000 (24GB VRAM)
- **CUDA**: 11.8+ (for RTX A5000)
- **Python**: 3.9+
- **RAM**: 64GB+ (recommended)

## 📦 Installation

### 1. System Updates

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# CentOS/RHEL
sudo yum update -y
```

### 2. CUDA and cuDNN Installation

```bash
# Check NVIDIA driver
nvidia-smi

# CUDA toolkit (if not installed)
# For Ubuntu:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install -y cuda-toolkit-11-8
```

### 3. Python Environment

```bash
# Check Python 3.9+
python3 --version

# Create virtual environment
python3 -m venv Pronouns
source Pronouns/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Project Configuration

```bash
# Navigate to project directory
cd /path/to/Pronouns

# Create log directory
mkdir -p logs runs

# Set permissions
chmod +x train_asr_service.sh
```

## ⚙️ Configuration

### config.py Settings

Special settings for Linux server:

```python
# Linux server settings
MULTIPROCESSING_START_METHOD = "fork"  # Faster on Linux
CUDA_VISIBLE_DEVICES = None  # Use all GPUs
LOG_DIR = "logs"
LOG_LEVEL = "INFO"
ENABLE_TENSORBOARD = True
```

### CUDA Device Selection

If you have multiple GPUs:

```python
# Use only GPU 0
CUDA_VISIBLE_DEVICES = "0"

# Use GPU 0 and 1
CUDA_VISIBLE_DEVICES = "0,1"
```

## 🚀 Usage

### Manual Training

```bash
# Activate environment
source Pronouns/bin/activate

# Data preparation
python3 prepare_training_data.py Furkan

# Model training
python3 train_adapter.py Furkan
```

### Training with Service Script

```bash
# Make script executable
chmod +x train_asr_service.sh

# Start training
./train_asr_service.sh Furkan
```

### Background Training (nohup)

```bash
# Run in background with nohup
nohup python3 train_adapter.py Furkan > logs/training.log 2>&1 &

# Save process ID
echo $! > training.pid

# Monitor training
tail -f logs/training.log

# Stop training
kill $(cat training.pid)
```

### Systemd Service (Optional)

1. **Edit service file**:
   ```bash
   sudo nano /etc/systemd/system/asr-training.service
   ```
   
   Update the following in `asr-training.service`:
   - `User=YOUR_USERNAME` → Actual username
   - `WorkingDirectory=/path/to/Pronouns` → Actual project path
   - `ExecStart=/usr/bin/python3 /path/to/Pronouns/train_adapter.py Furkan` → Actual paths

2. **Enable service**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable asr-training.service
   sudo systemctl start asr-training.service
   ```

3. **Check status**:
   ```bash
   sudo systemctl status asr-training.service
   sudo journalctl -u asr-training.service -f
   ```

## 📊 Monitoring

### GPU Monitoring

```bash
# Continuous GPU monitoring
watch -n 1 nvidia-smi

# Or
nvidia-smi -l 1

# Detailed GPU information
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv -l 1
```

### CPU and Memory Monitoring

```bash
# htop (install if needed: sudo apt install htop)
htop

# Or top
top

# Memory usage
free -h

# Disk usage
df -h
```

### Log Monitoring

```bash
# Monitor latest logs
tail -f logs/training_*.log

# Filter error logs
grep -i error logs/training_*.log

# Logs for specific user
ls -lt logs/training_Furkan_*.log | head -1 | xargs tail -f
```

## 🔧 Optimizations

### 1. Multiprocessing (Linux Fork)

On Linux, `fork` method is faster than `spawn`:
- Less overhead
- Faster process startup
- Better memory sharing

### 2. CUDA Optimizations

```bash
# Clear CUDA cache (if needed)
rm -rf ~/.nv/

# CUDA device selection
export CUDA_VISIBLE_DEVICES=0
```

### 3. System Limits

```bash
# Increase file descriptor limit
ulimit -n 65536

# Check process limit
ulimit -u
```

### 4. I/O Scheduler (for SSD)

```bash
# noop scheduler for SSD (optional)
echo noop | sudo tee /sys/block/nvme0n1/queue/scheduler
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch size (config.py)
FINETUNE_BATCH_SIZE = 12  # from 16 to 12
```

### Multiprocessing Errors

```bash
# Check fork method
python3 -c "import multiprocessing as mp; print(mp.get_start_method())"

# Switch to spawn if needed (config.py)
MULTIPROCESSING_START_METHOD = "spawn"
```

### Permission Errors

```bash
# Log directory permissions
chmod 755 logs
chown -R $USER:$USER logs/

# Model directory permissions
chmod 755 data/models/personalized_models
```

### Process Crashes

```bash
# Core dumps check
ulimit -c unlimited

# Crash logs
dmesg | tail -50
journalctl -xe
```

## 📈 Performance Tips

### 1. Data Preprocessing

```python
# Can be increased for 48 cores (config.py)
DATA_PREPROCESSING_NUM_PROC = 24  # from 16 to 24
```

### 2. DataLoader Workers

```python
# Can be adjusted based on CPU
DATALOADER_NUM_WORKERS = 12  # from 8 to 12
```

### 3. Batch Size

For RTX A5000:
- Minimum: 8
- Optimal: 16
- Maximum: 32 (depends on VRAM)

## 🔐 Security

### Firewall

```bash
# Open required ports (for TensorBoard)
sudo ufw allow 6006/tcp
```

### User Permissions

```bash
# Access only to required directories
chmod 750 data/models/personalized_models
chmod 750 logs
```

## 📝 Log Management

### Log Rotation

```bash
# logrotate configuration
sudo nano /etc/logrotate.d/asr-training

# Content:
/path/to/Pronouns/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### Log Cleanup

```bash
# Delete logs older than 7 days
find logs/ -name "*.log" -mtime +7 -delete
```

## 🎯 Conclusion

System optimized for Linux server:

✅ **Fork multiprocessing** (faster)  
✅ **Detailed logging** (Linux log files)  
✅ **Systemd service** support  
✅ **Resource monitoring** tools  
✅ **CUDA optimizations**  
✅ **Process management**  

The system is now production-ready for Linux server environments!
