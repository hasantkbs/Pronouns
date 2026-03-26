# GPU desteği için NVIDIA CUDA tabanlı imaj
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    git \
    portaudio19-dev \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Bağımlılıkları kopyala ve yükle
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# API'yi başlat (Port 8000)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
