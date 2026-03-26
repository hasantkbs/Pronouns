#!/bin/bash
# Linux sunucu için model eğitim scripti
# Kullanım: ./train_asr_service.sh Furkan

set -e  # Hata durumunda dur

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parametreler
USER_ID=${1:-"Furkan"}
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ASR Model Eğitim - Linux Sunucu${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Python environment kontrolü
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 bulunamadı!${NC}"
    exit 1
fi

# CUDA kontrolü
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU bulundu${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo -e "${YELLOW}⚠️  NVIDIA GPU bulunamadı, CPU kullanılacak${NC}"
    echo ""
fi

# Venv kontrolü ve aktivasyonu
VENV_PATH="./venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${GREEN}✅ Sanal ortam bulundu: ${VENV_PATH}${NC}"
    PYTHON_EXE="$VENV_PATH/bin/python"
else
    echo -e "${YELLOW}⚠️  Sanal ortam bulunamadı, sistem python'ı kullanılacak${NC}"
    PYTHON_EXE="python3"
fi

# Log dizini oluştur
mkdir -p "$LOG_DIR"

# Eğitim başlat
echo -e "${GREEN}🚀 Model eğitimi başlatılıyor...${NC}"
echo -e "Kullanıcı: ${YELLOW}${USER_ID}${NC}"
echo -e "Log dosyası: ${YELLOW}${LOG_DIR}/training_${USER_ID}_${TIMESTAMP}.log${NC}"
echo ""

# Eğitimi başlat (nohup ile arka planda çalıştırılabilir)
$PYTHON_EXE train_adapter.py "$USER_ID" 2>&1 | tee "${LOG_DIR}/training_${USER_ID}_${TIMESTAMP}.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Eğitim başarıyla tamamlandı!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Eğitim sırasında hata oluştu (Exit code: $EXIT_CODE)${NC}"
    exit $EXIT_CODE
fi

