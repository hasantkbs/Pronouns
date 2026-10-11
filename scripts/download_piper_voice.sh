#!/usr/bin/env bash
# Piper Turkce sesini (tr_TR-dfki-medium, ~60 MB) bir kez indirir.
# Kullanim: bash scripts/download_piper_voice.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=data/models/tts
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium
mkdir -p "$DEST"

for f in tr_TR-dfki-medium.onnx tr_TR-dfki-medium.onnx.json; do
  if [ -s "$DEST/$f" ]; then
    echo "zaten var: $DEST/$f"
    continue
  fi
  curl -fSL --retry 3 -o "$DEST/$f.part" "$BASE/$f"
  mv "$DEST/$f.part" "$DEST/$f"
  echo "indirildi: $DEST/$f"
done
