#!/bin/bash
set -e
cd "$(dirname "$0")"
npm run cap:build:apk

SRC="android/app/build/outputs/apk/release/app-release.apk"
DEST="/home/mayasoft/app/Pronouns/data/apk/pronouns.apk"
BACKUP="/home/mayasoft/app/Pronouns/data/apk/pronouns_backup_$(date +%Y%m%d_%H%M%S).apk"

cp "$DEST" "$BACKUP"
cp "$SRC" "$DEST"

echo "✅ Deployed: $DEST ($(stat -c%s "$DEST") bytes)"
echo "   Previous build backed up to: $BACKUP"
