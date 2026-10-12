#!/bin/bash
set -e
cd "$(dirname "$0")"

APK_DIR="${APK_DIR:-/home/mayasoft/app/Pronouns/data/apk}"
VERSION_FILE="$APK_DIR/version.json"
read -r CODE NAME BUILT <<< "$(node scripts/apk-version.mjs next "$VERSION_FILE")"

npx vite build
npx cap sync android
(cd android && ./gradlew assembleRelease -PversionCode="$CODE" -PversionName="$NAME")

SRC="android/app/build/outputs/apk/release/app-release.apk"
DEST="$APK_DIR/pronouns.apk"
BACKUP="$APK_DIR/pronouns_backup_$(date +%Y%m%d_%H%M%S).apk"

cp "$DEST" "$BACKUP"
cp "$SRC" "$DEST"
# Surum APK yerine konduktan sonra yazilir: derleme basarisizsa numara artmaz
node scripts/apk-version.mjs write "$VERSION_FILE" "$CODE" "$NAME" "$BUILT"

echo "✅ Deployed: $DEST ($(stat -c%s "$DEST") bytes), versionCode $CODE ($NAME)"
echo "   Previous build backed up to: $BACKUP"
