# Canlı Arayüz ve Uygulama İçi Güncelleme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Furkanca APK'sı arayüzü canlı siteden yükleyen bir kabuğa dönüşsün; web değişiklikleri APK kurmadan gelsin, native değişikliklerde uygulamada "güncelleme var" şeridi çıksın.

**Architecture:** Capacitor `server.url` arayüzü `https://app.furkanca.algorixstudio.com`'dan yükler, internet yoksa APK içindeki `offline.html` açılır; `serve.json` html'in her açılışta yeniden doğrulanmasını sağlar. `deploy-android.sh` her derlemede `data/apk/version.json`'daki `versionCode`'u artırıp Gradle'a verir; sunucu `GET /apk/version` ile bu dosyayı sunar; ana sayfa `@capacitor/app` ile kurulu sürümü okuyup karşılaştırır ve `@capacitor/browser` ile `/apk`'yı açar.

**Tech Stack:** FastAPI + pytest (`/home/mayasoft/app/venv/bin/python`); React 18 + TypeScript + Vite + vitest; Capacitor 6 (`@capacitor/app` 6.0.3, `@capacitor/browser` 6.0.6); Gradle; `serve` (pronouns-web.service).

**Spec:** `docs/superpowers/specs/2026-10-12-canli-arayuz-design.md`

## Global Constraints

- Python: `/home/mayasoft/app/venv/bin/python`; sunucu takımı: `/home/mayasoft/app/venv/bin/python -m pytest tests/ --ignore=tests/test_utils.py -q -p no:cacheprovider`.
- Sunucu işi `/home/mayasoft/app/Pronouns`, dal `canli-arayuz` (açık, spec commit'i üstünde). İstemci işi `/home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite`, dal `worktree-react-rewrite`; `npm`/`npx` komutları `web/` altından.
- Canlı adres: `https://app.furkanca.algorixstudio.com`; çevrimdışı sayfa `offline.html`; `index.html` için `Cache-Control: no-cache`.
- Sürüm dosyası `/home/mayasoft/app/Pronouns/data/apk/version.json`; yoksa kayıtlı sürüm 1 (ilk derleme 2), `versionName` = `1.<versionCode-1>`; dosya yalnızca APK yerine konduktan sonra yazılır. `/apk/version` dosya yok/bozuksa `{"versionCode": 0}`, API anahtarı istemez.
- Güncelleme kontrolü yalnızca native platformda; her hatada `false` (uygulamayı asla engellemez).
- Plan sırasında **`npm run build` ve `deploy-android.sh` çalıştırılmaz** (canlı web ve APK'yı değiştirir; dağıtım kullanıcı onayıyla ayrı). `deploy-web.sh` hiç kullanılmaz.
- `data/users/FurkanV1/*` altındaki commit edilmemiş canlı veriye dokunma; her commit'te dosyaları tek tek `git add` et.
- Commit mesajları şu satırla biter: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **Uzaktan sayfada mikrofon izni:** Capacitor'ın `BridgeWebChromeClient.onPermissionRequest`'i izni kökene bakmadan veriyor (planlama sırasında kaynak kodda doğrulandı); kesin kanıt telefonda dağıtımdan sonra Furkanca ve Kayıt sayfalarında alınır. Çalışmazsa durulur, kullanıcıya danışılır.
2. **Eski `index.html`'in önbellekte kalması:** `serve.json` başlığı `/` ve `/kayit` gibi SPA yollarında `no-cache` döner (planlama sırasında yerel `serve` ile doğrulandı); `index.html`'e doğrudan istek 301 ile `/`'ye gider. Task 2 testi yapılandırmayı sabitler; davranış dağıtımdan sonra `curl -I` ile kontrol edilir.
3. **APK derlemesi yarıda kalırsa:** `set -e` ile betik durur, `version.json` yazılmaz, eski APK yerinde kalır. Task 3 testi okuma/yazma mantığını sabitler; betiğin sırası (yazma en sonda) kodda görünür.
4. **Kurulu sürüm okunamazsa (`App.getInfo().build` sayı değil):** `needsUpdate` `false` döner, şerit çıkmaz. Task 4 testi `NaN` durumunu sabitler.
5. **Eski kökendeki bekleyen kayıtlar:** kabuğa geçince görünmez olur; kod bunu çözmez, dağıtım adımı (önce "Tümünü Yükle") çözer. Spec'te ve dağıtım bölümünde yazılı.

---

### Task 1: `GET /apk/version` (sunucu)

**Files:**
- Modify: `api.py` (`import json`; `_API_KEY_EXEMPT_PATHS`; yeni `_apk_dir`, `apk_version`; `download_apk` `_apk_dir`'i kullanır)
- Test: `tests/test_apk_version.py`

**Interfaces:**
- Consumes: `config.BASE_PATH` (APK dizini onun kardeşi `apk/`, `/apk` ile aynı).
- Produces: `GET /apk/version` → `version.json` içeriği (`{"versionCode": int, "versionName": str, "built_at": str}`) ya da `{"versionCode": 0}`; API anahtarı gerekmez. Task 4'ün `getServerApkVersion` bunu okur.

- [ ] **Step 1: Write the failing test**

`tests/test_apk_version.py`:

```python
# -*- coding: utf-8 -*-
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient

import api
import config

client = TestClient(api.app)


@pytest.fixture
def apk_dir(tmp_path, monkeypatch):
    # /apk ile ayni yer: BASE_PATH'in kardesi olan apk/ dizini
    monkeypatch.setattr(config, "BASE_PATH", str(tmp_path / "users"))
    d = tmp_path / "apk"
    d.mkdir()
    return d


def test_returns_version_file_without_api_key(apk_dir):
    info = {"versionCode": 3, "versionName": "1.2", "built_at": "2026-10-12T12:00:00"}
    (apk_dir / "version.json").write_text(json.dumps(info), encoding="utf-8")
    res = client.get("/apk/version")
    assert res.status_code == 200
    assert res.json() == info


def test_missing_version_file_means_version_zero(apk_dir):
    assert client.get("/apk/version").json() == {"versionCode": 0}


def test_broken_version_file_means_version_zero(apk_dir):
    (apk_dir / "version.json").write_text("{bozuk", encoding="utf-8")
    assert client.get("/apk/version").json() == {"versionCode": 0}


def test_version_without_code_means_version_zero(apk_dir):
    (apk_dir / "version.json").write_text(json.dumps({"versionName": "1.0"}), encoding="utf-8")
    assert client.get("/apk/version").json() == {"versionCode": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_apk_version.py -q -p no:cacheprovider`
Expected: `4 failed` (uç nokta yok; anahtarsız istek 401).

- [ ] **Step 3: Implement**

Repo kökünde `git apply`:

```diff
diff --git a/api.py b/api.py
index 7933c7ac..2b466c8b 100644
--- a/api.py
+++ b/api.py
@@ -11,6 +11,7 @@ import shutil
 from pathlib import Path
 from typing import List, Optional, Dict, Any, Set
 import csv
+import json
 import hashlib
 
 import config
@@ -50,7 +51,7 @@ app.add_middleware(
 
 # API anahtarı ile korunmayan yollar: APK henüz kurulmamışken indirilebilmeli,
 # sentezlenmiş sesler audioplayers ile custom header olmadan oynatılıyor.
-_API_KEY_EXEMPT_PATHS = {"/apk"}
+_API_KEY_EXEMPT_PATHS = {"/apk", "/apk/version"}
 _API_KEY_EXEMPT_PREFIXES = ("/download/",)
 
 
@@ -588,10 +589,26 @@ p{{color:#666;line-height:1.7;margin-bottom:1.5rem;font-size:0.95rem}}
 </html>"""
 
 
+def _apk_dir() -> Path:
+    return Path(config.BASE_PATH).parent / "apk"
+
+
+@app.get("/apk/version")
+async def apk_version():
+    """Sunucudaki APK'nın sürümü; uygulama içi güncelleme şeridi bununla karşılaştırır."""
+    try:
+        info = json.loads((_apk_dir() / "version.json").read_text(encoding="utf-8"))
+        int(info["versionCode"])
+    except (OSError, ValueError, KeyError, TypeError):
+        # Dosya yoksa/bozuksa güncelleme önerilmez
+        return {"versionCode": 0}
+    return info
+
+
 @app.get("/apk")
 async def download_apk():
     """APK dosyasını indirir."""
-    apk_path = Path(config.BASE_PATH).parent / "apk" / "pronouns.apk"
+    apk_path = _apk_dir() / "pronouns.apk"
 
     _apk_logger.info(
         "APK download requested | path: %s | timestamp: %s",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/test_apk_version.py tests/test_api_key_middleware.py -q -p no:cacheprovider`
Expected: `11 passed`

- [ ] **Step 5: Run the full server suite**

Run: `/home/mayasoft/app/venv/bin/python -m pytest tests/ --ignore=tests/test_utils.py -q -p no:cacheprovider`
Expected: `116 passed` (112 + 4).

- [ ] **Step 6: Commit**

```bash
git add api.py tests/test_apk_version.py
git commit -m "feat: expose the published APK version for in-app updates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Kabuk APK — arayüzü canlı siteden yükle

Bu task'tan itibaren çalışma dizini: `/home/mayasoft/app/Pronouns/.claude/worktrees/react-rewrite` (git komutları buradan, `npm`/`npx` `web/` altından).

**Files:**
- Modify (tam içerik): `web/capacitor.config.ts`
- Create: `web/public/offline.html`, `web/public/serve.json`
- Test: `web/src/shell.test.ts`

**Interfaces:**
- Consumes: yok.
- Produces: `config.server.url = 'https://app.furkanca.algorixstudio.com'`, `config.server.errorPath = 'offline.html'`; `public/` Vite tarafından `dist/`'e kopyalanır (`serve` `dist/serve.json`'ı okur; `cap sync` `offline.html`'i APK'ya koyar).

- [ ] **Step 1: Write the failing test**

`web/src/shell.test.ts`:

```ts
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import config from '../capacitor.config';

const LIVE_URL = 'https://app.furkanca.algorixstudio.com';
const publicFile = (name: string) => new URL(`../public/${name}`, import.meta.url);

describe('kabuk APK', () => {
  it('loads the UI from the live site with a local offline page', () => {
    expect(config.server?.url).toBe(LIVE_URL);
    expect(config.server?.errorPath).toBe('offline.html');
  });

  it('ships an offline page that retries the live site without external resources', () => {
    expect(existsSync(publicFile('offline.html'))).toBe(true);
    const html = readFileSync(publicFile('offline.html'), 'utf-8');
    expect(html).toContain('İnternet bağlantısı yok');
    expect(html).toContain(LIVE_URL);
    expect(html).not.toMatch(/(src|href)="https?:/);
  });

  it('asks the server to revalidate html on every open', () => {
    const serve = JSON.parse(readFileSync(publicFile('serve.json'), 'utf-8'));
    expect(serve.headers).toContainEqual({
      source: '**/*.html',
      headers: [{ key: 'Cache-Control', value: 'no-cache' }],
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (`web/`): `npx vitest run src/shell.test.ts`
Expected: `3 failed` (`server` tanımsız; `offline.html` ve `serve.json` yok).

- [ ] **Step 3: Implement**

`web/capacitor.config.ts`:

```ts
import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.algorixstudio.furkanca',
  appName: 'Furkanca',
  webDir: 'dist',
  server: {
    // Arayuz canli siteden yuklenir: web degisiklikleri APK kurmadan gelir (npm run build).
    // APK yalnizca native degisikliklerde (izin, eklenti, simge) yeniden kurulur.
    url: 'https://app.furkanca.algorixstudio.com',
    // Internet yokken APK icindeki sayfa gosterilir (Capacitor eklentilerine erisemez)
    errorPath: 'offline.html',
  },
};

export default config;
```

`web/public/offline.html`:

```html
<!doctype html>
<html lang="tr">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Furkanca</title>
    <style>
      body { font-family: sans-serif; text-align: center; padding: 3rem 1rem; }
      button { font-size: 1.5rem; padding: 1rem 2rem; margin-top: 1rem; }
    </style>
  </head>
  <body>
    <h1>İnternet bağlantısı yok</h1>
    <p>Furkanca açılmak için internete bağlanmalı.</p>
    <button type="button" onclick="location.href = 'https://app.furkanca.algorixstudio.com'">
      Tekrar dene
    </button>
  </body>
</html>
```

`web/public/serve.json`:

```json
{
  "headers": [
    {
      "source": "**/*.html",
      "headers": [{ "key": "Cache-Control", "value": "no-cache" }]
    }
  ]
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (`web/`): `npx vitest run src/shell.test.ts && npx tsc -b`
Expected: `3 passed`; `tsc` çıktısız.

- [ ] **Step 5: Commit**

```bash
git add web/capacitor.config.ts web/public/offline.html web/public/serve.json web/src/shell.test.ts
git commit -m "feat: load the app UI from the live site with an offline fallback

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: APK sürüm numarası

**Files:**
- Create: `web/scripts/apk-version.mjs`
- Test: `web/scripts/apk-version.test.mjs`
- Modify (tam içerik): `web/deploy-android.sh`
- Modify: `web/android/app/build.gradle` (`versionCode`/`versionName` Gradle özelliklerinden)

**Interfaces:**
- Consumes: yok.
- Produces: `readVersionCode(file) -> number` (yok/bozuk → 1), `nextVersion(code, now?) -> {versionCode, versionName, built_at}`, `writeVersion(file, version)`; CLI `next FILE` → `"CODE NAME BUILT_AT"`, `write FILE CODE NAME BUILT_AT`. `deploy-android.sh` `APK_DIR` ortam değişkeniyle yönlendirilebilir (varsayılan `/home/mayasoft/app/Pronouns/data/apk`). Task 1'in `/apk/version`'ı bu dosyayı sunar.

- [ ] **Step 1: Write the failing test**

`web/scripts/apk-version.test.mjs`:

```js
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { nextVersion, readVersionCode, writeVersion } from './apk-version.mjs';

const tempFile = () => join(mkdtempSync(join(tmpdir(), 'apk-version-')), 'version.json');

describe('APK sürüm numarası', () => {
  it('starts from 1 when there is no version file, so the first build is 2', () => {
    const file = tempFile();
    expect(readVersionCode(file)).toBe(1);
    expect(nextVersion(readVersionCode(file)).versionCode).toBe(2);
  });

  it('treats a broken version file as version 1', () => {
    const file = tempFile();
    writeFileSync(file, '{bozuk');
    expect(readVersionCode(file)).toBe(1);
  });

  it('increments the stored version and derives the name', () => {
    const now = new Date('2026-10-12T12:00:00Z');
    expect(nextVersion(3, now)).toEqual({
      versionCode: 4,
      versionName: '1.3',
      built_at: '2026-10-12T12:00:00.000Z',
    });
  });

  it('reads back what it writes', () => {
    const file = tempFile();
    writeVersion(file, nextVersion(1));
    expect(readVersionCode(file)).toBe(2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (`web/`): `npx vitest run scripts/apk-version.test.mjs`
Expected: hata `Failed to load url ./apk-version.mjs`, `no tests`.

- [ ] **Step 3: Implement the version script**

`web/scripts/apk-version.mjs`:

```js
// APK surum numarasi: deploy-android.sh her derlemede bir artirir.
// Uygulama ici guncelleme seridi kurulu surumu sunucudaki version.json ile karsilastirir.
//
//   node scripts/apk-version.mjs next  FILE                 -> "CODE NAME BUILT_AT" (yazmaz)
//   node scripts/apk-version.mjs write FILE CODE NAME BUILT_AT
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

/** Kayitli surum; dosya yoksa ya da bozuksa 1 (ilk APK'larin surumu). */
export function readVersionCode(file) {
  try {
    const code = Number(JSON.parse(readFileSync(file, 'utf-8')).versionCode);
    return Number.isInteger(code) && code > 0 ? code : 1;
  } catch {
    return 1;
  }
}

export function nextVersion(currentCode, now = new Date()) {
  const versionCode = currentCode + 1;
  return { versionCode, versionName: `1.${versionCode - 1}`, built_at: now.toISOString() };
}

export function writeVersion(file, version) {
  writeFileSync(file, `${JSON.stringify(version, null, 2)}\n`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [command, file, code, name, builtAt] = process.argv.slice(2);
  if (command === 'next') {
    const v = nextVersion(readVersionCode(file));
    console.log(`${v.versionCode} ${v.versionName} ${v.built_at}`);
  } else if (command === 'write') {
    writeVersion(file, { versionCode: Number(code), versionName: name, built_at: builtAt });
  } else {
    console.error('usage: apk-version.mjs next FILE | write FILE CODE NAME BUILT_AT');
    process.exit(2);
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (`web/`): `npx vitest run scripts/apk-version.test.mjs && node scripts/apk-version.mjs next /nonexistent.json`
Expected: `4 passed`; ikinci komut `2 1.1 <ISO zaman>` yazar.

- [ ] **Step 5: Wire it into the build**

`web/deploy-android.sh` (tam içerik; eski `npm run cap:build:apk` yerine aynı üç adım, Gradle'a sürüm verilerek):

```bash
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
```

`web/android/app/build.gradle` (worktree kökünden `git apply`):

```diff
diff --git a/web/android/app/build.gradle b/web/android/app/build.gradle
index 4e8e0893..40545dad 100644
--- a/web/android/app/build.gradle
+++ b/web/android/app/build.gradle
@@ -7,8 +7,10 @@ android {
         applicationId "com.algorixstudio.furkanca"
         minSdkVersion rootProject.ext.minSdkVersion
         targetSdkVersion rootProject.ext.targetSdkVersion
-        versionCode 1
-        versionName "1.0"
+        // deploy-android.sh her derlemede artirir (-PversionCode/-PversionName); uygulama ici
+        // guncelleme seridi kurulu surumu sunucudaki data/apk/version.json ile karsilastirir
+        versionCode((project.findProperty('versionCode') ?: '1').toString().toInteger())
+        versionName((project.findProperty('versionName') ?: '1.0').toString())
         testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
         aaptOptions {
              // Files and dirs to omit from the packaged assets dir, modified to accommodate modern web apps.
```

- [ ] **Step 6: Verify the script parses and the gradle file is valid**

Run (`web/`): `bash -n deploy-android.sh && (cd android && ./gradlew -q help -PversionCode=7 -PversionName=1.6 > /dev/null) && echo ok`
Expected: `ok` (Gradle yapılandırması `versionCode`'u sayıya çevirebiliyor). Gerçek derleme dağıtımda yapılır; planlama sırasında aynı betik geçici bir `APK_DIR` ile çalıştırıldı ve `aapt dump badging` `versionCode='2' versionName='1.1'` gösterdi.

- [ ] **Step 7: Commit**

```bash
git add web/scripts/apk-version.mjs web/scripts/apk-version.test.mjs web/deploy-android.sh web/android/app/build.gradle
git commit -m "feat: bump the APK version on every Android build

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Uygulama içi güncelleme şeridi

**Files:**
- Modify: `web/package.json`, `web/package-lock.json` (eklentiler)
- Modify: `web/android/app/capacitor.build.gradle`, `web/android/capacitor.settings.gradle` (`npx cap sync android` üretir)
- Modify: `web/src/lib/api.ts` (`getServerApkVersion`), `web/src/lib/api.test.ts`
- Create: `web/src/lib/appUpdate.ts`, `web/src/lib/appUpdate.test.ts`
- Modify (tam içerik): `web/src/pages/HomePage.tsx`

**Interfaces:**
- Consumes: Task 1 `GET /apk/version`; mevcut `fetchWithTimeout` (`deadlines.ts`), `resolveUrl` (`api.ts`).
- Produces: `getServerApkVersion(): Promise<number>`; `UpdateDeps {isNative, installedVersion, serverVersion}`; `needsUpdate(installed, server): boolean`; `checkForUpdate(deps?): Promise<boolean>`; `openApkDownload(): Promise<void>`; ana sayfada şerit "Uygulama güncellemesi var · Güncelle".

- [ ] **Step 1: Install the plugins and register them natively**

Run (`web/`): `npm install @capacitor/app@^6.0.3 @capacitor/browser@^6.0.6 && npx cap sync android && git -C .. diff --stat -- web/android`
Expected: `package.json`'da iki eklenti; `capacitor.build.gradle` ve `capacitor.settings.gradle`'da `:capacitor-app` ve `:capacitor-browser` satırları:

```diff
diff --git a/web/android/app/capacitor.build.gradle b/web/android/app/capacitor.build.gradle
index fdb4970c..e4215d1b 100644
--- a/web/android/app/capacitor.build.gradle
+++ b/web/android/app/capacitor.build.gradle
@@ -9,7 +9,8 @@ android {
 
 apply from: "../capacitor-cordova-android-plugins/cordova.variables.gradle"
 dependencies {
-
+    implementation project(':capacitor-app')
+    implementation project(':capacitor-browser')
 
 }
 
diff --git a/web/android/capacitor.settings.gradle b/web/android/capacitor.settings.gradle
index 9a5fa872..fde830c1 100644
--- a/web/android/capacitor.settings.gradle
+++ b/web/android/capacitor.settings.gradle
@@ -1,3 +1,9 @@
 // DO NOT EDIT THIS FILE! IT IS GENERATED EACH TIME "capacitor update" IS RUN
 include ':capacitor-android'
 project(':capacitor-android').projectDir = new File('../node_modules/@capacitor/android/capacitor')
+
+include ':capacitor-app'
+project(':capacitor-app').projectDir = new File('../node_modules/@capacitor/app/android')
+
+include ':capacitor-browser'
+project(':capacitor-browser').projectDir = new File('../node_modules/@capacitor/browser/android')
```

(`cap sync` mevcut `dist/`'i APK varlıklarına kopyalar; canlı `dist/`'i değiştirmez.)

- [ ] **Step 2: Write the failing tests**

`web/src/lib/appUpdate.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { checkForUpdate, needsUpdate, type UpdateDeps } from './appUpdate';

function deps(overrides: Partial<UpdateDeps> = {}): UpdateDeps {
  return {
    isNative: () => true,
    installedVersion: async () => 2,
    serverVersion: async () => 2,
    ...overrides,
  };
}

describe('needsUpdate', () => {
  it('is true only when the server has a newer build', () => {
    expect(needsUpdate(2, 3)).toBe(true);
    expect(needsUpdate(3, 3)).toBe(false);
    expect(needsUpdate(4, 3)).toBe(false);
  });

  it('is false when the installed build is unknown', () => {
    expect(needsUpdate(Number.NaN, 3)).toBe(false);
  });
});

describe('checkForUpdate', () => {
  it('offers an update when the server build is newer', async () => {
    expect(await checkForUpdate(deps({ serverVersion: async () => 3 }))).toBe(true);
  });

  it('does not offer an update for the same build', async () => {
    expect(await checkForUpdate(deps())).toBe(false);
  });

  it('never checks in a web browser', async () => {
    let asked = false;
    const result = await checkForUpdate(
      deps({
        isNative: () => false,
        serverVersion: async () => {
          asked = true;
          return 9;
        },
      }),
    );
    expect(result).toBe(false);
    expect(asked).toBe(false);
  });

  it('stays quiet when the version request fails', async () => {
    const failing = deps({
      serverVersion: async () => {
        throw new Error('ağ yok');
      },
    });
    expect(await checkForUpdate(failing)).toBe(false);
  });
});
```

`web/src/lib/api.test.ts` (worktree kökünden `git apply`):

```diff
diff --git a/web/src/lib/api.test.ts b/web/src/lib/api.test.ts
index 9e27da22..d24921f6 100644
--- a/web/src/lib/api.test.ts
+++ b/web/src/lib/api.test.ts
@@ -1,5 +1,5 @@
 import { afterEach, describe, expect, it, vi } from 'vitest';
-import { getNextWord, getProgress, getWordSets, uploadRecording } from './api';
+import { getNextWord, getProgress, getServerApkVersion, getWordSets, uploadRecording } from './api';
 
 function stubFetch(body: unknown) {
   const fetchMock = vi.fn(
@@ -72,3 +72,16 @@ describe('kayıt türü (kind)', () => {
     expect(sentForm(fetchMock).get('kind')).toBe('word');
   });
 });
+
+describe('getServerApkVersion', () => {
+  it('returns the server build number', async () => {
+    const fetchMock = stubFetch({ versionCode: 3, versionName: '1.2' });
+    expect(await getServerApkVersion()).toBe(3);
+    expect(String(fetchMock.mock.calls[0][0])).toContain('/apk/version');
+  });
+
+  it('fails on a server error', async () => {
+    vi.stubGlobal('fetch', vi.fn(async () => new Response('x', { status: 500 })));
+    await expect(getServerApkVersion()).rejects.toThrow('500');
+  });
+});
```

- [ ] **Step 3: Run tests to verify they fail**

Run (`web/`): `npx vitest run src/lib/appUpdate.test.ts src/lib/api.test.ts`
Expected: `appUpdate.test.ts` yüklenemez (`Failed to load url ./appUpdate`); `api.test.ts`'te 2 test `getServerApkVersion is not a function` ile kalır.

- [ ] **Step 4: Implement**

`web/src/lib/api.ts` (worktree kökünden `git apply`):

```diff
diff --git a/web/src/lib/api.ts b/web/src/lib/api.ts
index c009110f..ff42295f 100644
--- a/web/src/lib/api.ts
+++ b/web/src/lib/api.ts
@@ -188,6 +188,16 @@ export async function translateSpeech(audioBlob: Blob): Promise<TranslateRespons
   return data;
 }
 
+const APK_VERSION_TIMEOUT_MS = 10000;
+
+/** Sunucudaki APK'nın sürüm numarası (versionCode); yoksa 0. */
+export async function getServerApkVersion(): Promise<number> {
+  const res = await fetchWithTimeout(`${API_BASE_URL}/apk/version`, {}, APK_VERSION_TIMEOUT_MS);
+  if (!res.ok) throw new Error(`Sürüm alınamadı: ${res.status}`);
+  const data = (await res.json()) as { versionCode?: number };
+  return Number(data.versionCode ?? 0);
+}
+
 export function resolveUrl(pathOrUrl: string): string {
   return pathOrUrl.startsWith('http') ? pathOrUrl : `${API_BASE_URL}${pathOrUrl}`;
 }
```

`web/src/lib/appUpdate.ts`:

```ts
import { App } from '@capacitor/app';
import { Browser } from '@capacitor/browser';
import { Capacitor } from '@capacitor/core';
import { getServerApkVersion, resolveUrl } from './api';

// Arayuz sunucudan yuklendigi icin bu kontrol yalnizca native (APK) degisiklikleri icindir.

export interface UpdateDeps {
  isNative: () => boolean;
  installedVersion: () => Promise<number>;
  serverVersion: () => Promise<number>;
}

const nativeDeps: UpdateDeps = {
  isNative: () => Capacitor.isNativePlatform(),
  installedVersion: async () => Number((await App.getInfo()).build),
  serverVersion: getServerApkVersion,
};

export function needsUpdate(installed: number, server: number): boolean {
  return Number.isFinite(installed) && server > installed;
}

/** Web'de ve her hatada false: guncelleme seridi uygulamayi asla engellemez. */
export async function checkForUpdate(deps: UpdateDeps = nativeDeps): Promise<boolean> {
  if (!deps.isNative()) return false;
  try {
    const [installed, server] = await Promise.all([deps.installedVersion(), deps.serverVersion()]);
    return needsUpdate(installed, server);
  } catch {
    return false;
  }
}

/** APK telefonun tarayicisinda iner; Android eski uygulamanin ustune kurar. */
export async function openApkDownload(): Promise<void> {
  await Browser.open({ url: resolveUrl('/apk') });
}
```

`web/src/pages/HomePage.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { checkForUpdate, openApkDownload } from '../lib/appUpdate';

export default function HomePage() {
  const [updateAvailable, setUpdateAvailable] = useState(false);

  useEffect(() => {
    let active = true;
    void checkForUpdate().then((available) => {
      if (active) setUpdateAvailable(available);
    });
    return () => {
      active = false;
    };
  }, []);

  return (
    <main>
      {updateAvailable && (
        <div role="status" style={{ background: '#fff3cd', padding: '0.75rem', marginBottom: '1rem' }}>
          <strong>Uygulama güncellemesi var</strong>{' '}
          <button type="button" onClick={() => void openApkDownload()}>
            Güncelle
          </button>
        </div>
      )}
      <h1>Furkanca</h1>
      <p>Konuşma Asistanı</p>
      <nav>
        <ul>
          <li>
            <Link to="/furkanca">Furkanca</Link>
          </li>
          <li>
            <Link to="/kayit">Kayıt</Link>
          </li>
          <li>
            <Link to="/ayarlar">Ayarlar</Link>
          </li>
        </ul>
      </nav>
    </main>
  );
}
```

- [ ] **Step 5: Run the whole client suite**

Run (`web/`): `npx tsc -b && npm test 2>&1 | tail -4`
Expected: `tsc` çıktısız; `Test Files  10 passed`, `Tests  64 passed`.

- [ ] **Step 6: Commit**

```bash
git add web/package.json web/package-lock.json web/android/app/capacitor.build.gradle web/android/capacitor.settings.gradle web/src/lib/api.ts web/src/lib/api.test.ts web/src/lib/appUpdate.ts web/src/lib/appUpdate.test.ts web/src/pages/HomePage.tsx
git commit -m "feat: offer native app updates from the home page

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Dağıtım ve elle doğrulama (plan sonrası, kullanıcı onayıyla)

1. Sunucu: `canli-arayuz` → `master` merge + push; kullanıcı `sudo systemctl restart pronouns-api.service`. Kontrol: `curl -s https://furkanca.algorixstudio.com/apk/version` → `{"versionCode":0}`.
2. Web: `cd .claude/worktrees/react-rewrite/web && npm run build`; ardından kullanıcı **bir kez** `sudo systemctl restart pronouns-web.service` (`serve` `serve.json`'ı yalnızca açılışta okur). Kontrol: `curl -sI https://app.furkanca.algorixstudio.com/kayit | grep -i cache-control` → `no-cache`. Doğrulanmadan 4. adıma geçilmez.
3. Furkan **eski** uygulamada Bekleyen Kayıtlar → "Tümünü Yükle" (kabukta eski bekleyenler görünmez).
4. APK: `web/deploy-android.sh` → `versionCode 2`; Furkan `/apk`'dan **üstüne** kurar (son elle kurulum). Telefonda: Furkanca'da konuşma (mikrofon), Kayıt'ta kelime ve cümle kaydı, uçak modunda açılışta `offline.html`.
5. Şerit denemesi (isteğe bağlı): ikinci `deploy-android.sh` → `versionCode 3`; uygulamayı açınca şerit çıkar, "Güncelle" ile üstüne kurulur.
6. Mikrofon uzaktan sayfada çalışmazsa: dur, kullanıcıya danış (geri dönüş: `server.url`'süz kodu yeni sürüm numarasıyla derlemek).
