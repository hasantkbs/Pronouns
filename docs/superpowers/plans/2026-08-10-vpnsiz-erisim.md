# VPN'siz Erişim + API Anahtarı — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mobil uygulamanın VPN'e bağlanmadan sunucuya (genel IP) erişebilmesini sağlamak ve bu erişimi basit bir paylaşılan API anahtarıyla korumak.

**Architecture:** `api.py`'ye tüm state-değiştiren uçları koruyan bir Starlette middleware'i eklenir (`/apk` ve `/download/*` hariç); `lib/main.dart`'taki `kBaseUrl` genel IP'ye çevrilir ve tüm HTTP çağrılarına aynı anahtarı taşıyan bir header eklenir.

**Tech Stack:** FastAPI/Starlette middleware (Python), `package:http` (Dart/Flutter).

**Spec:** `docs/superpowers/specs/2026-08-10-vpnsiz-erisim-design.md`

## Global Constraints

- API anahtarı header adı: `X-API-Key`.
- Anahtar değeri (hem `config.py` hem `lib/main.dart`'ta **birebir aynı** olmalı): `c2a6dd1afc678eb93f0c46c8c1528e2b1c07676fbde0951f`
- Anahtarsız/yanlış anahtarlı istekler `401` ve `{"error": "invalid_api_key"}` JSON gövdesiyle reddedilir.
- Anahtar gerektirmeyen (whitelist) yollar: `/apk` ve `/download/` ile başlayan yollar. Başka hiçbir yol whitelist'e eklenmez.
- Yeni genel sunucu adresi: `http://88.255.236.143:8001`.
- Bu plan tamamlanmadan `2026-08-10-konusmaci-dogrulama.md` planı başlatılmamalı (o plan bu middleware'in var olduğunu varsayıyor).

---

### Task 1: Sunucu tarafında API anahtarı middleware'i

**Files:**
- Modify: `config.py`
- Modify: `api.py:1-6` (import satırları), `api.py:34-41` (CORS middleware bloğunun hemen altı)
- Test: `tests/test_api_key_middleware.py`

**Interfaces:**
- Consumes: `config.API_SECRET_KEY` (bu task'ta eklenir).
- Produces: Middleware, kendisinden sonraki her route'u otomatik korur. Sonraki task'lar (ve `2026-08-10-konusmaci-dogrulama.md` planı) yeni endpoint eklerken bu middleware'e dokunmaz — whitelist'e eklenmedikçe otomatik korunurlar.

- [ ] **Step 1: `config.py`'ye API anahtarı sabitini ekle**

`config.py` dosyasının sonuna ekle:

```python

# --- API Güvenliği ---
# Mobil uygulama tüm isteklerde bu değeri `X-API-Key` header'ında gönderir.
# lib/main.dart içindeki _kApiKey ile birebir aynı olmalı.
API_SECRET_KEY = "c2a6dd1afc678eb93f0c46c8c1528e2b1c07676fbde0951f"
```

- [ ] **Step 2: Başarısız testi yaz**

`tests/test_api_key_middleware.py` dosyasını oluştur:

```python
# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_request_without_key_is_rejected():
    res = client.get("/collect/word-sets")
    assert res.status_code == 401
    assert res.json()["error"] == "invalid_api_key"


def test_request_with_wrong_key_is_rejected():
    res = client.get(
        "/collect/word-sets", headers={"X-API-Key": "yanlis-anahtar"}
    )
    assert res.status_code == 401


def test_request_with_correct_key_is_allowed():
    res = client.get(
        "/collect/word-sets", headers={"X-API-Key": config.API_SECRET_KEY}
    )
    assert res.status_code == 200


def test_apk_download_is_exempt_from_key():
    res = client.get("/apk")
    assert res.status_code != 401


def test_download_path_is_exempt_from_key():
    res = client.get("/download/does-not-exist.wav")
    assert res.status_code != 401
```

- [ ] **Step 2b: Testin şu an başarısız olduğunu doğrula**

Run: `cd /home/mayasoft/app/Pronouns && source /home/mayasoft/app/venv/bin/activate && python -m pytest tests/test_api_key_middleware.py -v`
Expected: `test_request_without_key_is_rejected` ve `test_request_with_wrong_key_is_rejected` FAIL (çünkü şu an her istek 200 dönüyor, middleware henüz yok). Diğer üç test zaten PASS olabilir — bu normal.

- [ ] **Step 3: `api.py`'ye middleware'i ekle**

`api.py`'nin import bloğunu (satır 4-5) şu şekilde güncelle:

Eskisi:
```python
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
```

Yenisi:
```python
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
```

CORS middleware bloğunun (satır 34-40) hemen altına, `# Global sistemler` yorumundan önce ekle:

Eskisi:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global sistemler (Lazy loading için None)
```

Yenisi:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API anahtarı ile korunmayan yollar: APK henüz kurulmamışken indirilebilmeli,
# sentezlenmiş sesler audioplayers ile custom header olmadan oynatılıyor.
_API_KEY_EXEMPT_PATHS = {"/apk"}
_API_KEY_EXEMPT_PREFIXES = ("/download/",)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    path = request.url.path
    if path in _API_KEY_EXEMPT_PATHS or path.startswith(_API_KEY_EXEMPT_PREFIXES):
        return await call_next(request)
    if request.headers.get("x-api-key") != config.API_SECRET_KEY:
        return JSONResponse(status_code=401, content={"error": "invalid_api_key"})
    return await call_next(request)


# Global sistemler (Lazy loading için None)
```

- [ ] **Step 4: Testin artık geçtiğini doğrula**

Run: `cd /home/mayasoft/app/Pronouns && source /home/mayasoft/app/venv/bin/activate && python -m pytest tests/test_api_key_middleware.py -v`
Expected: 5/5 PASS.

- [ ] **Step 5: Commit**

```bash
git add config.py api.py tests/test_api_key_middleware.py
git commit -m "feat: require X-API-Key header on all API routes except APK/audio download"
```

---

### Task 2: İstemci tarafında genel adrese geçiş + API anahtarı

**Files:**
- Modify: `lib/main.dart:18-21` (sabitler), `lib/main.dart:48` (`_base()` fonksiyonunun altı), ve her `http.get`/`http.post`/`http.MultipartRequest` çağrı noktası.

**Interfaces:**
- Consumes: Task 1'de sunucuya eklenen `X-API-Key` kontrolü (aynı sabit değer).
- Produces: `_authHeaders()` yardımcı fonksiyonu — bundan sonraki her yeni ağ çağrısı bunu kullanmalı.

- [ ] **Step 1: Sabitleri güncelle**

`lib/main.dart:18-21`:

Eskisi:
```dart
const String kBaseUrl = 'http://10.10.108.10:8001';
const String kUserId = 'FurkanV1';
const String _kAdminUsername = 'admin';
const String _kAdminPassword = 'adminpassword';
```

Yenisi:
```dart
const String kBaseUrl = 'http://88.255.236.143:8001';
const String kUserId = 'FurkanV1';
const String _kAdminUsername = 'admin';
const String _kAdminPassword = 'adminpassword';
const String _kApiKey = 'c2a6dd1afc678eb93f0c46c8c1528e2b1c07676fbde0951f';
```

- [ ] **Step 2: `_authHeaders` yardımcı fonksiyonunu ekle**

`lib/main.dart:48` (`String _base() => ...` satırının hemen altına) ekle:

```dart
Map<String, String> _authHeaders([Map<String, String>? extra]) =>
    {'X-API-Key': _kApiKey, ...?extra};
```

- [ ] **Step 3: `/translate` isteğine header ekle (`_FurkancaPageState._handleSegment`)**

Eskisi (satır ~460-462):
```dart
      final req = http.MultipartRequest('POST', uri)
        ..fields['user_id'] = kUserId
        ..files.add(await http.MultipartFile.fromPath('audio', file.path));
```

Yenisi:
```dart
      final req = http.MultipartRequest('POST', uri)
        ..headers.addAll(_authHeaders())
        ..fields['user_id'] = kUserId
        ..files.add(await http.MultipartFile.fromPath('audio', file.path));
```

- [ ] **Step 4: `_KayitPageState`'teki üç GET isteğine header ekle**

Eskisi (satır ~695):
```dart
      final res = await http.get(Uri.parse('${_base()}/collect/word-sets'));
```
Yenisi:
```dart
      final res =
          await http.get(Uri.parse('${_base()}/collect/word-sets'), headers: _authHeaders());
```

Eskisi (satır ~709-711, `_refreshWord`):
```dart
      final uri = Uri.parse('${_base()}/collect/next-word?user_id=$kUserId'
          '&set_file=$_setFile&exclude=${Uri.encodeQueryComponent(exclude)}');
      final res = await http.get(uri);
```
Yenisi:
```dart
      final uri = Uri.parse('${_base()}/collect/next-word?user_id=$kUserId'
          '&set_file=$_setFile&exclude=${Uri.encodeQueryComponent(exclude)}');
      final res = await http.get(uri, headers: _authHeaders());
```

Eskisi (satır ~730-732, `_refreshProgress`):
```dart
      final uri = Uri.parse(
          '${_base()}/collect/progress?user_id=$kUserId&set_file=$_setFile');
      final res = await http.get(uri);
```
Yenisi:
```dart
      final uri = Uri.parse(
          '${_base()}/collect/progress?user_id=$kUserId&set_file=$_setFile');
      final res = await http.get(uri, headers: _authHeaders());
```

- [ ] **Step 5: `_PendingUploadsPageState._uploadAll`'daki `/record` isteğine header ekle**

Eskisi (satır ~1102-1107):
```dart
        final req = http.MultipartRequest(
            'POST', Uri.parse('${_base()}/record'))
          ..fields['user_id'] = kUserId
          ..fields['word'] = take.word
          ..files
              .add(await http.MultipartFile.fromPath('audio', take.filePath));
```
Yenisi:
```dart
        final req = http.MultipartRequest(
            'POST', Uri.parse('${_base()}/record'))
          ..headers.addAll(_authHeaders())
          ..fields['user_id'] = kUserId
          ..fields['word'] = take.word
          ..files
              .add(await http.MultipartFile.fromPath('audio', take.filePath));
```

- [ ] **Step 6: `_AyarlarPageState`'teki dört isteğe header ekle**

Eskisi (satır ~1393-1396, `_fetchSettings` → `/settings` GET):
```dart
      final res = await http.get(
        Uri.parse('${_base()}/settings?user_id=$kUserId'),
      );
```
Yenisi:
```dart
      final res = await http.get(
        Uri.parse('${_base()}/settings?user_id=$kUserId'),
        headers: _authHeaders(),
      );
```

Eskisi (satır ~1414-1417, `/model/info` GET):
```dart
      final infoRes = await http.get(
        Uri.parse('${_base()}/model/info?user_id=$kUserId'),
      );
```
Yenisi:
```dart
      final infoRes = await http.get(
        Uri.parse('${_base()}/model/info?user_id=$kUserId'),
        headers: _authHeaders(),
      );
```

Eskisi (satır ~1435-1447, `_saveSettings` → `/settings` POST):
```dart
      final res = await http.post(
        Uri.parse('${_base()}/settings'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
```
Yenisi:
```dart
      final res = await http.post(
        Uri.parse('${_base()}/settings'),
        headers: _authHeaders({'Content-Type': 'application/json'}),
        body: jsonEncode({
```

Eskisi (satır ~1467-1477, `_startFineTune` → `/fine-tune` POST):
```dart
      final res = await http.post(
        Uri.parse('${_base()}/fine-tune'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
```
Yenisi:
```dart
      final res = await http.post(
        Uri.parse('${_base()}/fine-tune'),
        headers: _authHeaders({'Content-Type': 'application/json'}),
        body: jsonEncode({
```

- [ ] **Step 7: Statik analiz ve mevcut testlerin bozulmadığını doğrula**

Run: `cd /home/mayasoft/app/Pronouns && flutter analyze`
Expected: Hata yok (mevcut lint uyarıları dışında yeni hata çıkmamalı).

Run: `cd /home/mayasoft/app/Pronouns && flutter test`
Expected: Tüm testler PASS (mevcut testler ağ çağrısı yapmıyor, bu değişiklikten etkilenmemeli).

- [ ] **Step 8: Commit**

```bash
git add lib/main.dart
git commit -m "feat: switch app to public server address and send X-API-Key on every request"
```

---

### Task 3: Sunucuyu güncelle ve uçtan uca doğrula

**Files:** Yok (kod değişikliği içermez — dağıtım/operasyon adımı).

**Interfaces:**
- Consumes: Task 1 ve Task 2'nin tamamlanmış hali.
- Produces: Çalışır durumda, VPN gerektirmeyen bir API + APK.

- [ ] **Step 1: Sunucu servisini yeniden başlat**

Run: `sudo systemctl restart pronouns-api.service`
Expected: `sudo systemctl status pronouns-api.service` çıktısı `active (running)` göstermeli.

- [ ] **Step 2: Anahtarsız isteğin reddedildiğini genel IP üzerinden doğrula**

Run: `curl -s -o /dev/null -w "%{http_code}\n" http://88.255.236.143:8001/collect/word-sets`
Expected: `401`

- [ ] **Step 3: Doğru anahtarla isteğin geçtiğini doğrula**

Run: `curl -s -H "X-API-Key: c2a6dd1afc678eb93f0c46c8c1528e2b1c07676fbde0951f" http://88.255.236.143:8001/collect/word-sets`
Expected: `{"sets": [...]}` (200, JSON gövde).

- [ ] **Step 4: APK indirmenin anahtarsız çalıştığını doğrula**

Run: `curl -s -o /dev/null -w "%{http_code}\n" http://88.255.236.143:8001/apk`
Expected: `401` DEĞİL (200 veya APK akışına özgü bir kod; kesinlikle 401 olmamalı).

- [ ] **Step 5: Yeni APK'yı derle ve telefonda manuel test et**

Run: `cd /home/mayasoft/app/Pronouns && flutter build apk --debug`

Telefonda (VPN kapalıyken, sadece mobil veri/Wi-Fi ile):
1. Yeni APK'yı kur, uygulamayı aç.
2. Furkanca ekranında "Başlat" → konuş → yanıtın geldiğini doğrula.
3. Kayıt ekranında bir kelime kaydet, "Bekleyen Kayıtlar" → "Tümünü Yükle" → başarılı yüklendiğini doğrula.
4. Ayarlar'a admin/adminpassword ile gir, ayarları kaydet, model bilgisinin yüklendiğini doğrula.

Bu adımlar manuel; otomatik test kapsamında değildir (spec'in test planında da böyle belirtilmiştir).
