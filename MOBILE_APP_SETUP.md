# Mobil Uygulama (Flutter) Bağlantı Rehberi

FurkanV1 modeli eğitimi tamamlandı ve API port 8001 üzerinden aktif. Mobil uygulamayı bu sunucuya bağlamak için aşağıdaki adımları izleyin.

## 1. Sunucu Bilgilerini Güncelleme

Flutter projesinde `lib/main.dart` dosyasındaki sabitleri sunucu IP adresinizle güncelleyin.

```dart
// lib/main.dart

// Mevcut IP yerine sunucunuzun IP adresini yazın
const String kBaseUrl = 'http://88.255.236.143:8001'; // Dış erişim (VPN/Public)
// Alternatif (Yerel ağ): 'http://10.10.108.10:8001'
```

> **Not:** Eğer sunucu yerel ağdaysa, mobil cihazın ve sunucunun aynı Wi-Fi ağına bağlı olduğundan emin olun.

## 2. Ağ İzinleri (Network Permissions)

Uygulamanın HTTP (şifresiz) bağlantı yapabilmesi için aşağıdaki yapılandırmaların yapıldığından emin olun:

### Android
`android/app/src/main/AndroidManifest.xml` dosyasında `application` etiketi içinde şu satır bulunmalıdır:
```xml
<application
    ...
    android:usesCleartextTraffic="true">
```

### iOS
`ios/Runner/Info.plist` dosyasına şu anahtarı ekleyin:
```xml
<key>NSAppTransportSecurity</key>
<dict>
  <key>NSAllowsArbitraryLoads</key>
  <true/>
</dict>
```

## 3. Mikrofon İzinleri
Uygulama zaten `permission_handler` kullanıyor, ancak cihaz ayarlarından mikrofon izninin verildiğini manuel olarak kontrol edin.

## 4. Test Etme
1. Uygulamayı derleyip çalıştırın.
2. **Kayıt** paneline girin ve sunucudan kelimelerin gelip gelmediğini kontrol edin.
3. **Furkanca** paneline girip bir ses kaydı alarak "Düzeltilmiş" metnin gelip gelmediğini test edin.

## 5. Sorun Giderme
- **Bağlantı Hatası:** Sunucuda `8001` portunun dış dünyaya açık olduğundan (Firewall/UFW ayarları) emin olun.
- **Kayıt Hatası:** Sunucu tarafındaki `api.log` dosyasını `tail -f api.log` komutuyla izleyerek hataları teşhis edin.
