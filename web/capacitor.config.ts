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
