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
