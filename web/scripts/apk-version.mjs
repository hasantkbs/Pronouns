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
