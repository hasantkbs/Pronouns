/** Ilerleme cubugu icin tam sayi yuzde; toplam yoksa 0, her zaman 0-100. */
export function progressPercent(completed: number, total: number): number {
  if (total <= 0) return 0;
  return Math.min(100, Math.max(0, Math.round((completed / total) * 100)));
}

/** Kayitlari kelimeye gore gruplar; kelimeler ilk goruldukleri sirada. */
export function groupByWord<T extends { word: string }>(takes: T[]): [string, T[]][] {
  const groups = new Map<string, T[]>();
  for (const take of takes) {
    const list = groups.get(take.word) ?? [];
    list.push(take);
    groups.set(take.word, list);
  }
  return Array.from(groups.entries());
}

/** Gecmis kartlarindaki saat: SS:DD:ss */
export function formatClock(date: Date): string {
  const pad = (n: number) => n.toString().padStart(2, '0');
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

/** Segment secenekleri: sunucudaki deger listede yoksa sona eklenir ki secili gorunsun. */
export function withCurrent(options: string[], current: string): string[] {
  return current && !options.includes(current) ? [...options, current] : options;
}
