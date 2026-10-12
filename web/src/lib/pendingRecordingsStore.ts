import { openDB, type IDBPDatabase } from 'idb';
import type { RecordKind } from './api';

export interface PendingTake {
  id: number;
  /** Kelime ya da cümle metni. */
  word: string;
  audioBlob: Blob;
  recordedAt: string;
  /** Cümle modundan önce kaydedilenlerde yok. */
  kind?: RecordKind;
}

/** Alanı olmayan eski kayıtlar kelimedir. */
export function takeKind(take: Pick<PendingTake, 'kind'>): RecordKind {
  return take.kind ?? 'word';
}

const DB_NAME = 'furkanca-pending-recordings';
const STORE_NAME = 'takes';
const WORD_INDEX = 'by-word';

let dbPromise: Promise<IDBPDatabase> | null = null;

function getDb(): Promise<IDBPDatabase> {
  if (!dbPromise) {
    dbPromise = openDB(DB_NAME, 1, {
      upgrade(db) {
        const store = db.createObjectStore(STORE_NAME, {
          keyPath: 'id',
          autoIncrement: true,
        });
        store.createIndex(WORD_INDEX, 'word');
      },
    });
  }
  return dbPromise;
}

export async function addTake(
  word: string,
  audioBlob: Blob,
  kind: RecordKind = 'word',
  now: Date = new Date(),
): Promise<PendingTake> {
  const db = await getDb();
  const record = { word, audioBlob, recordedAt: now.toISOString(), kind };
  const id = (await db.add(STORE_NAME, record)) as number;
  return { id, ...record };
}

export async function removeTake(id: number): Promise<void> {
  const db = await getDb();
  await db.delete(STORE_NAME, id);
}

export async function getAllTakes(): Promise<PendingTake[]> {
  const db = await getDb();
  return db.getAll(STORE_NAME);
}

export async function getTakesForWord(word: string): Promise<PendingTake[]> {
  const db = await getDb();
  return db.getAllFromIndex(STORE_NAME, WORD_INDEX, word);
}

export async function getPendingWords(kind: RecordKind = 'word'): Promise<Set<string>> {
  const takes = await getAllTakes();
  return new Set(takes.filter((t) => takeKind(t) === kind).map((t) => t.word));
}

export async function _resetForTests(): Promise<void> {
  if (dbPromise) {
    const db = await dbPromise;
    db.close();
  }
  dbPromise = null;
  await new Promise<void>((resolve, reject) => {
    const req = indexedDB.deleteDatabase(DB_NAME);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
    req.onblocked = () => resolve();
  });
}
