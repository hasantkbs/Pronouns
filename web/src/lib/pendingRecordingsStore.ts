import { openDB, type IDBPDatabase } from 'idb';

export interface PendingTake {
  id: number;
  word: string;
  audioBlob: Blob;
  recordedAt: string;
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
  now: Date = new Date(),
): Promise<PendingTake> {
  const db = await getDb();
  const record = { word, audioBlob, recordedAt: now.toISOString() };
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

export async function getPendingWords(): Promise<Set<string>> {
  const takes = await getAllTakes();
  return new Set(takes.map((t) => t.word));
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
