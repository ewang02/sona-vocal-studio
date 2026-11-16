/**
 * Per-device storage for data too large for localStorage.
 *
 * localStorage is capped at roughly 5 MB per origin, which a few dozen edited
 * contours or a handful of card backgrounds exhaust. IndexedDB has no such
 * practical limit. Values written by earlier versions under the same key in
 * localStorage are read once, moved across, and removed.
 */

const DATABASE = "sona";
const STORE = "kv";

let opening: Promise<IDBDatabase> | null = null;

function open(): Promise<IDBDatabase> {
  if (opening) return opening;
  opening = new Promise<IDBDatabase>((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB is unavailable."));
      return;
    }
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) {
        request.result.createObjectStore(STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB could not open."));
    request.onblocked = () => reject(new Error("IndexedDB is blocked by another tab."));
  });
  // A failed open (private mode, blocked) should be retried on the next call.
  opening.catch(() => {
    opening = null;
  });
  return opening;
}

async function run<T>(
  mode: IDBTransactionMode,
  action: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const database = await open();
  return new Promise<T>((resolve, reject) => {
    const transaction = database.transaction(STORE, mode);
    const request = action(transaction.objectStore(STORE));
    transaction.oncomplete = () => resolve(request.result);
    transaction.onerror = () => reject(transaction.error ?? new Error("Storage failed."));
    transaction.onabort = () => reject(transaction.error ?? new Error("Storage was aborted."));
  });
}

function readLegacy(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function removeLegacy(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Nothing to clean up if storage is unavailable.
  }
}

/** Read a stored value, or null. Never throws: unreadable storage reads as empty. */
export async function loadStored<T>(
  key: string,
  parseLegacy?: (raw: string) => T | null,
): Promise<T | null> {
  try {
    const value = await run<T | undefined>("readonly", (store) => store.get(key));
    if (value !== undefined) return value;
  } catch {
    // Fall through to the legacy copy; it may be all this browser can offer.
  }
  if (!parseLegacy) return null;
  const legacy = readLegacy(key);
  if (legacy === null) return null;
  let parsed: T | null = null;
  try {
    parsed = parseLegacy(legacy);
  } catch {
    parsed = null;
  }
  if (parsed === null) return null;
  try {
    await run("readwrite", (store) => store.put(parsed, key));
    removeLegacy(key);
  } catch {
    // Keep the legacy copy if it could not be moved.
  }
  return parsed;
}

/** Store a value. Rejects if this browser cannot persist it. */
export async function saveStored<T>(key: string, value: T): Promise<void> {
  await run("readwrite", (store) => store.put(value, key));
  removeLegacy(key);
}

/** Remove a value from both the current and the legacy store. Never throws. */
export async function removeStored(key: string): Promise<void> {
  removeLegacy(key);
  try {
    await run("readwrite", (store) => store.delete(key));
  } catch {
    // Already absent as far as this browser can tell.
  }
}
