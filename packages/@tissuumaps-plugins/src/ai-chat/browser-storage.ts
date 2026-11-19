// The skills and the plugins that the user wants to keep are kept in
// IndexedDB, which, unlike localStorage, can hold files as they are. Both are
// in one database of the chat, each in its own part.

/** The database of the chat, named like its localStorage keys */
const databaseName = "tissuumaps.ai-chat";

/** The parts of the database, each holding entries by name or by id */
const parts = { skills: "name", plugins: "id" } as const;

/** A part of the database */
export type StoragePart = keyof typeof parts;

/**
 * Opens the database of the chat, and creates or completes it the first time
 *
 * @returns The database
 * @throws If the browser has no IndexedDB, or does not allow it
 */
function openDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === "undefined") {
    return Promise.reject(
      new Error(
        "this browser cannot remember skills and plugins (it has no IndexedDB)",
      ),
    );
  }
  return new Promise((resolve, reject) => {
    // version 1 had the skills only
    const request = indexedDB.open(databaseName, 2);
    request.onupgradeneeded = () => {
      for (const [part, keyPath] of Object.entries(parts)) {
        if (!request.result.objectStoreNames.contains(part)) {
          request.result.createObjectStore(part, { keyPath });
        }
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(
        request.error ?? new Error("the browser storage cannot be opened"),
      );
  });
}

/**
 * Runs one request on a part of the database, in its own transaction
 *
 * @param part - The part, e.g. "skills"
 * @param mode - Whether the request only reads
 * @param makeRequest - Makes the request
 * @param options - Optional abort signal
 * @returns The result of the request, once the transaction is complete
 */
export async function runRequest<TResult>(
  part: StoragePart,
  mode: IDBTransactionMode,
  makeRequest: (store: IDBObjectStore) => IDBRequest<TResult>,
  options?: { signal?: AbortSignal },
): Promise<TResult> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const database = await openDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = database.transaction(part, mode);
      const request = makeRequest(transaction.objectStore(part));
      transaction.oncomplete = () => resolve(request.result);
      transaction.onerror = () =>
        reject(transaction.error ?? new Error("the browser storage failed"));
      transaction.onabort = () =>
        reject(transaction.error ?? new Error("the browser storage failed"));
    });
  } finally {
    database.close();
  }
}
