// The files of Python (Pyodide) are downloaded by the panel, never by the
// sandbox, which has no network: only the files that are needed, when they
// are first needed. Each file is checked against its SHA-256 before it is
// used: the files of Pyodide itself against the values below, its packages
// against the list of packages (pyodide-lock.json), which is checked too.
// The checked files are kept in the Cache Storage of the page, so that Python
// starts without downloading anything the next time.

/** The version of Pyodide, whose files are pinned below */
export const pyodideVersion = "314.0.7";

/** Where the files of this version of Pyodide are downloaded from */
const pyodideUrl = `https://cdn.jsdelivr.net/pyodide/v${pyodideVersion}/full/`;

/** The beginning of the names of the caches of Python, one per version */
const cachePrefix = "tissuumaps-python-";

/** The cache of the files of this version */
const cacheName = `${cachePrefix}${pyodideVersion}`;

/** The files of Pyodide itself: their SHA-256, and their size in bytes */
const pinnedFiles: Record<string, { sha256: string; bytes: number }> = {
  "pyodide.mjs": {
    sha256: "6f1d60f7bf529beb300f0f47983c921d3982363640ba20af0e38efdddbc66109",
    bytes: 17_931,
  },
  "pyodide.asm.mjs": {
    sha256: "f7cdc8ece80678ceb712f8e65ebe6d3a83203a180c399865f49612a051693635",
    bytes: 1_250_344,
  },
  "pyodide.asm.wasm": {
    sha256: "cc36e3cab04fdfc9a63ff13eb52eae2b911bf46c025cc7b281f394bd3de1d5e6",
    bytes: 9_598_218,
  },
  "python_stdlib.zip": {
    sha256: "fa1957e5777068fc4f7437f96d860ae2fbe9c19732ba06c84e004ec16dd7dd7a",
    bytes: 2_545_637,
  },
  // the list of the packages, with their SHA-256
  "pyodide-lock.json": {
    sha256: "5dc2fc119108bc148c7457dc86e7675b5c87e1cafd420b9c34c1eaef7b36c010",
    bytes: 119_077,
  },
};

/** The files that Python needs to start */
export const coreFiles = [
  "pyodide.mjs",
  "pyodide.asm.mjs",
  "pyodide.asm.wasm",
  "python_stdlib.zip",
];

/** The name of the list of packages */
export const lockFile = "pyodide-lock.json";

/** How far the downloads are, in bytes; total is an estimate */
export type DownloadProgress = { done: number; total: number };

/** The files of Python, as the panel gives them to the sandbox */
export type PythonFiles = {
  /**
   * Gives a file of Pyodide: from the cache, or downloaded and checked
   *
   * @throws If the file is not one of Pyodide, cannot be downloaded, or is
   * not the expected file
   */
  get: (
    name: string,
    options?: { signal?: AbortSignal },
  ) => Promise<ArrayBuffer>;
};

/**
 * Writes bytes as hexadecimal digits
 *
 * @param buffer - The bytes
 * @returns e.g. "6f1d60f7..."
 */
function hex(buffer: ArrayBuffer): string {
  return [...new Uint8Array(buffer)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * The SHA-256 of a file, as hexadecimal digits
 *
 * @param buffer - The file
 * @returns The SHA-256
 * @throws If the page cannot compute it: only secure pages (https, or
 * localhost) can
 */
async function sha256(buffer: ArrayBuffer): Promise<string> {
  if (globalThis.crypto?.subtle === undefined) {
    throw new Error(
      "Python can only be checked, and so only used, in a secure page (https, or localhost)",
    );
  }
  return hex(await crypto.subtle.digest("SHA-256", buffer));
}

/**
 * The cache of the files of Python, if the page can keep files
 *
 * @returns The cache, or undefined
 */
async function openCache(): Promise<Cache | undefined> {
  try {
    return await globalThis.caches?.open(cacheName);
  } catch {
    // e.g. a page that is not secure, or storage that the browser refuses
    return undefined;
  }
}

/**
 * Makes the source of the files of Python for the sandbox
 *
 * @param onDownload - Called while files are downloaded, and with undefined
 * when there is nothing more to download
 * @param options - Other pinned files than those of Pyodide, for the tests
 * @returns The files
 */
export function makePythonFiles(
  onDownload: (progress: DownloadProgress | undefined) => void,
  options?: { pinned?: Record<string, { sha256: string; bytes: number }> },
): PythonFiles {
  const pinned = options?.pinned ?? pinnedFiles;
  /** The packages, by file name, with their SHA-256 */
  let packages: Map<string, string> | undefined;
  /** The files being given, so that each is downloaded once */
  const pending = new Map<string, Promise<ArrayBuffer>>();
  const progress: DownloadProgress = { done: 0, total: 0 };
  let downloading = 0;

  async function expectedSha256(
    name: string,
    signal: AbortSignal | undefined,
  ): Promise<string> {
    const file = pinned[name];
    if (file !== undefined) {
      return file.sha256;
    }
    if (packages === undefined) {
      const lock = JSON.parse(
        new TextDecoder().decode(await give(lockFile, signal)),
      ) as { packages: Record<string, { file_name: string; sha256: string }> };
      packages = new Map(
        Object.values(lock.packages).map((item) => [
          item.file_name,
          item.sha256,
        ]),
      );
    }
    const expected = packages.get(name);
    if (expected === undefined) {
      throw new Error(`"${name}" is not a file of Pyodide ${pyodideVersion}`);
    }
    return expected;
  }

  async function download(
    name: string,
    expected: string,
    signal: AbortSignal | undefined,
  ): Promise<ArrayBuffer> {
    const response = await fetch(pyodideUrl + name, { signal });
    if (!response.ok || response.body === null) {
      throw new Error(
        `could not download ${name} of Python: ${response.status} ${response.statusText}`,
      );
    }
    // the size, for the progress: the files of Pyodide itself are
    // compressed while they travel, so their own size is pinned; the
    // packages are compressed already, and travel at about their size
    const size =
      pinned[name]?.bytes ??
      Number(response.headers.get("Content-Length") ?? 0);
    downloading++;
    progress.total += size;
    onDownload({ ...progress });
    const parts: Uint8Array[] = [];
    let read = 0;
    try {
      const reader = response.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        parts.push(value);
        read += value.length;
        progress.done += value.length;
        progress.total = Math.max(progress.total, progress.done);
        onDownload({ ...progress });
      }
    } finally {
      // what was not read of the estimate of a file that was stopped
      if (read < size) {
        progress.total -= size - read;
      }
      downloading--;
      if (downloading === 0) {
        progress.done = 0;
        progress.total = 0;
        onDownload(undefined);
      }
    }
    const buffer = new Uint8Array(read);
    let offset = 0;
    for (const part of parts) {
      buffer.set(part, offset);
      offset += part.length;
    }
    if ((await sha256(buffer.buffer)) !== expected) {
      throw new Error(
        `${name} of Python is not the expected file (its SHA-256 differs), so it was not used`,
      );
    }
    return buffer.buffer;
  }

  async function give(
    name: string,
    signal: AbortSignal | undefined,
  ): Promise<ArrayBuffer> {
    const expected = await expectedSha256(name, signal);
    const cache = await openCache();
    const cached = await cache?.match(pyodideUrl + name);
    if (cached !== undefined) {
      return cached.arrayBuffer();
    }
    const buffer = await download(name, expected, signal);
    await cache?.put(pyodideUrl + name, new Response(buffer.slice(0)));
    return buffer;
  }

  return {
    get: async (name, options) => {
      const { signal } = options ?? {};
      signal?.throwIfAborted();
      let given = pending.get(name);
      if (given === undefined) {
        given = give(name, signal).finally(() => pending.delete(name));
        pending.set(name, given);
      }
      // a copy: the sandbox takes the one it gets
      return (await given).slice(0);
    },
  };
}

/**
 * Counts the files of Python that this browser keeps
 *
 * @returns The number of files, in all the versions
 */
export async function countPythonFiles(): Promise<number> {
  try {
    let count = 0;
    for (const name of (await globalThis.caches?.keys()) ?? []) {
      if (name.startsWith(cachePrefix)) {
        count += (await (await caches.open(name)).keys()).length;
      }
    }
    return count;
  } catch {
    return 0;
  }
}

/**
 * Forgets the files of Python that this browser keeps, of all the versions
 *
 * @returns The number of files forgotten
 */
export async function forgetPythonFiles(): Promise<number> {
  const count = await countPythonFiles();
  for (const name of (await globalThis.caches?.keys()) ?? []) {
    if (name.startsWith(cachePrefix)) {
      await caches.delete(name);
    }
  }
  return count;
}
