import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  countPythonFiles,
  forgetPythonFiles,
  makePythonFiles,
  pyodideVersion,
} from "./python-files";

const cdn = `https://cdn.jsdelivr.net/pyodide/v${pyodideVersion}/full/`;
const sha256 = async (text: string) =>
  [
    ...new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)),
    ),
  ]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

// made-up files of Pyodide, with a list of packages
const files: Record<string, string> = {
  "pyodide.mjs": "export function loadPyodide() {}",
  "numpy.whl": "the bytes of numpy",
};
files["pyodide-lock.json"] = JSON.stringify({
  packages: {
    numpy: {
      file_name: "numpy.whl",
      sha256: await sha256(files["numpy.whl"]!),
    },
  },
});
const pinned = Object.fromEntries(
  await Promise.all(
    ["pyodide.mjs", "pyodide-lock.json"].map(async (name) => [
      name,
      { sha256: await sha256(files[name]!), bytes: files[name]!.length },
    ]),
  ),
) as Record<string, { sha256: string; bytes: number }>;

/** A Cache Storage in memory */
function makeCaches() {
  const stores = new Map<string, Map<string, Response>>();
  const open = (name: string) => {
    const store = stores.get(name) ?? new Map<string, Response>();
    stores.set(name, store);
    return Promise.resolve({
      match: (url: string) => Promise.resolve(store.get(url)?.clone()),
      put: (url: string, response: Response) => {
        store.set(url, response);
        return Promise.resolve();
      },
      keys: () => Promise.resolve([...store.keys()]),
    });
  };
  return {
    stores,
    open,
    keys: () => Promise.resolve([...stores.keys()]),
    delete: (name: string) => Promise.resolve(stores.delete(name)),
  };
}

const text = (buffer: ArrayBuffer) => new TextDecoder().decode(buffer);

let caches: ReturnType<typeof makeCaches>;
let served: Record<string, string>;
const fetchFile = vi.fn((url: string) => {
  const name = url.slice(cdn.length);
  const body = served[name];
  return Promise.resolve(
    body === undefined
      ? new Response("not found", { status: 404, statusText: "Not Found" })
      : new Response(body, {
          headers: { "Content-Length": String(body.length) },
        }),
  );
});

beforeEach(() => {
  caches = makeCaches();
  served = { ...files };
  fetchFile.mockClear();
  vi.stubGlobal("caches", caches);
  vi.stubGlobal("fetch", fetchFile);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("makePythonFiles", () => {
  it("downloads a file once, shows the download, and keeps the file in the browser", async () => {
    const progress: unknown[] = [];
    const python = makePythonFiles((step) => progress.push(step), {
      pinned,
    });
    expect(text(await python.get("pyodide.mjs"))).toBe(files["pyodide.mjs"]);
    expect(text(await python.get("pyodide.mjs"))).toBe(files["pyodide.mjs"]);
    expect(fetchFile).toHaveBeenCalledTimes(1);
    expect(progress.at(-2)).toEqual({ done: 32, total: 32 });
    expect(progress.at(-1)).toBeUndefined();
    expect([
      ...caches.stores.get(`tissuumaps-python-${pyodideVersion}`)!.keys(),
    ]).toEqual([`${cdn}pyodide.mjs`]);

    // another page, later: from the browser, without downloading
    const later = makePythonFiles(() => {}, { pinned });
    expect(text(await later.get("pyodide.mjs"))).toBe(files["pyodide.mjs"]);
    expect(fetchFile).toHaveBeenCalledTimes(1);
  });

  it("checks the packages against the list of packages", async () => {
    const python = makePythonFiles(() => {}, { pinned });
    expect(text(await python.get("numpy.whl"))).toBe(files["numpy.whl"]);
    expect(fetchFile.mock.calls.map(([url]) => url)).toEqual([
      `${cdn}pyodide-lock.json`,
      `${cdn}numpy.whl`,
    ]);
  });

  it("refuses other files, and files that differ, and keeps none of them", async () => {
    served["numpy.whl"] = "other bytes";
    const python = makePythonFiles(() => {}, { pinned });
    await expect(python.get("evil.js")).rejects.toThrow(
      `"evil.js" is not a file of Pyodide ${pyodideVersion}`,
    );
    await expect(python.get("numpy.whl")).rejects.toThrow(
      "numpy.whl of Python is not the expected file (its SHA-256 differs), so it was not used",
    );
    delete served["pyodide.mjs"];
    await expect(python.get("pyodide.mjs")).rejects.toThrow(
      "could not download pyodide.mjs of Python: 404 Not Found",
    );
    expect([
      ...caches.stores.get(`tissuumaps-python-${pyodideVersion}`)!.keys(),
    ]).toEqual([`${cdn}pyodide-lock.json`]);
  });

  it("works in a browser that keeps no files, and can be stopped", async () => {
    vi.stubGlobal("caches", undefined);
    const python = makePythonFiles(() => {}, { pinned });
    expect(text(await python.get("pyodide.mjs"))).toBe(files["pyodide.mjs"]);
    const controller = new AbortController();
    controller.abort(new Error("stopped"));
    await expect(
      python.get("pyodide.mjs", { signal: controller.signal }),
    ).rejects.toThrow("stopped");
  });
});

describe("countPythonFiles and forgetPythonFiles", () => {
  it("count and forget the files of Python of every version, and nothing else", async () => {
    await (await caches.open("tissuumaps-python-0.1")).put("a", new Response());
    const python = makePythonFiles(() => {}, { pinned });
    await python.get("pyodide.mjs");
    await caches.open("other");
    expect(await countPythonFiles()).toBe(2);
    expect(await forgetPythonFiles()).toBe(2);
    expect(await countPythonFiles()).toBe(0);
    expect([...caches.stores.keys()]).toEqual(["other"]);
  });
});
