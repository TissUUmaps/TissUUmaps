import { afterEach, describe, expect, it, vi } from "vitest";

import type { HierarchicalStoreArray } from "../HierarchicalStore";
import type { ZarrStore } from "./ZarrStore";
import { openZarrStore } from "./openZarrStore";

type FakeFile = { kind: "file"; name: string; getFile: () => Promise<File> };
type FakeDir = {
  kind: "directory";
  name: string;
  getDirectoryHandle: (name: string) => Promise<FakeDir>;
  getFileHandle: (name: string) => Promise<FakeFile>;
};

const encoder = new TextEncoder();
const tableAttrs = { "encoding-type": "anndata" };

function toJson(value: unknown): Uint8Array<ArrayBuffer> {
  return encoder.encode(JSON.stringify(value));
}

/**
 * Writes a Zarr v3 store with its metadata consolidated in the root
 * `zarr.json`, holding an int32 array `x` in the given group
 *
 * @returns The files of the store, by path below the store root
 */
function writeStore(group: string): Map<string, Uint8Array<ArrayBuffer>> {
  const files = new Map<string, Uint8Array<ArrayBuffer>>();
  const metadata: Record<string, unknown> = {};
  const writeNode = (path: string, node: unknown) => {
    metadata[path] = node;
    files.set(`/${path}/zarr.json`, toJson(node));
  };
  const parts = group.split("/");
  for (let i = 1; i <= parts.length; i++) {
    writeNode(parts.slice(0, i).join("/"), {
      zarr_format: 3,
      node_type: "group",
      attributes: i === parts.length ? tableAttrs : {},
    });
  }
  writeNode(`${group}/x`, {
    zarr_format: 3,
    node_type: "array",
    shape: [3],
    data_type: "int32",
    chunk_grid: { name: "regular", configuration: { chunk_shape: [3] } },
    chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
    fill_value: 0,
    codecs: [{ name: "bytes", configuration: { endian: "little" } }],
    attributes: {},
  });
  files.set(
    `/${group}/x/c/0`,
    new Uint8Array(new Int32Array([1, 2, 3]).buffer),
  );
  files.set(
    "/zarr.json",
    toJson({
      zarr_format: 3,
      node_type: "group",
      attributes: {},
      consolidated_metadata: {
        kind: "inline",
        must_understand: false,
        metadata,
      },
    }),
  );
  return files;
}

/**
 * Stubs `fetch` to serve the files of a store below a URL path
 *
 * @param statusOf - The status of a missing path, 404 by default
 */
function serveStore(
  storePath: string,
  files: Map<string, Uint8Array<ArrayBuffer>>,
  statusOf: (path: string) => number = () => 404,
) {
  const servedFiles = new Map(
    Array.from(files, ([path, bytes]) => [storePath + encodeURI(path), bytes]),
  );
  const fetchMock = vi.fn((request: Request) => {
    const path = new URL(request.url).pathname;
    const bytes = servedFiles.get(path);
    return Promise.resolve(
      bytes !== undefined
        ? new Response(bytes)
        : new Response(null, { status: statusOf(path) }),
    );
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function makeDir(
  name: string,
  entries: Record<string, FakeFile | FakeDir>,
): FakeDir {
  const open = (
    entryName: string,
    kind: FakeFile["kind"] | FakeDir["kind"],
  ) => {
    const entry = entries[entryName];
    if (entry === undefined) {
      return Promise.reject(
        new DOMException(`${entryName} not found`, "NotFoundError"),
      );
    }
    if (entry.kind !== kind) {
      return Promise.reject(
        new DOMException(
          `${entryName} is a ${entry.kind}`,
          "TypeMismatchError",
        ),
      );
    }
    return Promise.resolve(entry);
  };
  return {
    kind: "directory",
    name,
    getDirectoryHandle: (entryName) =>
      open(entryName, "directory") as Promise<FakeDir>,
    getFileHandle: (entryName) => open(entryName, "file") as Promise<FakeFile>,
  };
}

/** Builds a directory holding files by path below it */
function makeDirTree(
  name: string,
  files: Map<string, Uint8Array<ArrayBuffer>>,
): FakeDir {
  const entries: Record<string, FakeFile | FakeDir> = {};
  const dirFiles = new Map<string, Map<string, Uint8Array<ArrayBuffer>>>();
  for (const [path, bytes] of files) {
    const [entryName = "", ...rest] = path.split("/").filter(Boolean);
    if (rest.length === 0) {
      const file = new File([bytes], entryName);
      entries[entryName] = {
        kind: "file",
        name: entryName,
        getFile: () => Promise.resolve(file),
      };
    } else {
      const subFiles =
        dirFiles.get(entryName) ?? new Map<string, Uint8Array<ArrayBuffer>>();
      subFiles.set(`/${rest.join("/")}`, bytes);
      dirFiles.set(entryName, subFiles);
    }
  }
  for (const [dirName, subFiles] of dirFiles) {
    entries[dirName] = makeDirTree(dirName, subFiles);
  }
  return makeDir(name, entries);
}

async function expectTable(store: ZarrStore): Promise<void> {
  expect(await store.get("")).toMatchObject({
    kind: "group",
    attrs: tableAttrs,
    keys: ["x"],
  });
  const array = (await store.get("x")) as HierarchicalStoreArray;
  expect(array.kind).toBe("array");
  expect(await array.read()).toEqual(new Int32Array([1, 2, 3]));
}

describe("openZarrStore", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("opens a group below the store root of a URL", async () => {
    serveStore("/data/sdata.zarr", writeStore("tables/adata"));
    await expectTable(
      await openZarrStore("https://example.org/data/sdata.zarr/tables/adata"),
    );
  });

  it("opens a store at the root of a host", async () => {
    serveStore("", writeStore("tables/adata"));
    await expectTable(await openZarrStore("https://example.org/tables/adata"));
  });

  it("decodes the group path but keeps the root URL encoded", async () => {
    const fetchMock = serveStore(
      "/my%20data/sdata.zarr",
      writeStore("tables/my table"),
    );
    await expectTable(
      await openZarrStore(
        "https://example.org/my%20data/sdata.zarr/tables/my%20table",
      ),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "https://example.org/my%20data/sdata.zarr/zarr.json",
      }),
    );
  });

  it("walks past failures other than missing files", async () => {
    serveStore("/data/sdata.zarr", writeStore("tables/adata"), (path) =>
      path.startsWith("/data/sdata.zarr/tables/") ? 403 : 404,
    );
    await expectTable(
      await openZarrStore("https://example.org/data/sdata.zarr/tables/adata"),
    );
  });

  it("rejects if no store has consolidated metadata", async () => {
    const files = writeStore("tables/adata");
    files.set(
      "/zarr.json",
      toJson({ zarr_format: 3, node_type: "group", attributes: {} }),
    );
    serveStore("/data/sdata.zarr", files);
    const promise = openZarrStore(
      "https://example.org/data/sdata.zarr/tables/adata",
    );
    await expect(promise).rejects.toThrow(
      "No Zarr store with consolidated metadata",
    );
    await expect(promise).rejects.toMatchObject({
      cause: expect.any(Error) as unknown,
    });
  });

  it("opens a group below the store root of a workspace path", async () => {
    const workspace = makeDir("", {
      "sdata.zarr": makeDirTree("sdata.zarr", writeStore("tables/adata")),
    }) as unknown as FileSystemDirectoryHandle;
    await expectTable(
      await openZarrStore("/sdata.zarr/tables/adata", { workspace }),
    );
  });

  it("rejects with the reason of an aborted signal", async () => {
    const fetchMock = serveStore(
      "/data/sdata.zarr",
      writeStore("tables/adata"),
    );
    const controller = new AbortController();
    const reason = new Error("aborted");
    controller.abort(reason);
    await expect(
      openZarrStore("https://example.org/data/sdata.zarr/tables/adata", {
        signal: controller.signal,
      }),
    ).rejects.toBe(reason);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
