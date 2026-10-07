import { describe, expect, it } from "vitest";

import { OMEZarrUtils } from "./OMEZarrUtils";

type FakeEntries = { [name: string]: string | FileSystemDirectoryHandle };

/** A workspace directory holding the given text files and subdirectories */
function makeDir(entries: FakeEntries): FileSystemDirectoryHandle {
  const get = (name: string, kind: "file" | "directory") => {
    const entry = entries[name];
    if (entry === undefined) {
      return Promise.reject(new DOMException(name, "NotFoundError"));
    }
    if ((typeof entry === "string") !== (kind === "file")) {
      return Promise.reject(new DOMException(name, "TypeMismatchError"));
    }
    return Promise.resolve(
      typeof entry === "string"
        ? {
            kind,
            name,
            getFile: () => Promise.resolve(new File([entry], name)),
          }
        : entry,
    );
  };
  return {
    kind: "directory",
    getDirectoryHandle: (name: string) => get(name, "directory"),
    getFileHandle: (name: string) => get(name, "file"),
  } as unknown as FileSystemDirectoryHandle;
}

const multiscales = [{ axes: [], datasets: [] }];

describe("OMEZarrUtils", () => {
  describe("isStoreSource", () => {
    it("checks every segment for the Zarr store extension", () => {
      expect(OMEZarrUtils.isStoreSource("/a/image.ome.zarr")).toBe(true);
      expect(OMEZarrUtils.isStoreSource("/a/x.zarr/labels/cells")).toBe(true);
      expect(
        OMEZarrUtils.isStoreSource("https://data.example/x.ZARR/images/a"),
      ).toBe(true);
      expect(OMEZarrUtils.isStoreSource("/a/image.tif")).toBe(false);
    });
  });

  describe("isZipSource", () => {
    it("checks for the zipped OME-Zarr extension", () => {
      expect(OMEZarrUtils.isZipSource("/a/image.OZX")).toBe(true);
      expect(OMEZarrUtils.isZipSource("/a/image.zarr")).toBe(false);
    });
  });

  describe("readImageName", () => {
    const workspace = makeDir({
      "named.zarr": makeDir({
        "zarr.json": JSON.stringify({
          zarr_format: 3,
          node_type: "group",
          attributes: { ome: { multiscales: [{ name: "Cells" }] } },
        }),
      }),
      "blank.zarr": makeDir({
        ".zgroup": JSON.stringify({ zarr_format: 2 }),
        ".zattrs": JSON.stringify({ multiscales: [{ name: " " }] }),
      }),
      "unnamed.zarr": makeDir({
        ".zgroup": JSON.stringify({ zarr_format: 2 }),
        ".zattrs": JSON.stringify({ multiscales }),
      }),
    });

    it("reads the name of the first multiscales", async () => {
      await expect(
        OMEZarrUtils.readImageName("/named.zarr", workspace),
      ).resolves.toBe("Cells");
    });

    it("ignores blank and missing names", async () => {
      await expect(
        OMEZarrUtils.readImageName("/blank.zarr", workspace),
      ).resolves.toBeUndefined();
      await expect(
        OMEZarrUtils.readImageName("/unnamed.zarr", workspace),
      ).resolves.toBeUndefined();
    });

    it("does not read zipped OME-Zarr files", async () => {
      await expect(
        OMEZarrUtils.readImageName("/named.ozx", workspace),
      ).resolves.toBeUndefined();
    });
  });

  describe("readAttributes", () => {
    const workspace = makeDir({
      "v05.zarr": makeDir({
        "zarr.json": JSON.stringify({
          zarr_format: 3,
          node_type: "group",
          attributes: { ome: { multiscales } },
        }),
      }),
      "v04.zarr": makeDir({
        ".zgroup": JSON.stringify({ zarr_format: 2 }),
        ".zattrs": JSON.stringify({ multiscales, "image-label": {} }),
      }),
      "none.zarr": makeDir({}),
    });

    it("reads the OME attributes of OME-Zarr 0.5 and 0.4 groups", async () => {
      await expect(
        OMEZarrUtils.readAttributes("/v05.zarr", workspace),
      ).resolves.toEqual({ multiscales });
      await expect(
        OMEZarrUtils.readAttributes("/v04.zarr", workspace),
      ).resolves.toEqual({ multiscales, "image-label": {} });
    });

    it("returns undefined for a directory that is no Zarr group", async () => {
      await expect(
        OMEZarrUtils.readAttributes("/none.zarr", workspace),
      ).resolves.toBeUndefined();
    });

    it("rejects if the operation is aborted", async () => {
      await expect(
        OMEZarrUtils.readAttributes("/v05.zarr", workspace, {
          signal: AbortSignal.abort(),
        }),
      ).rejects.toThrow();
    });
  });
});
