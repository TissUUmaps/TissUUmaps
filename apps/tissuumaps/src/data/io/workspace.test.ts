import { afterEach, describe, expect, it, vi } from "vitest";

import {
  isWorkspaceSupported,
  pickProjectFile,
  pickWorkspace,
  pickWorkspacePath,
  readDroppedItems,
  resolveWorkspacePath,
} from "./workspace";

const directory = {
  kind: "directory",
  name: "workspace",
} as FileSystemDirectoryHandle;
const projectFile = {
  kind: "file",
  name: "project.tm4",
} as FileSystemFileHandle;
const dataFile = {
  kind: "file",
  name: "cells.csv",
} as FileSystemFileHandle;

function makeWorkspace(segments: string[] | null): FileSystemDirectoryHandle {
  return {
    kind: "directory",
    name: "workspace",
    resolve: () => Promise.resolve(segments),
  } as unknown as FileSystemDirectoryHandle;
}

function stubDirectoryPicker(
  picker: ((options?: unknown) => Promise<FileSystemDirectoryHandle>) | null,
): void {
  vi.stubGlobal("showDirectoryPicker", picker ?? undefined);
}

function stubOpenFilePicker(
  picker: ((options?: unknown) => Promise<FileSystemFileHandle[]>) | null,
): void {
  vi.stubGlobal("showOpenFilePicker", picker ?? undefined);
}

describe("workspace", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  describe("isWorkspaceSupported", () => {
    it("returns false without the picker", () => {
      stubDirectoryPicker(null);
      expect(isWorkspaceSupported()).toBe(false);
    });

    it("returns true with the picker", () => {
      stubDirectoryPicker(() => Promise.resolve(directory));
      expect(isWorkspaceSupported()).toBe(true);
    });
  });

  describe("pickWorkspace", () => {
    it("returns the picked directory", async () => {
      const picker = vi.fn(() => Promise.resolve(directory));
      stubDirectoryPicker(picker);
      await expect(pickWorkspace()).resolves.toBe(directory);
      expect(picker).toHaveBeenCalledWith(
        expect.objectContaining({ mode: "read" }),
      );
    });

    it("returns null when the user cancels", async () => {
      stubDirectoryPicker(() =>
        Promise.reject(new DOMException("Aborted", "AbortError")),
      );
      await expect(pickWorkspace()).resolves.toBeNull();
    });

    it("rethrows other errors", async () => {
      const error = new DOMException("Denied", "NotAllowedError");
      stubDirectoryPicker(() => Promise.reject(error));
      await expect(pickWorkspace()).rejects.toBe(error);
    });

    it("rejects without the picker", async () => {
      stubDirectoryPicker(null);
      await expect(pickWorkspace()).rejects.toThrow(/not supported/);
    });
  });

  describe("pickProjectFile", () => {
    it("returns the picked file", async () => {
      const picker = vi.fn(() => Promise.resolve([projectFile]));
      stubOpenFilePicker(picker);
      await expect(pickProjectFile()).resolves.toBe(projectFile);
      expect(picker).toHaveBeenCalledWith(
        expect.objectContaining({ multiple: false }),
      );
    });

    it("opens the picker in the given directory", async () => {
      const picker = vi.fn(() => Promise.resolve([projectFile]));
      stubOpenFilePicker(picker);
      await pickProjectFile({ startIn: directory });
      expect(picker).toHaveBeenCalledWith(
        expect.objectContaining({ startIn: directory }),
      );
    });

    it("returns null when the picker returns no file", async () => {
      stubOpenFilePicker(() => Promise.resolve([]));
      await expect(pickProjectFile()).resolves.toBeNull();
    });

    it("returns null when the user cancels", async () => {
      stubOpenFilePicker(() =>
        Promise.reject(new DOMException("Aborted", "AbortError")),
      );
      await expect(pickProjectFile()).resolves.toBeNull();
    });

    it("rethrows other errors", async () => {
      const error = new DOMException("Denied", "NotAllowedError");
      stubOpenFilePicker(() => Promise.reject(error));
      await expect(pickProjectFile()).rejects.toBe(error);
    });

    it("rejects without the picker", async () => {
      stubOpenFilePicker(null);
      await expect(pickProjectFile()).rejects.toThrow(/not supported/);
    });
  });

  describe("pickWorkspacePath", () => {
    it("returns the workspace path of the picked file", async () => {
      const workspace = makeWorkspace(["data", "cells.csv"]);
      const picker = vi.fn(() => Promise.resolve([dataFile]));
      stubDirectoryPicker(() => Promise.resolve(directory));
      stubOpenFilePicker(picker);
      await expect(pickWorkspacePath(workspace, "file")).resolves.toBe(
        "/data/cells.csv",
      );
      expect(picker).toHaveBeenCalledWith({ startIn: workspace });
    });

    it("returns the workspace path of the picked directory", async () => {
      const workspace = makeWorkspace(["data", "image.ome.zarr"]);
      const picker = vi.fn(() => Promise.resolve(directory));
      stubDirectoryPicker(picker);
      stubOpenFilePicker(() => Promise.resolve([dataFile]));
      await expect(pickWorkspacePath(workspace, "directory")).resolves.toBe(
        "/data/image.ome.zarr",
      );
      expect(picker).toHaveBeenCalledWith({ mode: "read", startIn: workspace });
    });

    it("rejects the workspace itself", async () => {
      stubDirectoryPicker(() => Promise.resolve(directory));
      stubOpenFilePicker(() => Promise.resolve([dataFile]));
      await expect(
        pickWorkspacePath(makeWorkspace([]), "directory"),
      ).rejects.toThrow(/connected folder itself/);
    });

    it("rejects a file outside the workspace", async () => {
      stubDirectoryPicker(() => Promise.resolve(directory));
      stubOpenFilePicker(() => Promise.resolve([dataFile]));
      await expect(
        pickWorkspacePath(makeWorkspace(null), "file"),
      ).rejects.toThrow(/not in the connected folder/);
    });

    it("returns null when the user cancels", async () => {
      stubDirectoryPicker(() =>
        Promise.reject(new DOMException("Aborted", "AbortError")),
      );
      stubOpenFilePicker(() => Promise.resolve([dataFile]));
      await expect(
        pickWorkspacePath(makeWorkspace(["image.ome.zarr"]), "directory"),
      ).resolves.toBeNull();
    });
  });

  describe("resolveWorkspacePath", () => {
    it("returns the workspace path of a file in the workspace", async () => {
      await expect(
        resolveWorkspacePath(dataFile, makeWorkspace(["data", "cells.csv"])),
      ).resolves.toBe("/data/cells.csv");
    });

    it("rejects the workspace itself, and anything outside it", async () => {
      await expect(
        resolveWorkspacePath(directory, makeWorkspace([])),
      ).rejects.toThrow(/connected folder itself/);
      await expect(
        resolveWorkspacePath(dataFile, makeWorkspace(null)),
      ).rejects.toThrow(/not in the connected folder/);
    });
  });

  describe("readDroppedItems", () => {
    const file = new File([""], "cells.csv");

    it("reads the dropped files with their handles, if available", async () => {
      const dataTransfer = {
        items: [
          {
            kind: "file",
            getAsFile: () => file,
            getAsFileSystemHandle: () => Promise.resolve(dataFile),
          },
          { kind: "string", getAsFile: () => null },
          { kind: "file", getAsFile: () => file },
        ],
      } as unknown as DataTransfer;
      await expect(readDroppedItems(dataTransfer)).resolves.toEqual([
        { handle: dataFile, file },
        { handle: null, file },
      ]);
    });

    it("reads an item whose handle cannot be obtained without one", async () => {
      const dataTransfer = {
        items: [
          {
            kind: "file",
            getAsFile: () => file,
            getAsFileSystemHandle: () =>
              Promise.reject(new DOMException("Gone", "NotFoundError")),
          },
        ],
      } as unknown as DataTransfer;
      await expect(readDroppedItems(dataTransfer)).resolves.toEqual([
        { handle: null, file },
      ]);
    });
  });
});
