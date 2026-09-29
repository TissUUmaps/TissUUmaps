import { afterEach, describe, expect, it, vi } from "vitest";

import {
  isWorkspaceSupported,
  pickProjectFile,
  pickWorkspace,
} from "./workspace";

const directory = {
  kind: "directory",
  name: "workspace",
} as FileSystemDirectoryHandle;
const projectFile = {
  kind: "file",
  name: "project.tmap",
} as FileSystemFileHandle;

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
});
