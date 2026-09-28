import { describe, expect, it, vi } from "vitest";
// The immer middleware types complete the project store's type, which this
// test program does not otherwise see
import type {} from "zustand/middleware/immer";

import { createProject } from "@tissuumaps/core";

import { projectStore } from "@/stores/project";

import {
  hasUnsavedChanges,
  loadProject,
  resolveProjectSource,
  saveProjectToSourceFile,
} from "./project";

const projectFile = {
  kind: "file",
  name: "project.tm4",
} as FileSystemFileHandle;

/**
 * Creates a workspace whose `resolve` returns the given segments, standing in
 * for a directory tree that does or does not contain the project file
 */
function makeWorkspace(segments: string[] | null) {
  const resolve = vi.fn(() => Promise.resolve(segments));
  const workspace = { kind: "directory", name: "", resolve };
  return {
    resolve,
    workspace: workspace as unknown as FileSystemDirectoryHandle,
  };
}

describe("resolveProjectSource", () => {
  it("returns the workspace-relative path of a file in the workspace", async () => {
    const { workspace, resolve } = makeWorkspace(["study", "project.tm4"]);
    await expect(resolveProjectSource(projectFile, workspace)).resolves.toBe(
      "/study/project.tm4",
    );
    expect(resolve).toHaveBeenCalledWith(projectFile);
  });

  it("returns the path of a file in the workspace root", async () => {
    const { workspace } = makeWorkspace(["project.tm4"]);
    await expect(resolveProjectSource(projectFile, workspace)).resolves.toBe(
      "/project.tm4",
    );
  });

  it("returns null for a file outside the workspace", async () => {
    const { workspace } = makeWorkspace(null);
    await expect(
      resolveProjectSource(projectFile, workspace),
    ).resolves.toBeNull();
  });

  it("returns null without an open workspace, without resolving", async () => {
    const { resolve } = makeWorkspace(["project.tm4"]);
    await expect(resolveProjectSource(projectFile, null)).resolves.toBeNull();
    expect(resolve).not.toHaveBeenCalled();
  });

  it("throws if aborted before resolving", async () => {
    const { workspace, resolve } = makeWorkspace(["project.tm4"]);
    await expect(
      resolveProjectSource(projectFile, workspace, {
        signal: AbortSignal.abort(),
      }),
    ).rejects.toThrow();
    expect(resolve).not.toHaveBeenCalled();
  });

  it("throws if aborted while resolving", async () => {
    const abortController = new AbortController();
    const resolve = vi.fn(() => {
      abortController.abort();
      return Promise.resolve(["project.tm4"]);
    });
    const workspace = {
      kind: "directory",
      name: "",
      resolve,
    } as unknown as FileSystemDirectoryHandle;
    await expect(
      resolveProjectSource(projectFile, workspace, {
        signal: abortController.signal,
      }),
    ).rejects.toThrow();
  });
});

describe("hasUnsavedChanges", () => {
  const project = createProject({ name: "Project" });
  const state = {
    ...project,
    source: null,
    sourceFile: null,
    instanceId: "instance",
    savedProject: project,
  };

  it("returns false if every part is the saved one", () => {
    expect(hasUnsavedChanges(state)).toBe(false);
  });

  it("returns true if a part was changed", () => {
    expect(hasUnsavedChanges({ ...state, name: "Renamed" })).toBe(true);
  });

  it("returns true if the background color was changed", () => {
    expect(
      hasUnsavedChanges({
        ...state,
        viewerBackgroundColor: { r: 1, g: 1, b: 1 },
      }),
    ).toBe(true);
  });

  it("returns true if a part was replaced by an equal copy", () => {
    expect(hasUnsavedChanges({ ...state, layers: [...state.layers] })).toBe(
      true,
    );
  });

  it("ignores where the project was loaded from", () => {
    expect(
      hasUnsavedChanges({ ...state, source: "https://example.com/p.tmap" }),
    ).toBe(false);
  });
});

/**
 * Creates a writable file stream whose methods resolve, recording their calls
 */
function makeWritable() {
  return {
    write: vi.fn<(data: string) => Promise<void>>(() => Promise.resolve()),
    close: vi.fn(() => Promise.resolve()),
    abort: vi.fn(() => Promise.resolve()),
  };
}

/**
 * Creates a project file within the workspace that opens the given stream
 */
function makeSourceFile(writable: ReturnType<typeof makeWritable>) {
  const createWritable = vi.fn(() => Promise.resolve(writable));
  return {
    createWritable,
    sourceFile: {
      kind: "file",
      name: "project.tmap",
      createWritable,
    } as unknown as FileSystemFileHandle,
  };
}

describe("saveProjectToSourceFile", () => {
  it("rejects a project that was not loaded from the workspace", async () => {
    loadProject(createProject({ name: "Project" }), null);
    await expect(saveProjectToSourceFile()).rejects.toThrow();
  });

  it("writes the project and marks it saved once the file is closed", async () => {
    const writable = makeWritable();
    const { sourceFile } = makeSourceFile(writable);
    loadProject(
      createProject({ name: "Project" }),
      "/project.tmap",
      sourceFile,
    );
    projectStore.getState().setName("Renamed");
    writable.close.mockImplementation(() => {
      expect(hasUnsavedChanges(projectStore.getState())).toBe(true);
      return Promise.resolve();
    });
    await saveProjectToSourceFile();
    expect(writable.write.mock.calls[0]?.[0]).toContain("Renamed");
    expect(hasUnsavedChanges(projectStore.getState())).toBe(false);
  });

  it("aborts the file and keeps the changes when writing fails", async () => {
    const writable = makeWritable();
    writable.write.mockRejectedValue(new Error("disk full"));
    writable.abort.mockRejectedValue(new Error("abort failed"));
    const { sourceFile } = makeSourceFile(writable);
    loadProject(
      createProject({ name: "Project" }),
      "/project.tmap",
      sourceFile,
    );
    projectStore.getState().setName("Renamed");
    await expect(saveProjectToSourceFile()).rejects.toThrow("disk full");
    expect(writable.abort).toHaveBeenCalled();
    expect(hasUnsavedChanges(projectStore.getState())).toBe(true);
  });

  it("does not mark a project loaded during the write as saved", async () => {
    const writable = makeWritable();
    const { sourceFile } = makeSourceFile(writable);
    loadProject(
      createProject({ name: "Project" }),
      "/project.tmap",
      sourceFile,
    );
    writable.write.mockImplementation(() => {
      loadProject(createProject({ name: "Other" }), null);
      projectStore.getState().setName("Other renamed");
      return Promise.resolve();
    });
    await saveProjectToSourceFile();
    expect(hasUnsavedChanges(projectStore.getState())).toBe(true);
  });
});
