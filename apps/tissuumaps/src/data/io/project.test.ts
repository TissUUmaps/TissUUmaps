import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Project, ProjectProvider } from "@tissuumaps/core";

import { projectStore } from "@/stores/project";

import {
  downloadProject,
  openProjectFromFile,
  openProjectFromFileHandle,
  openProjectFromURL,
  resolveProjectSource,
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

/**
 * Creates a project provider whose `load` returns a project named `Loaded`,
 * with the given source, or the requested one by default
 */
function makeProjectProvider(source?: string) {
  const load = vi.fn((normalizedSource: string) =>
    Promise.resolve({
      project: { name: "Loaded" },
      source: source ?? normalizedSource,
    }),
  );
  return { load, projectProvider: { name: "Test", load } as ProjectProvider };
}

describe("openProjectFromURL", () => {
  beforeEach(() => {
    projectStore.getState().clear();
  });

  it("resolves the URL against the document base URL", async () => {
    const { load, projectProvider } = makeProjectProvider();
    await openProjectFromURL(projectProvider, "projects/p.tm4");
    const expectedUrl = new URL("projects/p.tm4", document.baseURI).href;
    expect(load).toHaveBeenCalledWith(expectedUrl, { signal: undefined });
    expect(projectStore.getState().source).toBe(expectedUrl);
  });

  it("opens the project with the source the project provider reports", async () => {
    const redirectedUrl = "https://cdn.example/p.tm4";
    const { projectProvider } = makeProjectProvider(redirectedUrl);
    const prevInstanceId = projectStore.getState().instanceId;
    await openProjectFromURL(projectProvider, "https://data.example/p.tm4");
    const projectState = projectStore.getState();
    expect(projectState.name).toBe("Loaded");
    expect(projectState.source).toBe(redirectedUrl);
    expect(projectState.instanceId).not.toBe(prevInstanceId);
  });

  it("rejects for a project provider that cannot load projects", async () => {
    await expect(
      openProjectFromURL({ name: "Test" }, "https://data.example/p.tm4"),
    ).rejects.toThrow("Project provider Test cannot load projects.");
  });
});

describe("openProjectFromFile", () => {
  const objectUrl = "blob:http://localhost/0f8e";
  const revokeObjectURL = vi.fn();

  beforeEach(() => {
    projectStore.getState().clear();
    revokeObjectURL.mockClear();
    // jsdom implements neither of the two
    URL.createObjectURL = () => objectUrl;
    URL.revokeObjectURL = revokeObjectURL;
  });

  afterEach(() => {
    // @ts-expect-error removes what beforeEach stubbed
    delete URL.createObjectURL;
    // @ts-expect-error removes what beforeEach stubbed
    delete URL.revokeObjectURL;
  });

  it("opens the project without a source, and revokes the object URL", async () => {
    const { load, projectProvider } = makeProjectProvider();
    await openProjectFromFile(projectProvider, new File([], "p.tm4"));
    expect(load).toHaveBeenCalledWith(objectUrl, { signal: undefined });
    expect(projectStore.getState().name).toBe("Loaded");
    expect(projectStore.getState().source).toBeNull();
    expect(revokeObjectURL).toHaveBeenCalledWith(objectUrl);
  });

  it("revokes the object URL if loading fails", async () => {
    const projectProvider: ProjectProvider = {
      name: "Test",
      load: () => Promise.reject(new Error("Broken")),
    };
    await expect(
      openProjectFromFile(projectProvider, new File([], "p.tm4")),
    ).rejects.toThrow("Broken");
    expect(revokeObjectURL).toHaveBeenCalledWith(objectUrl);
  });
});

describe("openProjectFromFileHandle", () => {
  beforeEach(() => {
    projectStore.getState().clear();
  });

  it("loads a file in the workspace through its workspace-relative path", async () => {
    const { workspace } = makeWorkspace(["study", "project.tm4"]);
    const { load, projectProvider } = makeProjectProvider();
    await openProjectFromFileHandle(projectProvider, projectFile, workspace);
    expect(load).toHaveBeenCalledWith("/study/project.tm4", {
      signal: undefined,
      workspace,
    });
    expect(projectStore.getState().source).toBe("/study/project.tm4");
  });

  it("loads a file outside the workspace like an uploaded file", async () => {
    const objectUrl = "blob:http://localhost/0f8e";
    URL.createObjectURL = () => objectUrl;
    URL.revokeObjectURL = () => {};
    try {
      const { workspace } = makeWorkspace(null);
      const file = new File([], "project.tm4");
      const outsideFile = {
        ...projectFile,
        getFile: () => Promise.resolve(file),
      } as FileSystemFileHandle;
      const { load, projectProvider } = makeProjectProvider();
      await openProjectFromFileHandle(projectProvider, outsideFile, workspace);
      expect(load).toHaveBeenCalledWith(objectUrl, { signal: undefined });
      expect(projectStore.getState().source).toBeNull();
    } finally {
      // @ts-expect-error removes what was stubbed above
      delete URL.createObjectURL;
      // @ts-expect-error removes what was stubbed above
      delete URL.revokeObjectURL;
    }
  });
});

describe("downloadProject", () => {
  const objectUrl = "blob:http://localhost/0f8e";
  let clickedLinks: HTMLAnchorElement[] = [];

  beforeEach(() => {
    projectStore.getState().clear();
    clickedLinks = [];
    // jsdom implements neither of the two, nor navigation for the link
    URL.createObjectURL = () => objectUrl;
    URL.revokeObjectURL = () => {};
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      clickedLinks.push(this);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    // @ts-expect-error removes what beforeEach stubbed
    delete URL.createObjectURL;
    // @ts-expect-error removes what beforeEach stubbed
    delete URL.revokeObjectURL;
  });

  it("writes the open project without store state and downloads the file", async () => {
    projectStore.getState().setName("Downloaded");
    const writtenProjects: Project[] = [];
    await downloadProject({
      name: "Test",
      toFile: (project) => {
        writtenProjects.push(project);
        return Promise.resolve(new File([], "Downloaded.tm4"));
      },
    });
    expect(writtenProjects).toHaveLength(1);
    expect(writtenProjects[0]).toMatchObject({ name: "Downloaded" });
    expect(writtenProjects[0]).not.toHaveProperty("source");
    expect(writtenProjects[0]).not.toHaveProperty("instanceId");
    expect(writtenProjects[0]).not.toHaveProperty("setName");
    expect(clickedLinks).toHaveLength(1);
    expect(clickedLinks[0]?.download).toBe("Downloaded.tm4");
    expect(clickedLinks[0]?.href).toBe(objectUrl);
  });

  it("rejects for a project provider that cannot write projects", async () => {
    await expect(downloadProject({ name: "Test" })).rejects.toThrow(
      "Project provider Test cannot write projects.",
    );
  });
});
