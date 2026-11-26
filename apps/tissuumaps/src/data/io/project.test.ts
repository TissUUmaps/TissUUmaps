import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Project, ProjectProvider, RawProject } from "@tissuumaps/core";

import { appStore } from "@/stores/app";
import { projectStore } from "@/stores/project";

import {
  appendProjectURLParam,
  clearProjectURLParams,
  downloadProject,
  findLocalProjectProviders,
  findProjectProviders,
  getProjectFileExtensions,
  getProjectFileTypes,
  hasUnsavedChanges,
  loadLocalProject,
  loadProjectFromURL,
  mergeProject,
  openProject,
  readProjectURLParams,
  resolveProjectSource,
  saveProjectCopyToFile,
  saveProjectToFile,
  setProjectURLParam,
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

  it("returns the workspace root for the workspace itself", async () => {
    const { workspace } = makeWorkspace([]);
    await expect(resolveProjectSource(workspace, workspace)).resolves.toBe("/");
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

describe("openProject", () => {
  beforeEach(() => {
    projectStore.getState().clear();
  });

  it("opens the project with its source and a fresh instance ID", () => {
    const prevInstanceId = projectStore.getState().instanceId;
    openProject({ project: { name: "Loaded" }, source: "/p.tm4" });
    const projectState = projectStore.getState();
    expect(projectState.name).toBe("Loaded");
    expect(projectState.source).toBe("/p.tm4");
    expect(projectState.instanceId).not.toBe(prevInstanceId);
  });

  it("opens the project with its save target and without unsaved changes", () => {
    const { saveTarget } = makeSaveTarget(makeWritable());
    openProject({ project: { name: "Loaded" }, source: "/p.tm4", saveTarget });
    const projectState = projectStore.getState();
    expect(projectState.saveTarget).toBe(saveTarget);
    expect(hasUnsavedChanges(projectState)).toBe(false);
  });

  it("drops what is not part of the project", () => {
    openProject({
      project: { name: "Loaded", unknown: 1 } as RawProject,
      source: null,
    });
    expect(projectStore.getState()).not.toHaveProperty("unknown");
  });
});

describe("hasUnsavedChanges", () => {
  beforeEach(() => {
    openProject({ project: { name: "Loaded" }, source: "/p.tm4" });
  });

  it("returns false right after opening a project", () => {
    expect(hasUnsavedChanges(projectStore.getState())).toBe(false);
  });

  it("returns true once a part was changed", () => {
    projectStore.getState().setName("Renamed");
    expect(hasUnsavedChanges(projectStore.getState())).toBe(true);
  });

  it("returns true once a part was replaced by an equal copy", () => {
    const projectState = projectStore.getState();
    projectStore.setState({ layers: [...projectState.layers] });
    expect(hasUnsavedChanges(projectStore.getState())).toBe(true);
  });

  it("ignores where the project was loaded from", () => {
    projectStore.setState({ source: "https://example.com/p.tm4" });
    expect(hasUnsavedChanges(projectStore.getState())).toBe(false);
  });
});

/**
 * Creates a writable file stream whose methods resolve, recording their calls
 */
function makeWritable() {
  return {
    write: vi.fn<(data: Blob) => Promise<void>>(() => Promise.resolve()),
    close: vi.fn(() => Promise.resolve()),
    abort: vi.fn(() => Promise.resolve()),
  };
}

/**
 * Creates a project file that grants write permission as given and opens the
 * given stream, with a project provider that writes the project's name as the
 * file
 */
function makeSaveTarget(
  writable: ReturnType<typeof makeWritable>,
  permission: PermissionState = "granted",
) {
  const toFile = vi.fn((project: Project) =>
    Promise.resolve(new File([project.name], "project.tm4")),
  );
  const requestPermission = vi.fn(() => Promise.resolve(permission));
  const saveTarget = {
    handle: {
      kind: "file",
      name: "project.tm4",
      requestPermission,
      createWritable: () => Promise.resolve(writable),
    } as unknown as FileSystemFileHandle,
    projectProvider: { name: "Test", toFile } as ProjectProvider,
  };
  return { toFile, requestPermission, saveTarget };
}

describe("saveProjectToFile, to the same file", () => {
  beforeEach(() => {
    projectStore.getState().clear();
  });

  it("requests write permission before anything else", async () => {
    const writable = makeWritable();
    const { toFile, requestPermission, saveTarget } = makeSaveTarget(writable);
    openProject({ project: { name: "Project" }, source: "/p.tm4", saveTarget });
    const saved = saveProjectToFile(saveTarget, "/p.tm4");
    expect(requestPermission).toHaveBeenCalledWith({ mode: "readwrite" });
    expect(toFile).not.toHaveBeenCalled();
    await saved;
  });

  it("rejects and writes nothing without write permission", async () => {
    const writable = makeWritable();
    const { toFile, saveTarget } = makeSaveTarget(writable, "denied");
    openProject({ project: { name: "Project" }, source: "/p.tm4", saveTarget });
    projectStore.getState().setName("Renamed");
    await expect(saveProjectToFile(saveTarget, "/p.tm4")).rejects.toThrow(
      "Permission to write 'project.tm4' was denied",
    );
    expect(toFile).not.toHaveBeenCalled();
    expect(hasUnsavedChanges(projectStore.getState())).toBe(true);
  });

  it("writes the project with its project provider and marks it saved", async () => {
    const writable = makeWritable();
    const { toFile, saveTarget } = makeSaveTarget(writable);
    openProject({ project: { name: "Project" }, source: "/p.tm4", saveTarget });
    projectStore.getState().setName("Renamed");
    await saveProjectToFile(saveTarget, "/p.tm4");
    expect(toFile.mock.calls[0]?.[0].name).toBe("Renamed");
    expect(toFile.mock.calls[0]?.[0]).not.toHaveProperty("source");
    expect(writable.write).toHaveBeenCalledWith(
      await toFile.mock.results[0]?.value,
    );
    expect(writable.close).toHaveBeenCalled();
    expect(hasUnsavedChanges(projectStore.getState())).toBe(false);
  });

  it("leaves the sources and their objects as they are", async () => {
    const { toFile, saveTarget } = makeSaveTarget(makeWritable());
    const sources = ["./a.tif", "/data/b.tif", "../c.tif", "https://e.org/d"];
    openProject({
      project: makeProjectWithSources(sources),
      source: "/data/p.tm4",
      saveTarget,
    });
    const { images } = projectStore.getState();
    await saveProjectToFile(saveTarget, "/data/p.tm4");
    expect(
      toFile.mock.calls[0]?.[0].images.map((image) => image.dataSource.source),
    ).toEqual(sources);
    expect(projectStore.getState().images).toBe(images);
  });

  it("aborts the file and keeps the changes when writing fails", async () => {
    const writable = makeWritable();
    writable.write.mockRejectedValue(new Error("disk full"));
    writable.abort.mockRejectedValue(new Error("abort failed"));
    const { saveTarget } = makeSaveTarget(writable);
    openProject({ project: { name: "Project" }, source: "/p.tm4", saveTarget });
    projectStore.getState().setName("Renamed");
    await expect(saveProjectToFile(saveTarget, "/p.tm4")).rejects.toThrow(
      "disk full",
    );
    expect(writable.abort).toHaveBeenCalled();
    expect(hasUnsavedChanges(projectStore.getState())).toBe(true);
  });

  it("does not mark a project opened during the write as saved", async () => {
    const writable = makeWritable();
    const { saveTarget } = makeSaveTarget(writable);
    openProject({ project: { name: "Project" }, source: "/p.tm4", saveTarget });
    writable.write.mockImplementation(() => {
      openProject({ project: { name: "Other" }, source: null });
      projectStore.getState().setName("Other renamed");
      return Promise.resolve();
    });
    await saveProjectToFile(saveTarget, "/p.tm4");
    expect(hasUnsavedChanges(projectStore.getState())).toBe(true);
  });

  it("does not mark the project saved once it belongs to another file", async () => {
    const writable = makeWritable();
    const { saveTarget } = makeSaveTarget(writable);
    openProject({ project: { name: "Project" }, source: "/p.tm4", saveTarget });
    projectStore.getState().setName("Renamed");
    writable.write.mockImplementation(() => {
      projectStore.setState({ saveTarget: { ...saveTarget } });
      return Promise.resolve();
    });
    await saveProjectToFile(saveTarget, "/p.tm4");
    expect(hasUnsavedChanges(projectStore.getState())).toBe(true);
  });
});

describe("saveProjectCopyToFile", () => {
  it("writes the project with its sources unchanged and leaves it open as it is", async () => {
    const { saveTarget: openTarget } = makeSaveTarget(makeWritable());
    openProject({
      project: makeProjectWithSources(["a.tif"]),
      source: "/data/p.tm4",
      saveTarget: openTarget,
    });
    projectStore.getState().setName("Renamed");
    const { toFile, saveTarget } = makeSaveTarget(makeWritable());
    await saveProjectCopyToFile(saveTarget);
    expect(toFile.mock.calls[0]?.[0].images[0]?.dataSource.source).toBe(
      "a.tif",
    );
    const projectState = projectStore.getState();
    expect(projectState.source).toBe("/data/p.tm4");
    expect(projectState.saveTarget).toBe(openTarget);
    expect(hasUnsavedChanges(projectState)).toBe(true);
  });
});

/**
 * Creates a raw project with one image per source
 */
function makeProjectWithSources(sources: string[]): RawProject {
  return {
    name: "Project",
    images: sources.map((source, i) => ({
      id: `image${i}`,
      name: `Image ${i}`,
      layer: "layer",
      dataSource: { type: "tiff", source },
    })),
  };
}

describe("saveProjectToFile, to a new file", () => {
  const workspace = makeWorkspace([]).workspace;

  beforeEach(() => {
    appStore.setState({ projectProviders: new Map() });
    appStore
      .getState()
      .registerProjectProvider("tm4", { name: "TM4", urlParam: "project" });
    appStore.getState().setWorkspace(workspace);
    window.history.replaceState({}, "", "/");
  });

  /**
   * Opens a project with one image per source, saves it as a new file and
   * returns the sources the open project switched to
   */
  async function saveSourcesAs(
    sources: string[],
    fromProjectSource: string | null,
    toProjectSource: string,
  ) {
    const { saveTarget } = makeSaveTarget(makeWritable());
    openProject({
      project: makeProjectWithSources(sources),
      source: fromProjectSource,
    });
    await saveProjectToFile(saveTarget, toProjectSource);
    return projectStore
      .getState()
      .images.map((image) => image.dataSource.source);
  }

  it("makes workspace paths relative to the new file", async () => {
    await expect(
      saveSourcesAs(["/data/a.tif", "/shared/b.tif"], null, "/data/study.tm4"),
    ).resolves.toEqual(["a.tif", "../shared/b.tif"]);
  });

  it("rebases paths relative to the old file", async () => {
    await expect(
      saveSourcesAs(
        ["images/a.tif"],
        "/old/project.tm4",
        "/new/sub/project.tm4",
      ),
    ).resolves.toEqual(["../../old/images/a.tif"]);
  });

  it("keeps URLs and app-relative paths", async () => {
    await expect(
      saveSourcesAs(
        ["https://example.com/a.tif", "//data/b.tif"],
        null,
        "/study.tm4",
      ),
    ).resolves.toEqual(["https://example.com/a.tif", "//data/b.tif"]);
  });

  it("makes paths relative to a URL project absolute", async () => {
    await expect(
      saveSourcesAs(
        ["images/a.tif"],
        "https://example.com/study/project.tm4",
        "/study.tm4",
      ),
    ).resolves.toEqual(["https://example.com/study/images/a.tif"]);
  });

  it("keeps the objects whose source does not change", async () => {
    const { saveTarget } = makeSaveTarget(makeWritable());
    openProject({
      project: makeProjectWithSources(["https://example.com/a.tif"]),
      source: null,
    });
    const image = projectStore.getState().images[0];
    await saveProjectToFile(saveTarget, "/study.tm4");
    expect(projectStore.getState().images[0]).toBe(image);
  });

  it("writes the rebased project and switches to the new file", async () => {
    const { toFile, saveTarget } = makeSaveTarget(makeWritable());
    openProject({
      project: makeProjectWithSources(["/data/a.tif"]),
      source: null,
    });
    await saveProjectToFile(saveTarget, "/data/study.tm4");
    const projectState = projectStore.getState();
    expect(toFile.mock.calls[0]?.[0].images[0]?.dataSource.source).toBe(
      "a.tif",
    );
    expect(projectState.source).toBe("/data/study.tm4");
    expect(projectState.saveTarget).toBe(saveTarget);
    expect(hasUnsavedChanges(projectState)).toBe(false);
  });

  it("keeps changes made during the write, unsaved", async () => {
    const writable = makeWritable();
    const { saveTarget } = makeSaveTarget(writable);
    openProject({
      project: makeProjectWithSources(["/data/a.tif"]),
      source: null,
    });
    writable.write.mockImplementation(() => {
      projectStore.getState().setName("Edited");
      return Promise.resolve();
    });
    await saveProjectToFile(saveTarget, "/data/study.tm4");
    const projectState = projectStore.getState();
    expect(projectState.name).toBe("Edited");
    expect(projectState.images[0]?.dataSource.source).toBe("a.tif");
    expect(hasUnsavedChanges(projectState)).toBe(true);
  });

  it("removes the project URLs from the address bar", async () => {
    const { saveTarget } = makeSaveTarget(makeWritable());
    openProject({ project: { name: "Project" }, source: null });
    window.history.replaceState({}, "", "/?project=https://example.com/p.tm4");
    await saveProjectToFile(saveTarget, "/study.tm4");
    expect(window.location.search).toBe("");
  });

  it("does not switch over once another workspace is open", async () => {
    const writable = makeWritable();
    const { saveTarget } = makeSaveTarget(writable);
    openProject({ project: { name: "Project" }, source: null });
    writable.write.mockImplementation(() => {
      appStore.getState().setWorkspace(makeWorkspace([]).workspace);
      return Promise.resolve();
    });
    await saveProjectToFile(saveTarget, "/study.tm4");
    expect(projectStore.getState().saveTarget).toBeNull();
  });

  it("keeps the URL of a project opened during the write", async () => {
    const writable = makeWritable();
    const { saveTarget } = makeSaveTarget(writable);
    openProject({ project: { name: "Project" }, source: null });
    writable.write.mockImplementation(() => {
      openProject({ project: { name: "Other" }, source: null });
      window.history.replaceState(
        {},
        "",
        "/?project=https://example.com/o.tm4",
      );
      return Promise.resolve();
    });
    await saveProjectToFile(saveTarget, "/study.tm4");
    expect(window.location.search).not.toBe("");
    expect(projectStore.getState().saveTarget).toBeNull();
  });
});

describe("loadProjectFromURL", () => {
  it("resolves the URL against the document base URL", async () => {
    const { load, projectProvider } = makeProjectProvider();
    const { source } = await loadProjectFromURL(
      projectProvider,
      "projects/p.tm4",
    );
    const expectedUrl = new URL("projects/p.tm4", document.baseURI).href;
    expect(load).toHaveBeenCalledWith(expectedUrl, { signal: undefined });
    expect(source).toBe(expectedUrl);
  });

  it("returns the source the project provider reports", async () => {
    const redirectedUrl = "https://cdn.example/p.tm4";
    const { projectProvider } = makeProjectProvider(redirectedUrl);
    await expect(
      loadProjectFromURL(projectProvider, "https://data.example/p.tm4"),
    ).resolves.toEqual({ project: { name: "Loaded" }, source: redirectedUrl });
  });

  it("rejects for a project provider that cannot load projects", async () => {
    await expect(
      loadProjectFromURL({ name: "Test" }, "https://data.example/p.tm4"),
    ).rejects.toThrow("Project provider 'Test' cannot load projects");
  });
});

describe("loadLocalProject", () => {
  const objectUrl = "blob:http://localhost/0f8e";
  const revokeObjectURL = vi.fn();
  const projectDirectory = {
    kind: "directory",
    name: "data.zarr",
  } as FileSystemDirectoryHandle;

  beforeEach(() => {
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

  it("loads a file without a source, and revokes the object URL", async () => {
    const { load, projectProvider } = makeProjectProvider();
    await expect(
      loadLocalProject(projectProvider, new File([], "p.tm4"), null),
    ).resolves.toEqual({
      project: { name: "Loaded" },
      source: null,
      saveTarget: null,
    });
    expect(load).toHaveBeenCalledWith(objectUrl, { signal: undefined });
    expect(revokeObjectURL).toHaveBeenCalledWith(objectUrl);
  });

  it("revokes the object URL if loading fails", async () => {
    const projectProvider: ProjectProvider = {
      name: "Test",
      load: () => Promise.reject(new Error("Broken")),
    };
    await expect(
      loadLocalProject(projectProvider, new File([], "p.tm4"), null),
    ).rejects.toThrow("Broken");
    expect(revokeObjectURL).toHaveBeenCalledWith(objectUrl);
  });

  it("loads a file in the workspace through its workspace-relative path", async () => {
    const { workspace } = makeWorkspace(["study", "project.tm4"]);
    const { load, projectProvider } = makeProjectProvider();
    const { source, saveTarget } = await loadLocalProject(
      projectProvider,
      projectFile,
      workspace,
    );
    expect(load).toHaveBeenCalledWith("/study/project.tm4", {
      signal: undefined,
      workspace,
    });
    expect(source).toBe("/study/project.tm4");
    expect(saveTarget).toEqual({ handle: projectFile, projectProvider });
  });

  it("loads a file outside the workspace like an uploaded file", async () => {
    const { workspace } = makeWorkspace(null);
    const outsideFile = {
      ...projectFile,
      getFile: () => Promise.resolve(new File([], "project.tm4")),
    } as FileSystemFileHandle;
    const { load, projectProvider } = makeProjectProvider();
    const { source, saveTarget } = await loadLocalProject(
      projectProvider,
      outsideFile,
      workspace,
    );
    expect(load).toHaveBeenCalledWith(objectUrl, { signal: undefined });
    expect(source).toBeNull();
    expect(saveTarget).toBeNull();
  });

  it("loads a directory in the workspace through its workspace-relative path", async () => {
    const { workspace } = makeWorkspace(["study", "data.zarr"]);
    const { load, projectProvider } = makeProjectProvider();
    const { source, saveTarget } = await loadLocalProject(
      projectProvider,
      projectDirectory,
      workspace,
    );
    expect(load).toHaveBeenCalledWith("/study/data.zarr", {
      signal: undefined,
      workspace,
    });
    expect(source).toBe("/study/data.zarr");
    expect(saveTarget).toBeNull();
  });

  it("loads the workspace itself through the workspace root", async () => {
    const { workspace } = makeWorkspace([]);
    const { load, projectProvider } = makeProjectProvider();
    const { source } = await loadLocalProject(
      projectProvider,
      workspace,
      workspace,
    );
    expect(load).toHaveBeenCalledWith("/", { signal: undefined, workspace });
    expect(source).toBe("/");
  });

  it("rejects a directory outside the workspace", async () => {
    const { workspace } = makeWorkspace(null);
    const { load, projectProvider } = makeProjectProvider();
    await expect(
      loadLocalProject(projectProvider, projectDirectory, workspace),
    ).rejects.toThrow(/not in the workspace/);
    expect(load).not.toHaveBeenCalled();
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

  it("names the file after where the open project was opened from", async () => {
    const names: (string | undefined)[] = [];
    const projectProvider: ProjectProvider = {
      name: "Test",
      toFile: (_project, options) => {
        names.push(options?.name);
        return Promise.resolve(new File([], "p.tm4"));
      },
    };
    projectStore.setState({ source: "https://data.example/my_experiment.tm4" });
    await downloadProject(projectProvider);
    projectStore.setState({ source: "/" });
    await downloadProject(projectProvider);
    projectStore.setState({ source: null });
    await downloadProject(projectProvider);
    await downloadProject(projectProvider, {
      project: projectStore.getState(),
    });
    expect(names).toEqual(["my_experiment", undefined, undefined, undefined]);
  });

  it("rejects for a project provider that cannot write projects", async () => {
    await expect(downloadProject({ name: "Test" })).rejects.toThrow(
      "Project provider 'Test' cannot write projects",
    );
  });
});

/** A project to merge, with relative, absolute and missing sources */
function makeMergedProject(): RawProject {
  return {
    name: "Merged",
    layers: [{ id: "merged-layer", name: "Merged" }],
    images: [
      {
        id: "merged-image",
        name: "Image",
        layer: "merged-layer",
        dataSource: { type: "tiff", source: "images/a.tif" },
      },
    ],
    points: [
      {
        id: "merged-points",
        name: "Points",
        layer: "merged-layer",
        dataSource: { type: "table", table: "merged-table" },
      },
    ],
    tables: [
      {
        id: "merged-table",
        name: "Table",
        dataSource: { type: "csv", source: "https://other.example/t.csv" },
      },
    ],
    colorMaps: [{ id: "shared-map", name: "Colors", values: {} }],
    viewerBackgroundColor: { r: 255, g: 255, b: 255 },
  };
}

describe("mergeProject", () => {
  const mergedProjectUrl = "https://data.example/merged/project.tm4";

  beforeEach(() => {
    projectStore.getState().clear();
    appStore.setState({
      projectOpen: false,
      expandedImageIds: [],
      expandedPointsIds: [],
      expandedTableIds: [],
    });
  });

  it("appends the merged objects, keeping the open project's source and instance", () => {
    projectStore.setState({ source: "https://data.example/open/project.tm4" });
    const { instanceId, layers } = projectStore.getState();
    mergeProject({ project: makeMergedProject(), source: mergedProjectUrl });
    const projectState = projectStore.getState();
    expect(projectState.source).toBe("https://data.example/open/project.tm4");
    expect(projectState.instanceId).toBe(instanceId);
    expect(projectState.layers.map((layer) => layer.id)).toEqual([
      ...layers.map((layer) => layer.id),
      "merged-layer",
    ]);
    expect(projectState.images.map((image) => image.id)).toEqual([
      "merged-image",
    ]);
    expect(projectState.points.map((points) => points.id)).toEqual([
      "merged-points",
    ]);
    expect(projectState.tables.map((table) => table.id)).toEqual([
      "merged-table",
    ]);
    expect(projectState.colorMaps.map((map) => map.id)).toEqual(["shared-map"]);
  });

  it("normalizes the merged data sources against where the project was loaded from", () => {
    mergeProject({ project: makeMergedProject(), source: mergedProjectUrl });
    const projectState = projectStore.getState();
    expect(projectState.images[0]?.dataSource.source).toBe(
      "https://data.example/merged/images/a.tif",
    );
    expect(projectState.tables[0]?.dataSource.source).toBe(
      "https://other.example/t.csv",
    );
    expect(projectState.points[0]?.dataSource.source).toBeUndefined();
  });

  it("leaves the given project unchanged", () => {
    const rawProject = makeMergedProject();
    mergeProject({ project: rawProject, source: mergedProjectUrl });
    expect(rawProject).toEqual(makeMergedProject());
  });

  it("adopts the name only for an open project without data, and ignores the viewer options", () => {
    const { viewerBackgroundColor } = projectStore.getState();
    mergeProject({ project: makeMergedProject(), source: mergedProjectUrl });
    expect(projectStore.getState().name).toBe("Merged");
    expect(projectStore.getState().viewerBackgroundColor).toEqual(
      viewerBackgroundColor,
    );
    projectStore.getState().setName("Open");
    mergeProject({
      project: {
        name: "Second",
        tables: [
          { id: "second-table", name: "Table", dataSource: { type: "csv" } },
        ],
      },
      source: mergedProjectUrl,
    });
    expect(projectStore.getState().name).toBe("Open");
  });

  it("expands the merged objects and marks the project open", () => {
    mergeProject({ project: makeMergedProject(), source: mergedProjectUrl });
    const appState = appStore.getState();
    expect(appState.expandedImageIds).toEqual(["merged-image"]);
    expect(appState.expandedPointsIds).toEqual(["merged-points"]);
    expect(appState.expandedTableIds).toEqual(["merged-table"]);
    expect(appState.projectOpen).toBe(true);
  });

  it("rejects a taken ID, leaving the open project unchanged", () => {
    mergeProject({ project: makeMergedProject(), source: mergedProjectUrl });
    const projectState = projectStore.getState();
    const rawProject = makeMergedProject();
    rawProject.layers = [{ id: "fresh-layer", name: "Fresh" }];
    expect(() => {
      mergeProject({ project: rawProject, source: mergedProjectUrl });
    }).toThrow("ID 'merged-image' is already taken in the open project");
    expect(projectStore.getState()).toBe(projectState);
  });

  it("normalizes the merged data sources against the workspace root", () => {
    const { workspace } = makeWorkspace([]);
    appStore.setState({ workspace });
    try {
      mergeProject({ project: makeMergedProject(), source: "/" });
      expect(projectStore.getState().images[0]?.dataSource.source).toBe(
        "/images/a.tif",
      );
    } finally {
      appStore.setState({ workspace: null });
    }
  });

  it("allows a map ID that is taken in another list of maps", () => {
    mergeProject({ project: makeMergedProject(), source: mergedProjectUrl });
    mergeProject({
      project: {
        name: "Sizes",
        sizeMaps: [{ id: "shared-map", name: "Sizes", values: {} }],
      },
      source: mergedProjectUrl,
    });
    expect(projectStore.getState().sizeMaps.map((map) => map.id)).toEqual([
      "shared-map",
    ]);
  });
});

/** Creates a project provider that loads projects with the given extension */
function makeLoadingProjectProvider(
  name: string,
  urlParam: string,
  extension: string,
): ProjectProvider {
  return {
    name,
    urlParam,
    fileTypes: { "application/octet-stream": [extension] },
    supports: (normalizedSource) =>
      Promise.resolve(normalizedSource.endsWith(extension)),
    load: (normalizedSource) =>
      Promise.resolve({ project: { name }, source: normalizedSource }),
  };
}

describe("project providers", () => {
  const tm4Provider = makeLoadingProjectProvider("TM4", "project", ".tm4");
  const zarrProvider = makeLoadingProjectProvider("Zarr", "zarr", ".zarr");

  beforeEach(() => {
    appStore.setState({ projectProviders: new Map() });
    appStore.getState().registerProjectProvider("tm4", tm4Provider);
    appStore.getState().registerProjectProvider("zarr", zarrProvider);
    window.history.replaceState({}, "", "/?other=1");
  });

  describe("findProjectProviders", () => {
    it("returns the project providers that support the source", async () => {
      await expect(
        findProjectProviders("https://data.example/a.zarr", null),
      ).resolves.toEqual([zarrProvider]);
    });

    it("returns all supporting project providers in registration order, however fast", async () => {
      const slowProvider: ProjectProvider = {
        ...tm4Provider,
        supports: () =>
          new Promise((resolve) => setTimeout(() => resolve(true), 10)),
      };
      appStore.setState({ projectProviders: new Map() });
      appStore.getState().registerProjectProvider("slow", slowProvider);
      appStore.getState().registerProjectProvider("zarr", zarrProvider);
      await expect(
        findProjectProviders("https://data.example/a.zarr", null),
      ).resolves.toEqual([slowProvider, zarrProvider]);
    });

    it("skips project providers that cannot load, or whose check rejects", async () => {
      appStore.setState({ projectProviders: new Map() });
      appStore.getState().registerProjectProvider("writer", {
        name: "Writer",
        supports: () => Promise.resolve(true),
        toFile: () => Promise.resolve(new File([], "p")),
      });
      appStore.getState().registerProjectProvider("broken", {
        ...tm4Provider,
        supports: () => Promise.reject(new Error("Broken")),
      });
      appStore.getState().registerProjectProvider("zarr", zarrProvider);
      await expect(
        findProjectProviders("https://data.example/a.zarr", null),
      ).resolves.toEqual([zarrProvider]);
    });

    it("returns no project provider if none supports the source", async () => {
      await expect(
        findProjectProviders("https://data.example/a.csv", null),
      ).resolves.toEqual([]);
    });

    it("rejects if cancelled", async () => {
      const abortController = new AbortController();
      abortController.abort();
      await expect(
        findProjectProviders("https://data.example/a.zarr", null, {
          signal: abortController.signal,
        }),
      ).rejects.toThrow();
    });
  });

  describe("findLocalProjectProviders", () => {
    it("matches files by their file extension, regardless of case", () => {
      expect(findLocalProjectProviders(new File([], "Project.TM4"))).toEqual([
        tm4Provider,
      ]);
      expect(findLocalProjectProviders(new File([], "a.csv"))).toEqual([]);
    });

    it("returns all matching project providers in registration order", () => {
      const otherProvider: ProjectProvider = { ...tm4Provider, name: "Other" };
      appStore.getState().registerProjectProvider("other", otherProvider);
      expect(findLocalProjectProviders(new File([], "p.tm4"))).toEqual([
        tm4Provider,
        otherProvider,
      ]);
    });

    it("skips project providers that cannot load", () => {
      appStore.setState({ projectProviders: new Map() });
      appStore.getState().registerProjectProvider("writer", {
        name: "Writer",
        fileTypes: { "application/json": [".tm4"] },
        toFile: () => Promise.resolve(new File([], "p")),
      });
      expect(findLocalProjectProviders(new File([], "project.tm4"))).toEqual(
        [],
      );
    });
  });

  describe("getProjectFileExtensions", () => {
    it("returns the file extensions of the project providers that can load", () => {
      appStore.getState().registerProjectProvider("other", {
        ...tm4Provider,
        fileTypes: { "application/json": [".tm4", ".json"] },
      });
      appStore.getState().registerProjectProvider("writer", {
        name: "Writer",
        fileTypes: { "application/xml": [".xml"] },
        toFile: () => Promise.resolve(new File([], "p")),
      });
      expect(getProjectFileExtensions()).toEqual([".tm4", ".zarr", ".json"]);
    });
  });

  describe("getProjectFileTypes", () => {
    it("offers all project files first, then those of each project provider", () => {
      appStore.getState().registerProjectProvider("json", {
        ...tm4Provider,
        name: "JSON",
        fileTypes: { "application/json": [".tm4", ".json"] },
      });
      appStore.getState().registerProjectProvider("writer", {
        name: "Writer",
        fileTypes: { "application/xml": [".xml"] },
        toFile: () => Promise.resolve(new File([], "p")),
      });
      expect(getProjectFileTypes()).toEqual([
        {
          description: "Project",
          accept: {
            "application/octet-stream": [".tm4", ".zarr"],
            "application/json": [".tm4", ".json"],
          },
        },
        {
          description: "TM4 project",
          accept: { "application/octet-stream": [".tm4"] },
        },
        {
          description: "Zarr project",
          accept: { "application/octet-stream": [".zarr"] },
        },
        {
          description: "JSON project",
          accept: { "application/json": [".tm4", ".json"] },
        },
      ]);
    });
  });

  describe("readProjectURLParams", () => {
    it("returns the named projects in the order of the parameters", () => {
      expect(
        readProjectURLParams("?zarr=a.zarr&other=1&project=b.tm4&zarr=c.zarr"),
      ).toEqual([
        { projectProvider: zarrProvider, projectUrl: "a.zarr" },
        { projectProvider: tm4Provider, projectUrl: "b.tm4" },
        { projectProvider: zarrProvider, projectUrl: "c.zarr" },
      ]);
    });

    it("skips empty values and project providers that cannot load", () => {
      appStore.getState().registerProjectProvider("writer", {
        name: "Writer",
        urlParam: "writer",
        toFile: () => Promise.resolve(new File([], "p")),
      });
      expect(readProjectURLParams("?project=&writer=w.tm4")).toEqual([]);
    });
  });

  describe("URL parameters", () => {
    it("replaces the parameters of every project provider when opening", () => {
      window.history.replaceState({}, "", "/?project=a.tm4&other=1&zarr=b");
      setProjectURLParam(zarrProvider, "c.zarr");
      expect(window.location.search).toBe("?other=1&zarr=c.zarr");
    });

    it("only removes the parameters for a project provider without one", () => {
      window.history.replaceState({}, "", "/?project=a.tm4&other=1");
      setProjectURLParam({ ...zarrProvider, urlParam: undefined }, "b");
      expect(window.location.search).toBe("?other=1");
    });

    it("appends a parameter when importing", () => {
      window.history.replaceState({}, "", "/?project=a.tm4");
      appendProjectURLParam(zarrProvider, "b.zarr");
      appendProjectURLParam(zarrProvider, "c.zarr");
      expect(window.location.search).toBe(
        "?project=a.tm4&zarr=b.zarr&zarr=c.zarr",
      );
    });

    it("leaves the URL alone when importing with a project provider without a parameter", () => {
      window.history.replaceState({}, "", "/?project=a.tm4");
      appendProjectURLParam({ ...zarrProvider, urlParam: undefined }, "b");
      expect(window.location.search).toBe("?project=a.tm4");
    });

    it("removes the parameters of every project provider", () => {
      window.history.replaceState({}, "", "/?project=a.tm4&other=1&zarr=b");
      clearProjectURLParams();
      expect(window.location.search).toBe("?other=1");
    });
  });
});
