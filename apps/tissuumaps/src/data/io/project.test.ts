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
  loadLocalProject,
  loadProjectFromURL,
  mergeProject,
  openProject,
  readProjectURLParams,
  resolveProjectSource,
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
    ).rejects.toThrow("Project provider Test cannot load projects.");
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
    ).resolves.toEqual({ project: { name: "Loaded" }, source: null });
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
    const { source } = await loadLocalProject(
      projectProvider,
      projectFile,
      workspace,
    );
    expect(load).toHaveBeenCalledWith("/study/project.tm4", {
      signal: undefined,
      workspace,
    });
    expect(source).toBe("/study/project.tm4");
  });

  it("loads a file outside the workspace like an uploaded file", async () => {
    const { workspace } = makeWorkspace(null);
    const outsideFile = {
      ...projectFile,
      getFile: () => Promise.resolve(new File([], "project.tm4")),
    } as FileSystemFileHandle;
    const { load, projectProvider } = makeProjectProvider();
    const { source } = await loadLocalProject(
      projectProvider,
      outsideFile,
      workspace,
    );
    expect(load).toHaveBeenCalledWith(objectUrl, { signal: undefined });
    expect(source).toBeNull();
  });

  it("loads a directory in the workspace through its workspace-relative path", async () => {
    const { workspace } = makeWorkspace(["study", "data.zarr"]);
    const { load, projectProvider } = makeProjectProvider();
    const { source } = await loadLocalProject(
      projectProvider,
      projectDirectory,
      workspace,
    );
    expect(load).toHaveBeenCalledWith("/study/data.zarr", {
      signal: undefined,
      workspace,
    });
    expect(source).toBe("/study/data.zarr");
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
      "Project provider Test cannot write projects.",
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
    }).toThrow("ID merged-image is taken");
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
