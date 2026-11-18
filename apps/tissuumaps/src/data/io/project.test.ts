import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Project, ProjectProvider, RawProject } from "@tissuumaps/core";

import { appStore } from "@/stores/app";
import { projectStore } from "@/stores/project";

import {
  clearProjectURLParams,
  downloadProject,
  importProjectFromURL,
  mergeProject,
  openProjectFromFile,
  openProjectFromFileHandle,
  openProjectFromURL,
  pickProjectProvider,
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
    mergeProject(makeMergedProject(), mergedProjectUrl);
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
    mergeProject(makeMergedProject(), mergedProjectUrl);
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
    mergeProject(rawProject, mergedProjectUrl);
    expect(rawProject).toEqual(makeMergedProject());
  });

  it("adopts the name only for an open project without data, and ignores the viewer options", () => {
    const { viewerBackgroundColor } = projectStore.getState();
    mergeProject(makeMergedProject(), mergedProjectUrl);
    expect(projectStore.getState().name).toBe("Merged");
    expect(projectStore.getState().viewerBackgroundColor).toEqual(
      viewerBackgroundColor,
    );
    projectStore.getState().setName("Open");
    mergeProject(
      {
        name: "Second",
        tables: [
          { id: "second-table", name: "Table", dataSource: { type: "csv" } },
        ],
      },
      mergedProjectUrl,
    );
    expect(projectStore.getState().name).toBe("Open");
  });

  it("expands the merged objects and marks the project open", () => {
    mergeProject(makeMergedProject(), mergedProjectUrl);
    const appState = appStore.getState();
    expect(appState.expandedImageIds).toEqual(["merged-image"]);
    expect(appState.expandedPointsIds).toEqual(["merged-points"]);
    expect(appState.expandedTableIds).toEqual(["merged-table"]);
    expect(appState.projectOpen).toBe(true);
  });

  it("rejects a taken ID, leaving the open project unchanged", () => {
    mergeProject(makeMergedProject(), mergedProjectUrl);
    const projectState = projectStore.getState();
    const rawProject = makeMergedProject();
    rawProject.layers = [{ id: "fresh-layer", name: "Fresh" }];
    expect(() => {
      mergeProject(rawProject, mergedProjectUrl);
    }).toThrow("ID merged-image is taken");
    expect(projectStore.getState()).toBe(projectState);
  });

  it("allows a map ID that is taken in another list of maps", () => {
    mergeProject(makeMergedProject(), mergedProjectUrl);
    mergeProject(
      {
        name: "Sizes",
        sizeMaps: [{ id: "shared-map", name: "Sizes", values: {} }],
      },
      mergedProjectUrl,
    );
    expect(projectStore.getState().sizeMaps.map((map) => map.id)).toEqual([
      "shared-map",
    ]);
  });
});

describe("importProjectFromURL", () => {
  beforeEach(() => {
    projectStore.getState().clear();
  });

  it("merges the project, resolving its sources against the reported source", async () => {
    const projectProvider: ProjectProvider = {
      name: "Test",
      load: () =>
        Promise.resolve({
          project: makeMergedProject(),
          source: "https://cdn.example/merged/project.tm4",
        }),
    };
    const { source } = projectStore.getState();
    await importProjectFromURL(projectProvider, "https://data.example/p.tm4");
    expect(projectStore.getState().source).toBe(source);
    expect(projectStore.getState().images[0]?.dataSource.source).toBe(
      "https://cdn.example/merged/images/a.tif",
    );
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

  describe("pickProjectProvider", () => {
    it("returns the first project provider that supports the source", async () => {
      await expect(
        pickProjectProvider("https://data.example/a.zarr", null),
      ).resolves.toBe(zarrProvider);
    });

    it("prefers earlier registered project providers over faster ones", async () => {
      const slowProvider: ProjectProvider = {
        ...tm4Provider,
        supports: () =>
          new Promise((resolve) => setTimeout(() => resolve(true), 10)),
      };
      appStore.setState({ projectProviders: new Map() });
      appStore.getState().registerProjectProvider("slow", slowProvider);
      appStore.getState().registerProjectProvider("zarr", zarrProvider);
      await expect(
        pickProjectProvider("https://data.example/a.zarr", null),
      ).resolves.toBe(slowProvider);
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
        pickProjectProvider("https://data.example/a.zarr", null),
      ).resolves.toBe(zarrProvider);
    });

    it("returns undefined if no project provider supports the source", async () => {
      await expect(
        pickProjectProvider("https://data.example/a.csv", null),
      ).resolves.toBeUndefined();
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

    it("removes the parameters of every project provider", () => {
      window.history.replaceState({}, "", "/?project=a.tm4&other=1&zarr=b");
      clearProjectURLParams();
      expect(window.location.search).toBe("?other=1");
    });
  });
});
