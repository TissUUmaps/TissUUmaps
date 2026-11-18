import { afterEach, describe, expect, it, vi } from "vitest";

import { JSONUtils, type RawProject, createProject } from "@tissuumaps/core";

import { TissUUmapsProjectProvider } from "./TissUUmapsProjectProvider";

const projectUrl = "https://data.example/projects/p1/project.tm4";
const rawProject: RawProject = {
  name: "My project",
  images: [
    {
      id: "img",
      name: "Image",
      layer: "layer",
      dataSource: { type: "tiff", source: "image.tif" },
    },
  ],
  osOptions: {
    viewerOptions: { maxZoomPixelRatio: Infinity },
    viewerAnimationStartOptions: {},
    viewerAnimationFinishOptions: {},
  },
};

/**
 * Stubs `fetch` to answer every request with a response
 *
 * `Response.url` cannot be set through the constructor, so redirects are
 * simulated by overriding it.
 */
function stubFetch(body: string | null, init?: ResponseInit, url = "") {
  const response = new Response(body, init);
  Object.defineProperty(response, "url", { value: url });
  const fetchMock = vi.fn(() => Promise.resolve(response));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** Creates a workspace that holds a project file in `/proj/project.tm4` */
function makeWorkspace(projectJSON: string) {
  const projectFile = {
    kind: "file",
    name: "project.tm4",
    getFile: () => Promise.resolve(new File([projectJSON], "project.tm4")),
  };
  const proj = {
    kind: "directory",
    name: "proj",
    getFileHandle: () => Promise.resolve(projectFile),
  };
  return {
    kind: "directory",
    name: "",
    getDirectoryHandle: () => Promise.resolve(proj),
  } as unknown as FileSystemDirectoryHandle;
}

describe("TissUUmapsProjectProvider", () => {
  const provider = new TissUUmapsProjectProvider();

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("supports", () => {
    it("supports project files, ignoring the case of the extension", async () => {
      expect(await provider.supports(projectUrl)).toBe(true);
      expect(await provider.supports("/proj/PROJECT.TM4")).toBe(true);
    });

    it("does not support other files", async () => {
      expect(await provider.supports("/proj/points.csv")).toBe(false);
      expect(await provider.supports("/proj/project.tm4.json")).toBe(false);
    });
  });

  describe("load", () => {
    it("loads a project from a URL, decoding non-finite numbers", async () => {
      const fetchMock = stubFetch(
        JSONUtils.stringify(rawProject),
        {},
        projectUrl,
      );
      const { project, source } = await provider.load(projectUrl);
      expect(fetchMock).toHaveBeenCalledWith(projectUrl, {
        signal: undefined,
      });
      expect(project).toEqual(rawProject);
      expect(source).toBe(projectUrl);
    });

    it("returns the URL a redirect led to as the source", async () => {
      const redirectedUrl = "https://cdn.example/p1/project.tm4";
      stubFetch(JSONUtils.stringify(rawProject), {}, redirectedUrl);
      const { source } = await provider.load(projectUrl);
      expect(source).toBe(redirectedUrl);
    });

    it("falls back to the requested URL for responses without one", async () => {
      stubFetch(JSONUtils.stringify(rawProject));
      const { source } = await provider.load(projectUrl);
      expect(source).toBe(projectUrl);
    });

    it("throws for an unsuccessful response", async () => {
      stubFetch(null, { status: 404, statusText: "Not Found" });
      await expect(provider.load(projectUrl)).rejects.toThrow(
        `Failed to load project from ${projectUrl}: 404 Not Found`,
      );
    });

    it("loads a project from the workspace, with its path as the source", async () => {
      const fetchMock = stubFetch(null);
      const workspace = makeWorkspace(JSONUtils.stringify(rawProject));
      const { project, source } = await provider.load("/proj/project.tm4", {
        workspace,
      });
      expect(fetchMock).not.toHaveBeenCalled();
      expect(project).toEqual(rawProject);
      expect(source).toBe("/proj/project.tm4");
    });

    it("throws if aborted", async () => {
      const fetchMock = stubFetch(JSONUtils.stringify(rawProject));
      await expect(
        provider.load(projectUrl, { signal: AbortSignal.abort() }),
      ).rejects.toThrow();
      expect(fetchMock).not.toHaveBeenCalled();
    });
  });

  describe("toFile", () => {
    it("writes the project as JSON, encoding non-finite numbers", async () => {
      const project = createProject(rawProject);
      const file = await provider.toFile(project);
      expect(file.type).toBe("application/json");
      expect(JSONUtils.parse(await file.text())).toEqual(project);
    });

    it("names the file after the sanitized project name", async () => {
      const project = createProject({ name: "  My  project: v2!  " });
      expect((await provider.toFile(project)).name).toBe("My-project-v2.tm4");
    });

    it("falls back to Untitled", async () => {
      const project = createProject({ name: " ?! " });
      expect((await provider.toFile(project)).name).toBe("Untitled.tm4");
    });
  });
});
