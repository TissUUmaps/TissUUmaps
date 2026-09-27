import { afterEach, describe, expect, it, vi } from "vitest";

import { SourceUtils } from "./SourceUtils";

const baseUrl = "https://app.example/tm/index.html";
const projectUrl = "https://data.example/projects/p1/project.tm4";
const projectPath = "/proj/project.tm4";

type FakeFile = { kind: "file"; name: string; getFile?: () => Promise<File> };
type FakeDir = {
  kind: "directory";
  name: string;
  getDirectoryHandle: (name: string) => Promise<FakeDir>;
  getFileHandle: (name: string) => Promise<FakeFile>;
};

function makeFile(name: string): FakeFile {
  return { kind: "file", name };
}

function makeDir(
  name: string,
  entries: Record<string, FakeFile | FakeDir>,
  options?: { onOpen?: () => void },
): FakeDir {
  const open = (
    entryName: string,
    kind: FakeFile["kind"] | FakeDir["kind"],
  ) => {
    options?.onOpen?.();
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

/**
 * /proj/project.tm4, /proj/points.csv, /proj/s:c.tif, /proj/sub/y.csv,
 * /shared/x.csv
 */
const pointsFile = makeFile("points.csv");
const colonFile = makeFile("s:c.tif");
const yFile = makeFile("y.csv");
const xFile = makeFile("x.csv");
const subDir = makeDir("sub", { "y.csv": yFile });
const workspace = makeDir("", {
  proj: makeDir("proj", {
    "project.tm4": makeFile("project.tm4"),
    "points.csv": pointsFile,
    "s:c.tif": colonFile,
    sub: subDir,
  }),
  shared: makeDir("shared", { "x.csv": xFile }),
}) as unknown as FileSystemDirectoryHandle;

describe("SourceUtils", () => {
  describe("isWorkspacePath", () => {
    it("returns true for workspace-relative paths", () => {
      expect(SourceUtils.isWorkspacePath("/proj/points.csv")).toBe(true);
      expect(SourceUtils.isWorkspacePath("/points.csv")).toBe(true);
    });

    it("returns false for URLs", () => {
      expect(SourceUtils.isWorkspacePath("https://x.example/f.csv")).toBe(
        false,
      );
      expect(SourceUtils.isWorkspacePath("blob:https://app.example/123")).toBe(
        false,
      );
      expect(SourceUtils.isWorkspacePath("file:///proj/points.csv")).toBe(
        false,
      );
    });

    it("returns false for app-relative paths", () => {
      expect(SourceUtils.isWorkspacePath("//data/points.csv")).toBe(false);
      expect(SourceUtils.isWorkspacePath("///data/points.csv")).toBe(false);
    });
  });

  describe("makeRelativePath", () => {
    it("expresses a path within the directory", () => {
      expect(
        SourceUtils.makeRelativePath("/proj/data/points.csv", "/proj"),
      ).toBe("data/points.csv");
    });

    it("climbs out of the directory with .. segments", () => {
      expect(
        SourceUtils.makeRelativePath("/shared/points.csv", "/proj/sub"),
      ).toBe("../../shared/points.csv");
    });

    it("expresses a path from the workspace root", () => {
      expect(SourceUtils.makeRelativePath("/points.csv", "/")).toBe(
        "points.csv",
      );
    });

    it("returns . for the directory itself", () => {
      expect(SourceUtils.makeRelativePath("/proj", "/proj")).toBe(".");
    });

    it("prefixes a first segment that looks like a URL scheme", () => {
      expect(SourceUtils.makeRelativePath("/proj/s1:ch2.tif", "/proj")).toBe(
        "./s1:ch2.tif",
      );
    });
  });

  describe("makeWorkspacePath", () => {
    it("joins the segments with the workspace prefix", () => {
      expect(SourceUtils.makeWorkspacePath(["proj", "points.csv"])).toBe(
        "/proj/points.csv",
      );
      expect(SourceUtils.makeWorkspacePath(["points.csv"])).toBe("/points.csv");
    });

    it("builds a workspace-relative path", () => {
      expect(
        SourceUtils.isWorkspacePath(
          SourceUtils.makeWorkspacePath(["proj", "points.csv"]),
        ),
      ).toBe(true);
    });

    it("normalizes to itself", () => {
      const workspacePath = SourceUtils.makeWorkspacePath(["proj", "a.csv"]);
      expect(
        SourceUtils.normalizeSource(workspacePath, workspace, null, {
          baseUrl,
        }),
      ).toBe(workspacePath);
    });
  });

  describe("getParentSource", () => {
    it("returns the parent directory of a URL, keeping its query", () => {
      expect(
        SourceUtils.getParentSource("https://data.example/a/b.zarr/?key=1"),
      ).toEqual({
        parentSource: "https://data.example/a?key=1",
        name: "b.zarr",
      });
    });

    it("decodes the name but keeps the parent URL encoded", () => {
      expect(
        SourceUtils.getParentSource("https://data.example/my%20dir/a%20b"),
      ).toEqual({
        parentSource: "https://data.example/my%20dir",
        name: "a b",
      });
    });

    it("goes up to the root of a URL's path, and no further", () => {
      expect(SourceUtils.getParentSource("https://data.example/a")).toEqual({
        parentSource: "https://data.example/",
        name: "a",
      });
      expect(SourceUtils.getParentSource("https://data.example/")).toBeNull();
    });

    it("returns the parent directory of a workspace-relative path", () => {
      expect(SourceUtils.getParentSource("/proj/data/b.zarr")).toEqual({
        parentSource: "/proj/data",
        name: "b.zarr",
      });
    });

    it("returns null directly in the workspace", () => {
      expect(SourceUtils.getParentSource("/b.zarr")).toBeNull();
    });
  });

  describe("normalizeSource", () => {
    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("throws for an empty source", () => {
      expect(() =>
        SourceUtils.normalizeSource("", workspace, projectUrl, { baseUrl }),
      ).toThrow("Empty source");
    });

    describe("URLs", () => {
      it("returns URLs with a scheme normalized, regardless of project and workspace", () => {
        expect(
          SourceUtils.normalizeSource(
            "https://x.example/a/../f.csv",
            workspace,
            projectPath,
            { baseUrl },
          ),
        ).toBe("https://x.example/f.csv");
        expect(
          SourceUtils.normalizeSource("HTTPS://X.example", null, null, {
            baseUrl,
          }),
        ).toBe("https://x.example/");
      });

      it("passes blob and data URLs through", () => {
        expect(
          SourceUtils.normalizeSource(
            "blob:https://app.example/123",
            workspace,
            projectPath,
            { baseUrl },
          ),
        ).toBe("blob:https://app.example/123");
        expect(
          SourceUtils.normalizeSource(
            "data:text/csv,a%2Cb",
            workspace,
            projectPath,
            {
              baseUrl,
            },
          ),
        ).toBe("data:text/csv,a%2Cb");
      });

      it("throws for an invalid URL", () => {
        expect(() =>
          SourceUtils.normalizeSource("http://", workspace, projectPath, {
            baseUrl,
          }),
        ).toThrow("Invalid URL: http://");
      });

      it("takes a relative path with a colon in its first segment for a URL", () => {
        expect(
          SourceUtils.normalizeSource("s:c.tif", workspace, projectPath, {
            baseUrl,
          }),
        ).toBe("s:c.tif");
      });
    });

    describe("app-relative paths", () => {
      it("resolves against the base URL", () => {
        expect(
          SourceUtils.normalizeSource(
            "//data/points.csv",
            workspace,
            projectPath,
            {
              baseUrl,
            },
          ),
        ).toBe("https://app.example/tm/data/points.csv");
      });

      it("follows URL semantics for .. and a leading /", () => {
        expect(
          SourceUtils.normalizeSource("//../points.csv", null, null, {
            baseUrl,
          }),
        ).toBe("https://app.example/points.csv");
        expect(
          SourceUtils.normalizeSource("///points.csv", null, null, {
            baseUrl,
          }),
        ).toBe("https://app.example/points.csv");
      });

      it("defaults to the document base URL", () => {
        vi.stubGlobal("document", {
          baseURI: "https://other.example/x/y.html",
        });
        expect(SourceUtils.normalizeSource("//points.csv", null, null)).toBe(
          "https://other.example/x/points.csv",
        );
      });

      it("throws for an invalid base URL", () => {
        expect(() =>
          SourceUtils.normalizeSource("//points.csv", null, null, {
            baseUrl: "not a url",
          }),
        ).toThrow(TypeError);
      });
    });

    describe("workspace-relative paths", () => {
      it("collapses segments and keeps the prefix with an open workspace", () => {
        expect(
          SourceUtils.normalizeSource(
            "/shared/./sub/../x.csv",
            workspace,
            projectUrl,
            { baseUrl },
          ),
        ).toBe("/shared/x.csv");
        expect(
          SourceUtils.normalizeSource("/a//b/", workspace, null, {
            baseUrl,
          }),
        ).toBe("/a/b");
      });

      it("throws if the path leaves the workspace", () => {
        expect(() =>
          SourceUtils.normalizeSource("/../x.csv", workspace, null, {
            baseUrl,
          }),
        ).toThrow("Path escapes workspace");
        expect(() =>
          SourceUtils.normalizeSource("/a/../../x.csv", workspace, null, {
            baseUrl,
          }),
        ).toThrow("Path escapes workspace");
      });

      it("throws if the path names the workspace root", () => {
        expect(() =>
          SourceUtils.normalizeSource("/", workspace, null, { baseUrl }),
        ).toThrow("Not a workspace file or directory: /");
        expect(() =>
          SourceUtils.normalizeSource("/./a/..", workspace, null, {
            baseUrl,
          }),
        ).toThrow("Not a workspace file or directory: /./a/..");
      });

      it("falls back to being app-relative without a workspace", () => {
        expect(
          SourceUtils.normalizeSource("/data/points.csv", null, projectUrl, {
            baseUrl,
          }),
        ).toBe("https://app.example/tm/data/points.csv");
      });
    });

    describe("project-relative paths", () => {
      it("resolves against a project URL", () => {
        expect(
          SourceUtils.normalizeSource("points.csv", workspace, projectUrl, {
            baseUrl,
          }),
        ).toBe("https://data.example/projects/p1/points.csv");
        expect(
          SourceUtils.normalizeSource("./points.csv", null, projectUrl, {
            baseUrl,
          }),
        ).toBe("https://data.example/projects/p1/points.csv");
        expect(
          SourceUtils.normalizeSource("../shared/x.csv", null, projectUrl, {
            baseUrl,
          }),
        ).toBe("https://data.example/projects/shared/x.csv");
      });

      it("drops .. segments beyond the root of a project URL silently", () => {
        expect(
          SourceUtils.normalizeSource("../../../../x.csv", null, projectUrl, {
            baseUrl,
          }),
        ).toBe("https://data.example/x.csv");
      });

      it("keeps a colon in the first segment relative when prefixed with ./", () => {
        expect(
          SourceUtils.normalizeSource("./s:c.tif", null, projectUrl, {
            baseUrl,
          }),
        ).toBe("https://data.example/projects/p1/s:c.tif");
        expect(
          SourceUtils.normalizeSource("./s:c.tif", workspace, projectPath, {
            baseUrl,
          }),
        ).toBe("/proj/s:c.tif");
      });

      it("resolves within the project file's directory for a project loaded from the workspace", () => {
        expect(
          SourceUtils.normalizeSource("points.csv", workspace, projectPath, {
            baseUrl,
          }),
        ).toBe("/proj/points.csv");
        expect(
          SourceUtils.normalizeSource(
            "./sub/../points.csv",
            workspace,
            projectPath,
            {
              baseUrl,
            },
          ),
        ).toBe("/proj/points.csv");
        expect(
          SourceUtils.normalizeSource("sub/y.csv", workspace, projectPath, {
            baseUrl,
          }),
        ).toBe("/proj/sub/y.csv");
        expect(
          SourceUtils.normalizeSource(
            "../shared/x.csv",
            workspace,
            projectPath,
            {
              baseUrl,
            },
          ),
        ).toBe("/shared/x.csv");
      });

      it("throws if the path leaves the workspace for a project loaded from the workspace", () => {
        expect(() =>
          SourceUtils.normalizeSource("../../x.csv", workspace, projectPath, {
            baseUrl,
          }),
        ).toThrow("Path escapes workspace");
      });

      it("falls back to being app-relative for a project loaded from the workspace without an open workspace", () => {
        expect(
          SourceUtils.normalizeSource("points.csv", null, projectPath, {
            baseUrl,
          }),
        ).toBe("https://app.example/tm/proj/points.csv");
        expect(
          SourceUtils.normalizeSource("../shared/x.csv", null, projectPath, {
            baseUrl,
          }),
        ).toBe("https://app.example/tm/shared/x.csv");
      });

      it("throws if the path does not form a valid URL with the project URL", () => {
        expect(() =>
          SourceUtils.normalizeSource("points.csv", null, "http://", {
            baseUrl,
          }),
        ).toThrow("Invalid project-relative path");
      });

      it("falls back to being workspace-relative without a project source", () => {
        expect(
          SourceUtils.normalizeSource("shared/x.csv", workspace, null, {
            baseUrl,
          }),
        ).toBe("/shared/x.csv");
      });

      it("falls back to being app-relative without a project source and workspace", () => {
        expect(
          SourceUtils.normalizeSource("./data/points.csv", null, null, {
            baseUrl,
          }),
        ).toBe("https://app.example/tm/data/points.csv");
      });
    });

    describe("idempotence", () => {
      it.each<[string, FileSystemDirectoryHandle | null, string | null]>([
        ["https://x.example/a/../f.csv", workspace, projectPath],
        ["//data/points.csv", workspace, projectPath],
        ["/shared/./x.csv", workspace, projectPath],
        ["/shared/x.csv", null, projectPath],
        ["./sub/../points.csv", workspace, projectPath],
        ["../shared/x.csv", workspace, projectPath],
        ["points.csv", null, projectPath],
        ["points.csv", null, projectUrl],
        ["shared/x.csv", workspace, null],
        ["points.csv", null, null],
      ])(
        "normalizes the normalized form of %s unchanged",
        (source, ws, projectSource) => {
          const normalized = SourceUtils.normalizeSource(
            source,
            ws,
            projectSource,
            {
              baseUrl,
            },
          );
          expect(
            SourceUtils.normalizeSource(normalized, ws, projectSource, {
              baseUrl,
            }),
          ).toBe(normalized);
        },
      );
    });
  });

  describe("resolveSource", () => {
    it("returns URLs as is", async () => {
      await expect(
        SourceUtils.resolveSource("https://x.example/f.csv", workspace),
      ).resolves.toBe("https://x.example/f.csv");
      await expect(
        SourceUtils.resolveSource("blob:https://app.example/123", null),
      ).resolves.toBe("blob:https://app.example/123");
    });

    it("rejects URLs too if already aborted", async () => {
      const controller = new AbortController();
      controller.abort();
      await expect(
        SourceUtils.resolveSource("https://x.example/f.csv", workspace, {
          signal: controller.signal,
        }),
      ).rejects.toMatchObject({ name: "AbortError" });
    });

    it("opens a file in the workspace root", async () => {
      await expect(
        SourceUtils.resolveSource("/proj/points.csv", workspace),
      ).resolves.toBe(pointsFile);
    });

    it("opens a file in nested directories", async () => {
      await expect(
        SourceUtils.resolveSource("/proj/sub/y.csv", workspace),
      ).resolves.toBe(yFile);
      await expect(
        SourceUtils.resolveSource("/proj/s:c.tif", workspace),
      ).resolves.toBe(colonFile);
    });

    it("rejects a workspace-relative path without a workspace", async () => {
      await expect(
        SourceUtils.resolveSource("/proj/points.csv", null),
      ).rejects.toThrow("without workspace");
    });

    it("rejects if a directory or the file does not exist", async () => {
      await expect(
        SourceUtils.resolveSource("/missing/points.csv", workspace),
      ).rejects.toMatchObject({ name: "NotFoundError" });
      await expect(
        SourceUtils.resolveSource("/proj/missing.csv", workspace),
      ).rejects.toMatchObject({ name: "NotFoundError" });
    });

    it("opens a directory", async () => {
      await expect(
        SourceUtils.resolveSource("/proj/sub", workspace),
      ).resolves.toBe(subDir);
    });

    it("rejects if a segment other than the last names a file", async () => {
      await expect(
        SourceUtils.resolveSource("/proj/points.csv/x.csv", workspace),
      ).rejects.toMatchObject({ name: "TypeMismatchError" });
    });

    it("rejects if the path leaves the workspace or names the root", async () => {
      await expect(
        SourceUtils.resolveSource("/../x.csv", workspace),
      ).rejects.toThrow("Path escapes workspace");
      await expect(SourceUtils.resolveSource("/", workspace)).rejects.toThrow(
        "Not a workspace file or directory: /",
      );
    });

    it("rejects with the abort reason if already aborted", async () => {
      const controller = new AbortController();
      controller.abort();
      await expect(
        SourceUtils.resolveSource("/proj/points.csv", workspace, {
          signal: controller.signal,
        }),
      ).rejects.toMatchObject({ name: "AbortError" });
    });

    it("rejects with the abort reason if aborted while opening directories", async () => {
      const controller = new AbortController();
      const abortingWorkspace = makeDir(
        "",
        { proj: makeDir("proj", { "points.csv": pointsFile }) },
        { onOpen: () => controller.abort() },
      ) as unknown as FileSystemDirectoryHandle;
      await expect(
        SourceUtils.resolveSource("/proj/points.csv", abortingWorkspace, {
          signal: controller.signal,
        }),
      ).rejects.toMatchObject({ name: "AbortError" });
    });

    it("rejects with the abort reason if aborted while opening a directory entry", async () => {
      const controller = new AbortController();
      const abortingWorkspace = makeDir(
        "",
        { sub: makeDir("sub", {}) },
        { onOpen: () => controller.abort() },
      ) as unknown as FileSystemDirectoryHandle;
      await expect(
        SourceUtils.resolveSource("/sub", abortingWorkspace, {
          signal: controller.signal,
        }),
      ).rejects.toMatchObject({ name: "AbortError" });
    });

    it("never throws synchronously", () => {
      const controller = new AbortController();
      controller.abort();
      const promises = [
        SourceUtils.resolveSource("/proj/points.csv", null),
        SourceUtils.resolveSource("/proj/points.csv", workspace, {
          signal: controller.signal,
        }),
        SourceUtils.resolveSource("/../x.csv", workspace),
      ];
      for (const promise of promises) {
        expect(promise).toBeInstanceOf(Promise);
        promise.catch(() => {});
      }
    });
  });

  describe("resolveSourceFile", () => {
    it("returns URLs as is", async () => {
      await expect(
        SourceUtils.resolveSourceFile("https://x.example/f.csv", workspace),
      ).resolves.toBe("https://x.example/f.csv");
    });

    it("opens a file", async () => {
      await expect(
        SourceUtils.resolveSourceFile("/proj/sub/y.csv", workspace),
      ).resolves.toBe(yFile);
    });

    it("rejects a directory", async () => {
      await expect(
        SourceUtils.resolveSourceFile("/proj/sub", workspace),
      ).rejects.toMatchObject({
        name: "TypeMismatchError",
        message: "Not a workspace file: /proj/sub",
      });
    });

    it("rejects like resolveSource", async () => {
      await expect(
        SourceUtils.resolveSourceFile("/proj/missing.csv", workspace),
      ).rejects.toMatchObject({ name: "NotFoundError" });
    });
  });

  describe("openSourceFile", () => {
    const opened = new File(["a,b"], "y.csv");
    let onGetFile: (() => void) | undefined;
    const openableWorkspace = makeDir("", {
      proj: makeDir("proj", {
        sub: makeDir("sub", {
          "y.csv": {
            ...makeFile("y.csv"),
            getFile: () => {
              onGetFile?.();
              return Promise.resolve(opened);
            },
          },
        }),
      }),
    }) as unknown as FileSystemDirectoryHandle;

    afterEach(() => {
      onGetFile = undefined;
    });

    it("returns URLs as is", async () => {
      await expect(
        SourceUtils.openSourceFile("https://x.example/f.csv", workspace),
      ).resolves.toEqual({ url: "https://x.example/f.csv" });
    });

    it("opens a file", async () => {
      await expect(
        SourceUtils.openSourceFile("/proj/sub/y.csv", openableWorkspace),
      ).resolves.toEqual({ file: opened });
    });

    it("rejects like resolveSourceFile", async () => {
      await expect(
        SourceUtils.openSourceFile("/proj/sub", workspace),
      ).rejects.toMatchObject({ name: "TypeMismatchError" });
    });

    it("rejects with the abort reason if aborted while opening the file", async () => {
      const controller = new AbortController();
      const reason = new Error("stop");
      onGetFile = () => controller.abort(reason);
      await expect(
        SourceUtils.openSourceFile("/proj/sub/y.csv", openableWorkspace, {
          signal: controller.signal,
        }),
      ).rejects.toBe(reason);
    });
  });

  describe("resolveSourceDirectory", () => {
    it("returns URLs as is", async () => {
      await expect(
        SourceUtils.resolveSourceDirectory(
          "https://x.example/d.zarr",
          workspace,
        ),
      ).resolves.toBe("https://x.example/d.zarr");
    });

    it("opens a directory", async () => {
      await expect(
        SourceUtils.resolveSourceDirectory("/proj/sub", workspace),
      ).resolves.toBe(subDir);
    });

    it("rejects a file", async () => {
      await expect(
        SourceUtils.resolveSourceDirectory("/proj/points.csv", workspace),
      ).rejects.toMatchObject({
        name: "TypeMismatchError",
        message: "Not a workspace directory: /proj/points.csv",
      });
    });

    it("rejects like resolveSource", async () => {
      await expect(
        SourceUtils.resolveSourceDirectory("/proj/missing", workspace),
      ).rejects.toMatchObject({ name: "NotFoundError" });
    });
  });
});
