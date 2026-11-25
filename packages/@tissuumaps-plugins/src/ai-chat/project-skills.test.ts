// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import { loadProjectSkills } from "./project-skills";

/** A folder: files by name with their text, and folders by name */
type Tree = { [name: string]: string | Tree };

/** A folder of the workspace, as the File System API gives it */
function fakeDirectory(name: string, tree: Tree): FileSystemDirectoryHandle {
  const entries = Object.entries(tree).map(([entryName, value]) =>
    typeof value === "string"
      ? {
          kind: "file",
          name: entryName,
          getFile: () => Promise.resolve(new File([value], entryName)),
        }
      : fakeDirectory(entryName, value),
  );
  const find = (entryName: string, kind: string) => {
    const entry = entries.find((candidate) => candidate.name === entryName);
    if (entry === undefined) {
      return Promise.reject(new DOMException(entryName, "NotFoundError"));
    }
    if (entry.kind !== kind) {
      return Promise.reject(new DOMException(entryName, "TypeMismatchError"));
    }
    return Promise.resolve(entry);
  };
  return {
    kind: "directory",
    name,
    getDirectoryHandle: (entryName: string) => find(entryName, "directory"),
    getFileHandle: (entryName: string) => find(entryName, "file"),
    async *values() {
      yield* await Promise.resolve(entries);
    },
  } as unknown as FileSystemDirectoryHandle;
}

/** A SKILL.md with a name */
function skillMd(name: string): string {
  return `---\nname: ${name}\ndescription: What ${name} does.\n---\n\nSteps.`;
}

/** Answers fetch() with the given files of a web server, by URL */
function fakeServer(files: Record<string, string>) {
  const fetchMock = vi.fn((url: URL | string) => {
    const text = files[String(url)];
    return Promise.resolve(
      text === undefined
        ? new Response("not found", { status: 404, statusText: "Not Found" })
        : new Response(text),
    );
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("loadProjectSkills in the workspace", () => {
  const workspace = fakeDirectory("root", {
    xenium: {
      "project.json": "{}",
      skills: {
        "lab-colors": {
          "SKILL.md": skillMd("lab-colors"),
          references: { "palette.md": "T cell: blue" },
        },
        hello: { "SKILL.md": skillMd("hello") },
        ".git": { HEAD: "ref" },
        Broken: { "SKILL.md": skillMd("Broken") },
      },
    },
    other: { "project.json": "{}" },
  });

  it("reads the skills folder next to the project file", async () => {
    const project = await loadProjectSkills("/xenium/project.json", workspace);
    expect(project?.where).toBe("/xenium/skills");
    expect(project?.skills.map((skill) => skill.name)).toEqual([
      "hello",
      "lab-colors",
    ]);
    const labColors = project?.skills[1];
    expect([...(labColors?.files.keys() ?? [])]).toEqual([
      "references/palette.md",
    ]);
    expect(project?.problems).toEqual([
      {
        folder: "Broken",
        error: 'the name "Broken" must be lowercase',
      },
    ]);
  });

  it("has no skills without a skills folder, a workspace or a project file", async () => {
    expect(await loadProjectSkills("/other/project.json", workspace)).toBe(
      null,
    );
    expect(await loadProjectSkills("/xenium/project.json", null)).toBe(null);
    expect(await loadProjectSkills(null, workspace)).toBe(null);
  });
});

describe("loadProjectSkills from a URL", () => {
  const base = "https://example.org/data/xenium";

  it("downloads the files that skills/index.json lists", async () => {
    fakeServer({
      [`${base}/skills/index.json`]: JSON.stringify({
        files: [
          "lab-colors/SKILL.md",
          "lab-colors/references/palette.md",
          "hello/SKILL.md",
        ],
      }),
      [`${base}/skills/lab-colors/SKILL.md`]: skillMd("lab-colors"),
      [`${base}/skills/lab-colors/references/palette.md`]: "T cell: blue",
      [`${base}/skills/hello/SKILL.md`]: skillMd("hello"),
    });
    const project = await loadProjectSkills(`${base}/project.json`, null);
    expect(project?.where).toBe(`${base}/skills`);
    expect(project?.skills.map((skill) => skill.name)).toEqual([
      "hello",
      "lab-colors",
    ]);
    expect(
      await project?.skills[1]?.files.get("references/palette.md")?.text(),
    ).toBe("T cell: blue");
    expect(project?.problems).toEqual([]);
  });

  it("has no skills without skills/index.json", async () => {
    const fetchMock = fakeServer({});
    expect(await loadProjectSkills(`${base}/project.json`, null)).toBe(null);
    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      `${base}/skills/index.json`,
    );
  });

  it("refuses paths outside the skills folder, and says which files are missing", async () => {
    fakeServer({
      [`${base}/skills/index.json`]: JSON.stringify({
        files: [
          "hello/SKILL.md",
          "hello/notes.md",
          "../project.json",
          "/etc/passwd",
          "https://example.com/x",
        ],
      }),
      [`${base}/skills/hello/SKILL.md`]: skillMd("hello"),
    });
    const project = await loadProjectSkills(`${base}/project.json`, null);
    expect(project?.skills.map((skill) => skill.name)).toEqual(["hello"]);
    expect(project?.problems).toEqual([
      {
        folder: "hello",
        error: "hello/notes.md cannot be downloaded (404 Not Found)",
      },
      {
        folder: "..",
        error:
          '"../project.json" in skills/index.json is not a path inside the skills folder',
      },
      {
        folder: "",
        error:
          '"/etc/passwd" in skills/index.json is not a path inside the skills folder',
      },
      {
        folder: "https:",
        error:
          '"https://example.com/x" in skills/index.json is not a path inside the skills folder',
      },
    ]);
  });

  it.each([
    ["not JSON", "{files:", "skills/index.json is not valid JSON"],
    [
      "a list",
      '["hello/SKILL.md"]',
      'skills/index.json must be {"files": [...]}',
    ],
    [
      "not texts",
      '{"files": [1]}',
      'skills/index.json must be {"files": [...]}',
    ],
  ])("explains an index that is %s", async (_, index, error) => {
    fakeServer({ [`${base}/skills/index.json`]: index });
    await expect(
      loadProjectSkills(`${base}/project.json`, null),
    ).rejects.toThrow(error);
  });
});
