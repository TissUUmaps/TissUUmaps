// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import {
  gitHubLimits,
  minimalSkillMd,
  parseGitHubUrl,
  readGitHubSkill,
  skillName,
} from "./github-skills";
import { pyodidePackages, skillPackageName } from "./python-imports";
import { confirmGitHubSkill } from "./skill-confirm";

/** The commit of the fake repository */
const commit = "a7e9fa555837a46ee47a02d7f6e4a70cb3892b0a";

/** The packages of the fake Pyodide */
const pyodide = () =>
  Promise.resolve(
    pyodidePackages(
      JSON.stringify({
        packages: {
          numpy: { name: "numpy", imports: ["numpy"] },
          pandas: { name: "pandas", imports: ["pandas"] },
        },
      }),
    ),
  );

/**
 * A fake GitHub: the API (repository, commit, files) and the raw files,
 * which lists the addresses it is asked for
 */
function fakeGitHub(options: {
  files: Record<string, string | Uint8Array>;
  license?: { spdx_id: string; name: string } | null;
  sizes?: Record<string, number>;
  status?: number;
  headers?: Record<string, string>;
}) {
  const asked: string[] = [];
  const fetchFile = (url: string | URL | Request) => {
    const address = url instanceof Request ? url.url : url.toString();
    asked.push(address);
    const json = (value: unknown) =>
      Promise.resolve(
        new Response(JSON.stringify(value), {
          status: options.status ?? 200,
          headers: options.headers ?? {},
        }),
      );
    if (address === "https://api.github.com/repos/lab/tools") {
      return json({
        full_name: "lab/tools",
        default_branch: "main",
        description: "Analytical tools",
        license: options.license === undefined ? null : options.license,
      });
    }
    if (address.startsWith("https://api.github.com/repos/lab/tools/commits")) {
      return json([{ sha: commit }]);
    }
    if (
      address.startsWith("https://api.github.com/repos/lab/tools/git/trees")
    ) {
      return json({
        tree: [
          { path: "src", type: "tree" },
          ...Object.entries(options.files).map(([path, content]) => ({
            path,
            type: "blob",
            size: options.sizes?.[path] ?? content.length,
          })),
        ],
      });
    }
    const raw = `https://raw.githubusercontent.com/lab/tools/${commit}/`;
    if (address.startsWith(raw)) {
      const path = decodeURIComponent(address.slice(raw.length));
      const content = options.files[path];
      return Promise.resolve(
        content === undefined
          ? new Response("", { status: 404 })
          : new Response(content as BodyInit),
      );
    }
    return Promise.resolve(new Response("", { status: 404 }));
  };
  return { asked, fetchFile };
}

describe("github-skills", () => {
  it("reads the addresses of repositories and of their folders", () => {
    expect(parseGitHubUrl("https://github.com/lab/tools")).toEqual({
      owner: "lab",
      repo: "tools",
      path: "",
    });
    expect(parseGitHubUrl("github.com/lab/tools.git/")).toEqual({
      owner: "lab",
      repo: "tools",
      path: "",
    });
    expect(
      parseGitHubUrl("https://github.com/lab/tools/tree/v2/skills/my%20skill"),
    ).toEqual({
      owner: "lab",
      repo: "tools",
      ref: "v2",
      path: "skills/my skill",
    });
    expect(
      parseGitHubUrl("https://github.com/lab/tools/blob/main/src/a.py?plain=1"),
    ).toEqual({ owner: "lab", repo: "tools", ref: "main", path: "src" });
    expect(parseGitHubUrl("lab/tools")).toEqual({
      owner: "lab",
      repo: "tools",
      path: "",
    });
    expect(() => parseGitHubUrl("https://gitlab.com/lab/tools")).toThrow(
      "is not the address of a repository on GitHub",
    );
    expect(() =>
      parseGitHubUrl("https://github.com/lab/tools/issues/3"),
    ).toThrow(
      "is not the address of a repository or of a folder of one on GitHub",
    );
    expect(skillName("Analytical_Enrichment Test!")).toBe(
      "analytical-enrichment-test",
    );
  });

  it("keeps only text and code, at the exact commit, notebooks without their outputs, and writes a minimal SKILL.md", async () => {
    const notebook = JSON.stringify({
      cells: [
        {
          cell_type: "code",
          source: "import seaborn",
          outputs: [{ data: { "image/png": "BIG" } }],
        },
      ],
      metadata: {},
      nbformat: 4,
    });
    const { asked, fetchFile } = fakeGitHub({
      license: { spdx_id: "MIT", name: "MIT License" },
      files: {
        LICENSE: "MIT License",
        "README.md": "# Tools",
        "tools/__init__.py": "import numpy\nimport squidpy\n",
        "example.ipynb": notebook,
        "logo.png": "PNG",
        ".github/workflows/ci.yml": "on: push",
        "data/big.json": "{}",
        "binary.txt": new Uint8Array([0x66, 0, 0x66]),
      },
      sizes: { "data/big.json": gitHubLimits.fileBytes + 1 },
    });
    const progress: string[] = [];
    const preview = await readGitHubSkill("https://github.com/lab/tools", {
      fetchFile,
      pyodide,
      onProgress: (text) => progress.push(text),
    });
    // three requests to the API, then the files, at the commit
    expect(asked.slice(0, 3)).toEqual([
      "https://api.github.com/repos/lab/tools",
      "https://api.github.com/repos/lab/tools/commits?sha=main&per_page=1",
      `https://api.github.com/repos/lab/tools/git/trees/${commit}?recursive=1`,
    ]);
    expect(
      asked
        .slice(3)
        .every((url) =>
          url.startsWith(
            `https://raw.githubusercontent.com/lab/tools/${commit}/`,
          ),
        ),
    ).toBe(true);
    expect(preview.origin).toEqual({
      repo: "lab/tools",
      path: "",
      commit,
      license: "MIT",
    });
    expect(preview.url).toBe("https://github.com/lab/tools");
    expect(preview.files.map(({ path }) => path).sort()).toEqual([
      "tools/LICENSE",
      "tools/README.md",
      "tools/SKILL.md",
      "tools/example.ipynb",
      "tools/tools/__init__.py",
    ]);
    expect(preview.skipped).toEqual([
      { path: ".github/workflows/ci.yml", reason: "hidden" },
      { path: "data/big.json", reason: "bigger than 2 MB" },
      { path: "logo.png", reason: "not text or code" },
      { path: "binary.txt", reason: "not a text file" },
    ]);
    const kept = await preview.files
      .find(({ path }) => path === "tools/example.ipynb")!
      .content.text();
    expect(kept).not.toContain("BIG");
    expect(kept).toContain("import seaborn");
    expect(preview.madeSkillMd).toBe(true);
    const [skill] = preview.skills;
    expect(skill?.name).toBe("tools");
    expect(skill?.license).toBe("MIT");
    expect(skill?.description).toBe(
      "Code of the GitHub repository lab/tools (Analytical tools), which has no instructions of its own yet. " +
        "Use it when the user asks for what this code does, or names the repository.",
    );
    expect(skill?.metadata).toEqual({
      source: "https://github.com/lab/tools",
      commit,
      "written-by": "TissUUmaps AI chat",
    });
    expect(skill?.instructions).toContain(
      'write your own code with run_code, with language "python" and\n   skill "tools"',
    );
    expect(
      preview.python.packages.map(({ name, pyodide }) => [name, pyodide]),
    ).toEqual([
      ["numpy", "numpy"],
      ["seaborn", undefined],
      ["squidpy", undefined],
    ]);
    expect(preview.python.broken.map(({ path }) => path)).toEqual([
      "example.ipynb",
      "tools/__init__.py",
    ]);
    expect(progress.at(-1)).toBe("checking the packages of Pyodide");
  });

  it("uses the SKILL.md of a folder, and says when a repository has no license", async () => {
    const { fetchFile } = fakeGitHub({
      files: {
        "skills/counter/SKILL.md":
          "---\nname: counter\ndescription: Counts.\n---\n\nRun scripts/count.py.",
        "skills/counter/scripts/count.py": "async def main(tm):\n    return 1",
        "other/file.py": "import x",
      },
    });
    const preview = await readGitHubSkill(
      "https://github.com/lab/tools/tree/main/skills/counter",
      { fetchFile, pyodide },
    );
    expect(preview.origin).toEqual({
      repo: "lab/tools",
      path: "skills/counter",
      commit,
    });
    expect(preview.url).toBe(
      `https://github.com/lab/tools/tree/${commit}/skills/counter`,
    );
    expect(preview.madeSkillMd).toBe(false);
    expect(preview.skills.map(({ name }) => name)).toEqual(["counter"]);
    expect([...preview.skills[0]!.files.keys()]).toEqual(["scripts/count.py"]);
    expect(preview.python).toEqual({
      hasPython: true,
      packages: [],
      broken: [],
    });
  });

  it("takes the license of the repository with a folder of it, and keeps a Python package importable by its name", async () => {
    const { fetchFile } = fakeGitHub({
      license: { spdx_id: "MIT", name: "MIT License" },
      files: {
        LICENSE: "MIT License\n\nCopyright (c) 2024 The authors",
        "README.md": "# Tools",
        "points2regions/__init__.py":
          "from ._points2regions import Points2Regions\n",
        "points2regions/_points2regions.py": "import scipy.sparse as sp\n",
      },
    });
    const preview = await readGitHubSkill(
      "https://github.com/lab/tools/tree/main/points2regions",
      { fetchFile, pyodide },
    );
    expect(preview.origin.license).toBe("MIT");
    expect(preview.topFiles).toEqual(["LICENSE"]);
    const skill = preview.skills[0]!;
    expect(skill.name).toBe("points2regions");
    expect([...skill.files.keys()].sort()).toEqual([
      "LICENSE",
      "__init__.py",
      "_points2regions.py",
    ]);
    expect(await skill.files.get("LICENSE")!.text()).toContain(
      "Copyright (c) 2024 The authors",
    );
    // the README of the top stays out: only the license comes with the folder
    expect(skill.files.has("README.md")).toBe(false);
    expect(skillPackageName(skill)).toBe("points2regions");

    // the card says where the text of the license comes from
    const card = confirmGitHubSkill(preview);
    expect(card.element.textContent).toContain(
      "License: MIT; its text (LICENSE) comes from the top of the repository, and is added to the skill",
    );
  });

  it("says why a repository cannot be read", async () => {
    const limited = fakeGitHub({
      files: {},
      status: 403,
      headers: {
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": "1893456000",
      },
    });
    await expect(
      readGitHubSkill("lab/tools", { fetchFile: limited.fetchFile, pyodide }),
    ).rejects.toThrow(
      "GitHub allows 60 requests an hour from this address without an account, and they are used up: try again after",
    );
    const missing = fakeGitHub({ files: {}, status: 404 });
    await expect(
      readGitHubSkill("lab/tools", { fetchFile: missing.fetchFile, pyodide }),
    ).rejects.toThrow("GitHub did not find it");
    const empty = fakeGitHub({ files: { "logo.png": "PNG" } });
    await expect(
      readGitHubSkill("lab/tools", { fetchFile: empty.fetchFile, pyodide }),
    ).rejects.toThrow(
      "the folder has no text or code files to make a skill of",
    );
    const folder = fakeGitHub({ files: { "a.py": "" } });
    await expect(
      readGitHubSkill("https://github.com/lab/tools/tree/main/nope", {
        fetchFile: folder.fetchFile,
        pyodide,
      }),
    ).rejects.toThrow('the repository has no folder "nope", or it is empty');
  });

  it("writes a valid minimal SKILL.md, also with quotes in the description", () => {
    const text = minimalSkillMd(
      "tools",
      { repo: "lab/tools", path: "", commit },
      'The "best" tools: fast',
    );
    expect(text).toContain(
      'description: "Code of the GitHub repository lab/tools (The \\"best\\" tools: fast), which has no instructions',
    );
    expect(text).not.toContain("license:");
  });
});
