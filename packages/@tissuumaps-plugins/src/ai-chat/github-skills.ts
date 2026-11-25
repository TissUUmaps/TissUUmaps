import { withoutOutputs } from "./notebooks";
import {
  type PyodidePackages,
  type PythonNeeds,
  analyzePython,
} from "./python-imports";
import {
  type FolderFile,
  type Skill,
  type SkillProblem,
  parseFrontmatter,
  readSkills,
} from "./skills";

// A skill can come from a GitHub repository, or from a folder of one. The
// panel reads it in the page (the sandbox never has network) with three
// requests to the API of GitHub (the repository, its commit, its files),
// which allows 60 an hour without an account, then downloads the files from
// raw.githubusercontent.com, at that exact commit. Only text and code are
// kept, within limits, and notebooks without their outputs. Nothing is kept
// until the user has seen what the skill holds and said yes. A repository
// without SKILL.md gets a minimal one, which says where the code comes from.

/** Where a skill from GitHub comes from */
export type GitHubOrigin = {
  /** e.g. "wahlby-lab/analytical-enrichment-test" */
  repo: string;
  /** The folder in the repository, or "" for all of it */
  path: string;
  /** The full SHA of the commit that the files come from */
  commit: string;
  /** The SPDX identifier of the license (e.g. "MIT"), or its name; none without a license */
  license?: string;
};

/** A file of the repository that was not kept, and why */
export type SkippedFile = { path: string; reason: string };

/** What a repository would add, for the user to confirm */
export type GitHubPreview = {
  origin: GitHubOrigin;
  /** The address of the repository, or of its folder, on github.com */
  url: string;
  /** The description of the repository on GitHub */
  description: string;
  /** The skills it adds, with their files */
  skills: Skill[];
  /** The skill folders that could not be read */
  problems: SkillProblem[];
  /** The files kept, by path in the folder of their skill, e.g. "anhood/SKILL.md" */
  files: FolderFile[];
  /** How many bytes the kept files have */
  bytes: number;
  /** The files left out */
  skipped: SkippedFile[];
  /** Whether the plugin wrote SKILL.md, since the repository has none */
  madeSkillMd: boolean;
  /**
   * The license files that a folder of the repository takes from its top
   * (e.g. "LICENSE"), since the license of a repository covers its folders
   */
  topFiles: string[];
  /** What its Python code needs */
  python: PythonNeeds;
};

/** The endings of the files that are kept: text and code */
export const keptEndings = [
  ".py",
  ".js",
  ".mjs",
  ".md",
  ".txt",
  ".json",
  ".toml",
  ".yaml",
  ".yml",
  ".ipynb",
];

/** The files kept whatever their ending, e.g. the license */
const keptNames = /^(licen[cs]e|copying|notice)(\.(md|txt|rst))?$/i;

/** The limits of a skill from GitHub */
export const gitHubLimits = {
  /** The largest file, in bytes */
  fileBytes: 2_000_000,
  /** The largest notebook before its outputs are removed, in bytes */
  notebookBytes: 20_000_000,
  /** All the files kept, in bytes */
  totalBytes: 10_000_000,
  /** The most files kept */
  files: 300,
};

/** How many files are downloaded at the same time */
const parallelDownloads = 6;

/**
 * Reads the address of a repository, or of a folder of one
 *
 * @param text - e.g. "https://github.com/owner/repo", ".../tree/main/skills/x"
 * or "owner/repo"
 * @returns Its owner, name, and, if the address gives them, the branch, tag or
 * commit, and the folder
 * @throws If it is not the address of a repository on GitHub
 */
export function parseGitHubUrl(text: string): {
  owner: string;
  repo: string;
  ref?: string;
  path: string;
} {
  const plain = text
    .trim()
    .replace(/^<|>$/g, "")
    .replace(/^(https?:\/\/)?(www\.)?github\.com\//i, "")
    .replace(/[?#].*$/, "")
    .replace(/\/+$/, "");
  const parts = plain.split("/");
  const [owner = "", repo = ""] = parts;
  const valid = /^[A-Za-z0-9_.-]+$/;
  if (
    /^[a-z]+:\/\//i.test(plain) ||
    !valid.test(owner) ||
    !valid.test(repo.replace(/\.git$/, ""))
  ) {
    throw new Error(
      `"${text.trim()}" is not the address of a repository on GitHub, e.g. https://github.com/owner/repository`,
    );
  }
  const kind = parts[2];
  if (kind !== undefined && kind !== "tree" && kind !== "blob") {
    throw new Error(
      `"${text.trim()}" is not the address of a repository or of a folder of one on GitHub`,
    );
  }
  const ref = parts[3];
  let path = parts
    .slice(4)
    .map((part) => decodeURIComponent(part))
    .join("/");
  // the address of a file: its folder
  if (kind === "blob") {
    path = path.split("/").slice(0, -1).join("/");
  }
  return {
    owner,
    repo: repo.replace(/\.git$/, ""),
    ...(ref === undefined ? {} : { ref }),
    path,
  };
}

/**
 * The name of a skill made from a folder, as the format wants it
 *
 * @param name - e.g. "Analytical_Enrichment_Test"
 * @returns e.g. "analytical-enrichment-test"
 */
export function skillName(name: string): string {
  return (
    name
      .normalize("NFKD")
      .replace(/[̀-ͯ]/g, "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 64)
      .replace(/-+$/, "") || "github-skill"
  );
}

/**
 * Writes a value of the frontmatter in double quotes
 *
 * @param text - The value
 * @returns e.g. "\"Code of \\\"x\\\"\""
 */
function quoted(text: string): string {
  return JSON.stringify(text.replace(/\s+/g, " ").trim());
}

/**
 * The SKILL.md that the plugin writes for a repository that has none: it says
 * where the code comes from, and how the model can learn to use it
 *
 * @param name - The name of the skill
 * @param origin - Where it comes from
 * @param about - The description of the repository on GitHub
 * @returns The text of SKILL.md
 */
export function minimalSkillMd(
  name: string,
  origin: GitHubOrigin,
  about: string,
): string {
  const where = `${origin.repo}${origin.path === "" ? "" : `/${origin.path}`}`;
  const summary = about.trim() === "" ? "" : ` (${about.trim().slice(0, 300)})`;
  return [
    "---",
    `name: ${name}`,
    `description: ${quoted(
      `Code of the GitHub repository ${where}${summary}, which has no instructions of its own yet. ` +
        "Use it when the user asks for what this code does, or names the repository.",
    )}`,
    ...(origin.license === undefined
      ? []
      : [`license: ${quoted(origin.license)}`]),
    "metadata:",
    `  source: ${quoted(`https://github.com/${where}`)}`,
    `  commit: ${quoted(origin.commit)}`,
    "  written-by: TissUUmaps AI chat",
    "---",
    "",
    `This skill was made by the AI chat from the GitHub repository ${where}, at the commit`,
    `${origin.commit.slice(0, 7)}. The repository has no SKILL.md, so these instructions are only a start.`,
    "",
    "1. Read its README and the files it points to (code, notebooks) with read_skill_file,",
    "   to learn what the code does and how to call it. Notebooks come as text, without",
    "   their outputs.",
    `2. To use its Python code, write your own code with run_code, with language "python" and`,
    `   skill "${name}": the code can import the modules of the repository, as from its top`,
    "   folder (and its src folder, if it has one). Only the packages of Pyodide are there:",
    "   the parts of the repository that need other packages do not work. Say so when a",
    "   request needs them.",
    "3. Once you have used the code and know it, propose better instructions to the user with",
    "   propose_skill_instructions: what the code does, which functions to call and how.",
    "",
  ].join("\n");
}

/** A file of the tree of a repository, as the API of GitHub gives it */
type TreeEntry = { path: string; type: string; size?: number };

/**
 * Asks the API of GitHub
 *
 * @param path - The path after https://api.github.com/
 * @param fetchFile - fetch, or another function like it in the tests
 * @param signal - Optional abort signal
 * @returns The answer, as JSON
 * @throws If GitHub does not answer, or with an error
 */
async function askGitHub(
  path: string,
  fetchFile: typeof fetch,
  signal?: AbortSignal,
): Promise<unknown> {
  const response = await fetchFile(`https://api.github.com/${path}`, {
    headers: { Accept: "application/vnd.github+json" },
    ...(signal === undefined ? {} : { signal }),
  });
  if (response.ok) {
    return response.json();
  }
  if (
    (response.status === 403 || response.status === 429) &&
    response.headers.get("x-ratelimit-remaining") === "0"
  ) {
    const reset = Number(response.headers.get("x-ratelimit-reset"));
    const when = Number.isFinite(reset)
      ? ` after ${new Date(reset * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
      : " later";
    throw new Error(
      `GitHub allows 60 requests an hour from this address without an account, and they are used up: try again${when}`,
    );
  }
  if (response.status === 404 || response.status === 422) {
    throw new Error(
      "GitHub did not find it: the repository, branch or folder does not exist, or is private",
    );
  }
  throw new Error(`GitHub answered ${response.status} ${response.statusText}`);
}

/**
 * Whether a file of a repository is kept, and if not, why
 *
 * @param path - The path of the file in the chosen folder
 * @param size - Its size in bytes
 * @returns Why it is left out, or "" if it is kept
 */
function leftOutBecause(path: string, size: number): string {
  const name = path.split("/").at(-1)!;
  if (path.split("/").some((part) => part.startsWith("."))) {
    return "hidden";
  }
  const ending = /\.[^.]+$/.exec(name)?.[0].toLowerCase() ?? "";
  if (!keptEndings.includes(ending) && !keptNames.test(name)) {
    return "not text or code";
  }
  const limit =
    ending === ".ipynb" ? gitHubLimits.notebookBytes : gitHubLimits.fileBytes;
  if (size > limit) {
    return `bigger than ${limit / 1e6} MB`;
  }
  return "";
}

/**
 * Reads a skill from GitHub, for the user to confirm: what the repository (or
 * its folder) holds at its latest commit, or at the given one, and what its
 * Python code needs. Nothing is kept yet.
 *
 * @param address - The address of the repository, or of a folder of it
 * @param options - fetch (another function in the tests), the packages of
 * Pyodide (or why they are not known), what to do with the progress, and an
 * optional abort signal
 * @returns What the skill would hold
 * @throws If the repository cannot be read, or holds nothing to keep
 */
export async function readGitHubSkill(
  address: string,
  options: {
    fetchFile?: typeof fetch;
    pyodide: () => Promise<PyodidePackages>;
    onProgress?: (text: string) => void;
    signal?: AbortSignal;
  },
): Promise<GitHubPreview> {
  const { fetchFile = fetch.bind(globalThis), signal, onProgress } = options;
  signal?.throwIfAborted();
  const { owner, repo, ref, path } = parseGitHubUrl(address);
  const name = `${owner}/${repo}`;
  onProgress?.(`reading ${name} on GitHub`);
  const info = (await askGitHub(`repos/${name}`, fetchFile, signal)) as {
    full_name?: string;
    default_branch?: string;
    description?: string | null;
    license?: { spdx_id?: string | null; name?: string | null } | null;
  };
  const fullName = info.full_name ?? name;
  const branch = ref ?? info.default_branch ?? "HEAD";
  const commits = (await askGitHub(
    `repos/${fullName}/commits?sha=${encodeURIComponent(branch)}&per_page=1`,
    fetchFile,
    signal,
  )) as { sha?: string }[];
  const commit = commits[0]?.sha;
  if (commit === undefined || !/^[0-9a-f]{40}$/.test(commit)) {
    throw new Error(`GitHub gave no commit for "${branch}"`);
  }
  const tree = (await askGitHub(
    `repos/${fullName}/git/trees/${commit}?recursive=1`,
    fetchFile,
    signal,
  )) as { tree?: TreeEntry[]; truncated?: boolean };
  const spdx = info.license?.spdx_id;
  const license =
    spdx !== undefined && spdx !== null && spdx !== "NOASSERTION"
      ? spdx
      : (info.license?.name ?? undefined);
  const origin: GitHubOrigin = {
    repo: fullName,
    path,
    commit,
    ...(license === undefined || license === null ? {} : { license }),
  };

  // the files of the chosen folder, within the limits
  const prefix = path === "" ? "" : `${path}/`;
  const inFolder = (tree.tree ?? []).filter(
    (entry) => entry.type === "blob" && entry.path.startsWith(prefix),
  );
  if (path !== "" && inFolder.length === 0) {
    throw new Error(`the repository has no folder "${path}", or it is empty`);
  }
  // a folder takes with it the license of the repository, from its top,
  // unless it has a file of the same name
  const own = new Set(
    inFolder.map((entry) => entry.path.slice(prefix.length).toLowerCase()),
  );
  const topLicenses =
    path === ""
      ? []
      : (tree.tree ?? []).filter(
          (entry) =>
            entry.type === "blob" &&
            !entry.path.includes("/") &&
            keptNames.test(entry.path) &&
            (entry.size ?? 0) <= gitHubLimits.fileBytes &&
            !own.has(entry.path.toLowerCase()),
        );
  /** The path of a file in the skill's folder */
  const insideOf = (entry: TreeEntry) =>
    entry.path.startsWith(prefix)
      ? entry.path.slice(prefix.length)
      : entry.path;
  const skipped: SkippedFile[] = [];
  const wanted: TreeEntry[] = [...topLicenses];
  let planned = 0;
  for (const entry of inFolder.sort((a, b) => a.path.localeCompare(b.path))) {
    const inside = entry.path.slice(prefix.length);
    const reason = leftOutBecause(inside, entry.size ?? 0);
    if (reason !== "") {
      skipped.push({ path: inside, reason });
    } else if (wanted.length >= gitHubLimits.files) {
      skipped.push({
        path: inside,
        reason: `over ${gitHubLimits.files} files`,
      });
    } else {
      wanted.push(entry);
      planned += entry.path.endsWith(".ipynb") ? 0 : (entry.size ?? 0);
    }
  }
  if (planned > gitHubLimits.totalBytes * 2) {
    throw new Error(
      `the folder has ${Math.round(planned / 1e6)} MB of text and code, too much for a skill (at most ${gitHubLimits.totalBytes / 1e6} MB)`,
    );
  }

  // the files, at the commit, as text
  const texts = new Map<string, string>();
  let next = 0;
  let done = 0;
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const download = async () => {
    while (next < wanted.length) {
      const entry = wanted[next++]!;
      const inside = insideOf(entry);
      const url = `https://raw.githubusercontent.com/${fullName}/${commit}/${entry.path
        .split("/")
        .map(encodeURIComponent)
        .join("/")}`;
      const response = await fetchFile(
        url,
        signal === undefined ? {} : { signal },
      );
      if (!response.ok) {
        skipped.push({
          path: inside,
          reason: `not downloaded (${response.status})`,
        });
        continue;
      }
      let text: string;
      try {
        text = decoder.decode(await response.arrayBuffer());
        if (text.includes(String.fromCharCode(0))) {
          throw new Error("binary");
        }
      } catch {
        signal?.throwIfAborted();
        skipped.push({ path: inside, reason: "not a text file" });
        continue;
      }
      if (inside.endsWith(".ipynb")) {
        try {
          text = withoutOutputs(text);
        } catch (error) {
          skipped.push({ path: inside, reason: (error as Error).message });
          continue;
        }
        if (text.length > gitHubLimits.fileBytes) {
          skipped.push({
            path: inside,
            reason: `bigger than ${gitHubLimits.fileBytes / 1e6} MB without its outputs`,
          });
          continue;
        }
      }
      texts.set(inside, text);
      done++;
      onProgress?.(`downloading ${name}: ${done} of ${wanted.length} files`);
    }
  };
  await Promise.all(
    Array.from(
      { length: Math.min(parallelDownloads, wanted.length) },
      download,
    ),
  );
  signal?.throwIfAborted();

  // in all at most the limit, the files in their order
  const kept = new Map<string, string>();
  let bytes = 0;
  for (const [inside, text] of [...texts].sort(([a], [b]) =>
    a.localeCompare(b),
  )) {
    const size = new TextEncoder().encode(text).length;
    if (bytes + size > gitHubLimits.totalBytes) {
      skipped.push({
        path: inside,
        reason: `over ${gitHubLimits.totalBytes / 1e6} MB in all`,
      });
      continue;
    }
    kept.set(inside, text);
    bytes += size;
  }
  if (kept.size === 0) {
    throw new Error("the folder has no text or code files to make a skill of");
  }

  // the skill folders: those with SKILL.md, or all of it with a minimal one
  let madeSkillMd = false;
  let folderOf = (inside: string) => inside;
  const ownSkillMd = kept.get("SKILL.md");
  if (ownSkillMd !== undefined) {
    // the skill's folder is named like the skill, as the format wants
    let named = skillName(path.split("/").at(-1) || repo);
    try {
      const { fields } = parseFrontmatter(ownSkillMd);
      const name = fields.get("name");
      if (typeof name === "string" && name !== "") {
        named = name;
      }
    } catch {
      // readSkills says what is wrong with it
    }
    folderOf = (inside) => `${named}/${inside}`;
  } else if (![...kept.keys()].some((inside) => inside.endsWith("/SKILL.md"))) {
    const named = skillName(path.split("/").at(-1) || repo);
    kept.set("SKILL.md", minimalSkillMd(named, origin, info.description ?? ""));
    madeSkillMd = true;
    folderOf = (inside) => `${named}/${inside}`;
  } else {
    folderOf = (inside) => `${skillName(repo)}/${inside}`;
  }
  const files = [...kept].map(([inside, text]) => ({
    path: folderOf(inside),
    content: new Blob([text], { type: "text/plain" }),
  }));
  const { skills, problems } = await readSkills(files, { signal });

  onProgress?.("checking the packages of Pyodide");
  let packages: PyodidePackages | Error;
  try {
    packages = await options.pyodide();
  } catch (error) {
    signal?.throwIfAborted();
    packages = error instanceof Error ? error : new Error(String(error));
  }
  return {
    origin,
    url: `https://github.com/${fullName}${path === "" ? "" : `/tree/${commit}/${path}`}`,
    description: info.description ?? "",
    skills,
    problems,
    files,
    bytes,
    skipped,
    madeSkillMd,
    topFiles: topLicenses
      .map((entry) => entry.path)
      .filter((name) => kept.has(name)),
    python: analyzePython(kept, packages),
  };
}
