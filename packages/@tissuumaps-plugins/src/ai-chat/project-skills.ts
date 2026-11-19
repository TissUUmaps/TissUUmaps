import { SourceUtils } from "@tissuumaps/core";

import {
  type FolderFile,
  type Skill,
  type SkillProblem,
  readSkills,
} from "./skills";

// A project can bring its own skills, in a folder "skills" next to its
// project file, with one folder per skill. In the connected folder (the
// workspace), the plugin lists that folder. A web server cannot list a
// folder, so a project opened from a URL lists the files of its skills in
// skills/index.json, e.g. {"files": ["lab-colors/SKILL.md", ...]}.

/** The skills of the open project */
export type ProjectSkills = {
  skills: Skill[];
  /** The skill folders that could not be read, and why */
  problems: SkillProblem[];
  /** Where the skills come from, e.g. "/xenium/skills" or a URL */
  where: string;
};

/**
 * A directory whose entries can be listed: the File System API can, but the
 * DOM types of TypeScript do not declare it yet
 */
type ListableDirectory = FileSystemDirectoryHandle & {
  values(): AsyncIterable<FileSystemDirectoryHandle | FileSystemFileHandle>;
};

/** The folder next to the project file that holds the skills of the project */
const skillsFolder = "skills";

/** The most files read from a skills folder, so that a wrong folder cannot fill the memory */
const maxFiles = 1000;

/**
 * Lists the files of a folder of the workspace, and of its folders; hidden
 * ones, e.g. ".git", are left out
 *
 * @param directory - The folder
 * @param path - Its path from the skills folder, "" for the skills folder
 * @param files - The list the files are added to, by path from the skills folder
 * @param options - Optional abort signal
 * @throws If there are too many files
 */
async function listFolder(
  directory: FileSystemDirectoryHandle,
  path: string,
  files: FolderFile[],
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  for await (const entry of (directory as ListableDirectory).values()) {
    // the iteration and getFile() take no abort signal
    signal?.throwIfAborted();
    if (entry.name.startsWith(".")) {
      continue;
    }
    const entryPath = path === "" ? entry.name : `${path}/${entry.name}`;
    if (entry.kind === "directory") {
      await listFolder(entry, entryPath, files, options);
    } else {
      if (files.length >= maxFiles) {
        throw new Error(`the skills folder has more than ${maxFiles} files`);
      }
      files.push({ path: entryPath, content: await entry.getFile() });
    }
  }
}

/**
 * Whether a path of skills/index.json stays inside the skills folder
 *
 * @param path - The path
 * @returns Whether it is a plain relative path, without ".." or a URL
 */
function isInsideFolder(path: string): boolean {
  const segments = path.split("/");
  return (
    path !== "" &&
    !path.startsWith("/") &&
    !segments[0]!.includes(":") &&
    segments.every((segment) => segment !== "" && segment !== "..")
  );
}

/**
 * Downloads the files that skills/index.json lists, next to a project opened
 * from a URL
 *
 * @param projectSource - The URL of the project
 * @param problems - The list the files that cannot be downloaded are added to
 * @param options - Optional abort signal
 * @returns The files, by path from the skills folder, or `null` if the
 * project has no skills/index.json
 * @throws If skills/index.json cannot be read
 */
async function fetchListedFiles(
  projectSource: string,
  problems: SkillProblem[],
  options?: { signal?: AbortSignal },
): Promise<FolderFile[] | null> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const indexUrl = new URL(`${skillsFolder}/index.json`, projectSource);
  const response = await fetch(indexUrl, { signal });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(
      `skills/index.json cannot be read (${response.status} ${response.statusText})`,
    );
  }
  let index: unknown;
  try {
    index = await response.json();
  } catch {
    signal?.throwIfAborted();
    throw new Error("skills/index.json is not valid JSON");
  }
  const paths = (index as { files?: unknown } | null)?.files;
  if (
    !Array.isArray(paths) ||
    !paths.every((path) => typeof path === "string")
  ) {
    throw new Error(
      'skills/index.json must be {"files": [...]}, with the paths of the files of the skills, e.g. "lab-colors/SKILL.md"',
    );
  }
  if (paths.length > maxFiles) {
    throw new Error(`skills/index.json lists more than ${maxFiles} files`);
  }
  // downloaded together; each gives its file or its problem, so that the
  // problems are listed in the order of skills/index.json
  const results = await Promise.all(
    paths.map(async (path): Promise<FolderFile | SkillProblem> => {
      const folder = path.split("/")[0]!;
      if (!isInsideFolder(path)) {
        return {
          folder,
          error: `"${path}" in skills/index.json is not a path inside the skills folder`,
        };
      }
      const fileUrl = new URL(`${skillsFolder}/${path}`, projectSource);
      try {
        const file = await fetch(fileUrl, { signal });
        if (!file.ok) {
          throw new Error(`${file.status} ${file.statusText}`);
        }
        return { path, content: await file.blob() };
      } catch (error) {
        signal?.throwIfAborted();
        return {
          folder,
          error: `${path} cannot be downloaded (${(error as Error).message})`,
        };
      }
    }),
  );
  const files: FolderFile[] = [];
  for (const result of results) {
    if ("error" in result) {
      problems.push(result);
    } else {
      files.push(result);
    }
  }
  return files;
}

/**
 * Reads the skills of the open project: from the folder "skills" next to its
 * project file in the workspace, or from the files that skills/index.json
 * lists next to a project opened from a URL
 *
 * @param projectSource - Where the project was loaded from: its URL, its
 * workspace-relative path, or `null`
 * @param workspace - The open workspace, if any
 * @param options - Optional abort signal
 * @returns The skills, or `null` if the project has none: no skills folder,
 * no skills/index.json, or a project that was loaded from neither
 * @throws If the skills folder or skills/index.json cannot be read
 */
export async function loadProjectSkills(
  projectSource: string | null,
  workspace: FileSystemDirectoryHandle | null,
  options?: { signal?: AbortSignal },
): Promise<ProjectSkills | null> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  if (
    projectSource === null ||
    (SourceUtils.isWorkspacePath(projectSource) && workspace === null)
  ) {
    return null;
  }
  const folder = SourceUtils.normalizeSource(
    skillsFolder,
    workspace,
    projectSource,
  );
  const problems: SkillProblem[] = [];
  let files: FolderFile[] | null;
  if (SourceUtils.isWorkspacePath(folder)) {
    let directory: string | FileSystemDirectoryHandle;
    try {
      directory = await SourceUtils.resolveSourceDirectory(folder, workspace, {
        signal,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "NotFoundError") {
        return null;
      }
      throw error;
    }
    files = [];
    await listFolder(directory as FileSystemDirectoryHandle, "", files, {
      signal,
    });
  } else {
    files = await fetchListedFiles(projectSource, problems, { signal });
    if (files === null) {
      return null;
    }
  }
  const read = await readSkills(files, { signal });
  return {
    skills: read.skills,
    problems: [...problems, ...read.problems],
    where: folder,
  };
}
