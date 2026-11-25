import { runRequest } from "./browser-storage";
import type { GitHubOrigin } from "./github-skills";
import type { FolderFile } from "./skills";

// The skills added by hand with "Remember this skill" are kept in IndexedDB
// (see browser-storage.ts), which can hold the files of a skill as they are.
// They are kept as files, not as read skills, so that they are read again with the
// rules of the plugin that opens them.

/** A skill as the browser remembers it: its files, and whether it is on */
export type StoredSkill = {
  /** The name of the skill */
  name: string;
  /** Whether the skill is on */
  enabled: boolean;
  /** The files, by path from the skill's folder, e.g. "stats/SKILL.md" */
  files: FolderFile[];
  /** Where a skill from GitHub comes from */
  github?: GitHubOrigin;
};

/** Where the panel remembers the skills added by hand */
export type SkillStorage = {
  /** Reads the remembered skills */
  load: typeof loadStoredSkills;
  /** Remembers a skill, or updates it */
  save: typeof storeSkill;
  /** Forgets a skill */
  remove: typeof forgetStoredSkill;
};

/**
 * Reads the skills the browser remembers
 *
 * @param options - Optional abort signal
 * @returns The skills; none if the browser has no IndexedDB, since then it
 * could never remember one
 * @throws If the database cannot be read
 */
export function loadStoredSkills(options?: {
  signal?: AbortSignal;
}): Promise<StoredSkill[]> {
  if (typeof indexedDB === "undefined") {
    return Promise.resolve([]);
  }
  return runRequest(
    "skills",
    "readonly",
    (store) => store.getAll() as IDBRequest<StoredSkill[]>,
    options,
  );
}

/**
 * Remembers a skill in the browser, or updates it (e.g. when it is turned off)
 *
 * @param skill - The skill
 * @param options - Optional abort signal
 * @throws If the browser cannot remember it
 */
export async function storeSkill(
  skill: StoredSkill,
  options?: { signal?: AbortSignal },
): Promise<void> {
  await runRequest("skills", "readwrite", (store) => store.put(skill), options);
}

/**
 * Makes the browser forget a skill
 *
 * @param name - The name of the skill
 * @param options - Optional abort signal
 * @throws If the browser storage cannot be changed
 */
export async function forgetStoredSkill(
  name: string,
  options?: { signal?: AbortSignal },
): Promise<void> {
  await runRequest(
    "skills",
    "readwrite",
    (store) => store.delete(name),
    options,
  );
}
