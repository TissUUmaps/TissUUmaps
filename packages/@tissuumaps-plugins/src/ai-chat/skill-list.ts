import type { GitHubOrigin } from "./github-skills";
import type { ProjectSkills } from "./project-skills";
import type { SkillStorage, StoredSkill } from "./skill-storage";
import { type FolderFile, type Skill, readSkills } from "./skills";
import { makeButton, quietTextStyle, smallButtonStyle } from "./ui";

/** A skill in the list of the panel */
type SkillEntry = {
  skill: Skill;
  /** Whether it came with the open project, or the user added it by hand */
  origin: "project" | "added";
  /** Whether the skill is on, so that the model can use it */
  enabled: boolean;
  /** Whether the browser remembers it for the next sessions; never for skills of the project */
  remembered: boolean;
  /** Where a skill added from GitHub comes from */
  github?: GitHubOrigin;
};

/** Where a skill comes from */
export type SkillOrigin =
  | { kind: "project"; where: string }
  | { kind: "github"; github: GitHubOrigin }
  | { kind: "added" };

/** The section of the settings with the skills */
export type SkillList = {
  /** The section */
  element: HTMLElement;
  /**
   * The skills that are on; a skill added by hand that is on hides a skill
   * of the project with the same name
   *
   * @returns The skills, by name
   */
  activeSkills: () => Skill[];
  /**
   * Shows the skills of the project that was opened, in place of those of
   * the project before
   *
   * @param project - Its skills, `null` if it has none, or why they could
   * not be read
   */
  setProjectSkills: (project: ProjectSkills | Error | null) => void;
  /**
   * Where a skill of the open project comes from
   *
   * @param skill - The skill
   * @returns e.g. "/xenium/skills", or `undefined` for a skill added by hand
   */
  projectSourceOf: (skill: Skill) => string | undefined;
  /**
   * Where a skill comes from: the open project, GitHub, or the user
   *
   * @param skill - The skill
   * @returns Its origin
   */
  originOf: (skill: Skill) => SkillOrigin;
  /**
   * Adds skills, e.g. from GitHub, in place of those with the same name
   *
   * @param skills - The skills
   * @param options - Where they come from, and whether the browser remembers them
   * @returns What happened, for the user
   */
  addSkills: (
    skills: Skill[],
    options: { github?: GitHubOrigin; remember: boolean },
  ) => Promise<string[]>;
  /**
   * Gives a skill added by hand or from GitHub new instructions: another
   * SKILL.md, with the same name
   *
   * @param skill - The skill
   * @param skillMd - The text of the new SKILL.md
   * @returns The skill with its new instructions
   * @throws If the skill is not one added by hand or from GitHub, or the new
   * SKILL.md is not valid
   */
  replaceSkillMd: (skill: Skill, skillMd: string) => Promise<Skill>;
};

/** The look of the checkboxes */
const checkboxStyle = "accent-color: var(--primary);";

/**
 * The files of a skill, as the browser remembers them
 *
 * @param entry - The skill in the list
 * @returns The skill to remember
 */
function toStored({ skill, enabled, github }: SkillEntry): StoredSkill {
  return {
    name: skill.name,
    enabled,
    ...(github === undefined ? {} : { github }),
    files: [
      { path: `${skill.name}/SKILL.md`, content: new Blob([skill.skillMd]) },
      ...[...skill.files].map(([path, content]) => ({
        path: `${skill.name}/${path}`,
        content,
      })),
    ],
  };
}

/**
 * Creates a checkbox with its label
 *
 * @param text - The label
 * @param checked - Whether it is checked
 * @param onChange - What a change does, with the new state
 * @returns The label, with the checkbox inside
 */
function makeCheckbox(
  text: string,
  checked: boolean,
  onChange: (checked: boolean) => void,
): HTMLLabelElement {
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = checked;
  box.style.cssText = checkboxStyle;
  box.addEventListener("change", () => onChange(box.checked));
  const label = document.createElement("label");
  label.style.cssText = "display: flex; gap: 4px; align-items: center;";
  label.append(box, text);
  return label;
}

/**
 * Creates the description of a skill: one line, which a click (or Enter)
 * opens in full and closes again
 *
 * @param text - The description
 * @param open - Whether it starts open
 * @param onToggle - Called with the new state
 * @returns The element
 */
function makeDescription(
  text: string,
  open: boolean,
  onToggle: (open: boolean) => void,
): HTMLElement {
  const description = document.createElement("div");
  description.textContent = text;
  description.tabIndex = 0;
  description.setAttribute("role", "button");
  description.style.cssText = `${quietTextStyle}cursor: pointer; overflow: hidden; text-overflow: ellipsis;`;
  const show = (isOpen: boolean) => {
    open = isOpen;
    description.style.whiteSpace = open ? "normal" : "nowrap";
    description.setAttribute("aria-expanded", String(open));
    description.title = open
      ? "Click to show one line"
      : "Click to show the whole description";
  };
  const toggle = () => {
    show(!open);
    onToggle(open);
  };
  description.addEventListener("click", toggle);
  description.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggle();
    }
  });
  show(open);
  return description;
}

/**
 * Creates the section of the settings with the skills: those of the open
 * project, and those added by hand. "Add skill…" reads a chosen folder, with
 * one skill or a folder of skills. Each skill can be turned off and on; a
 * skill added by hand can also be remembered by the browser or not, and
 * removed.
 *
 * @param storage - Where the skills added by hand are remembered
 * @param onChange - Called when the skills that are on change
 * @param options - What "Add skill from GitHub…" does, and an optional abort
 * signal, for the reads
 * @returns The section, the skills that are on, and a way to show the skills
 * of the project
 */
export function makeSkillList(
  storage: SkillStorage,
  onChange: () => void,
  options?: { addFromGitHub?: () => void; signal?: AbortSignal },
): SkillList {
  const { signal, addFromGitHub } = options ?? {};
  /** The skills added by hand */
  const entries: SkillEntry[] = [];
  /** The skills of the open project */
  const projectEntries: SkillEntry[] = [];
  /** Where the skills of the open project come from */
  let projectWhere = "";
  /** The descriptions shown in full, by origin and name */
  const openDescriptions = new Set<string>();

  const title = document.createElement("strong");
  title.textContent = "Skills";
  const folderInput = document.createElement("input");
  folderInput.type = "file";
  folderInput.webkitdirectory = true;
  folderInput.style.display = "none";
  const addButton = makeButton(
    "Add skill…",
    () => folderInput.click(),
    smallButtonStyle,
  );
  addButton.title =
    "Choose the folder of a skill (with its SKILL.md), or a folder of skills";
  const gitHubButton = makeButton(
    "Add skill from GitHub…",
    () => addFromGitHub?.(),
    smallButtonStyle,
  );
  gitHubButton.title =
    "Add the code of a GitHub repository, or of a folder of one, as a skill; you see what it holds before it is added";
  gitHubButton.hidden = addFromGitHub === undefined;
  const rememberLabel = makeCheckbox("Remember this skill", true, () => {});
  rememberLabel.title =
    "Keep the skill in this browser for the next sessions, in every project";
  const rememberBox = rememberLabel.querySelector("input")!;
  const header = document.createElement("div");
  header.style.cssText =
    "display: flex; flex-wrap: wrap; gap: 8px; align-items: center;";
  header.append(title, addButton, gitHubButton, rememberLabel, folderInput);

  const list = document.createElement("div");
  list.style.cssText = "display: flex; flex-direction: column; gap: 6px;";
  const status = document.createElement("div");
  status.setAttribute("role", "status");
  status.style.cssText = quietTextStyle;

  const element = document.createElement("div");
  element.style.cssText = "display: flex; flex-direction: column; gap: 6px;";
  element.append(header, list, status);

  /**
   * Runs a task started by the user or by the panel, without waiting for it;
   * a task stopped because the panel closed is not an error
   *
   * @param task - The task
   */
  function runSafely(task: Promise<unknown>): void {
    task.catch((error: unknown) => {
      if (!signal?.aborted) {
        console.error(error);
      }
    });
  }

  /**
   * Says what happened, under the list
   *
   * @param lines - What happened, one line each
   */
  function report(lines: string[]): void {
    status.textContent = lines.join("\n");
    status.style.whiteSpace = "pre-line";
  }

  /**
   * Remembers or forgets a skill in the browser; a skill that cannot be
   * remembered stays for this session only
   *
   * @param entry - The skill
   * @returns What went wrong, or "" if nothing
   */
  async function store(entry: SkillEntry): Promise<string> {
    try {
      if (entry.remembered) {
        await storage.save(toStored(entry), { signal });
      } else {
        await storage.remove(entry.skill.name, { signal });
      }
      return "";
    } catch (error) {
      signal?.throwIfAborted();
      const reason = (error as Error).message;
      if (!entry.remembered) {
        return `The browser could not forget ${entry.skill.name}: ${reason}`;
      }
      entry.remembered = false;
      return `${entry.skill.name} is kept for this session only: ${reason}`;
    }
  }

  /**
   * Whether a skill of the project is hidden by a skill added by hand with
   * the same name, which is on
   *
   * @param entry - The skill
   * @returns Whether it is hidden
   */
  function isHidden(entry: SkillEntry): boolean {
    return (
      entry.origin === "project" &&
      entries.some(
        (other) => other.enabled && other.skill.name === entry.skill.name,
      )
    );
  }

  /**
   * Creates the row of a skill in the list
   *
   * @param entry - The skill
   * @returns The row
   */
  function makeRow(entry: SkillEntry): HTMLElement {
    const { skill, origin } = entry;
    const enabledLabel = makeCheckbox(skill.name, entry.enabled, (on) => {
      entry.enabled = on;
      render();
      onChange();
      if (entry.remembered) {
        runSafely(store(entry).then((problem) => problem && report([problem])));
      }
    });
    enabledLabel.title = entry.enabled
      ? "On: the assistant can use this skill"
      : "Off: the assistant does not see this skill";
    enabledLabel.style.fontWeight = "600";
    const tag = document.createElement("span");
    const { github } = entry;
    tag.textContent = github === undefined ? origin : "GitHub";
    tag.title =
      origin === "project"
        ? `It comes with the open project, from ${projectWhere}`
        : github !== undefined
          ? `From github.com/${github.repo}${github.path === "" ? "" : `/${github.path}`}, commit ${github.commit.slice(0, 7)}, ${github.license === undefined ? "without a license" : `license ${github.license}`}`
          : "You added it with Add skill…";
    tag.style.cssText = `${quietTextStyle}border: 1px solid var(--border); border-radius: 6px; padding: 0 4px;`;
    const top = document.createElement("div");
    top.style.cssText =
      "display: flex; flex-wrap: wrap; gap: 8px; align-items: center;";
    top.append(enabledLabel, tag);

    if (origin === "added") {
      const rememberedLabel = makeCheckbox(
        "Remember",
        entry.remembered,
        (on) => {
          entry.remembered = on;
          runSafely(
            store(entry).then((problem) => {
              report([
                problem ||
                  (on
                    ? `${skill.name} is remembered by this browser.`
                    : `${skill.name} is kept for this session only.`),
              ]);
              render();
            }),
          );
        },
      );
      rememberedLabel.style.cssText += quietTextStyle;
      rememberedLabel.title =
        "Keep the skill in this browser for the next sessions, in every project";
      const removeButton = makeButton(
        "Remove",
        () => {
          entries.splice(entries.indexOf(entry), 1);
          render();
          onChange();
          report([`${skill.name} removed.`]);
          if (entry.remembered) {
            entry.remembered = false;
            runSafely(
              store(entry).then((problem) => problem && report([problem])),
            );
          }
        },
        smallButtonStyle,
      );
      removeButton.setAttribute("aria-label", `Remove ${skill.name}`);
      top.append(rememberedLabel, removeButton);
    }

    const key = `${origin}:${skill.name}`;
    const description = makeDescription(
      skill.description,
      openDescriptions.has(key),
      (open) => {
        if (open) {
          openDescriptions.add(key);
        } else {
          openDescriptions.delete(key);
        }
      },
    );
    const row = document.createElement("div");
    row.style.cssText = `display: flex; flex-direction: column; gap: 2px; opacity: ${entry.enabled && !isHidden(entry) ? 1 : 0.6};`;
    row.append(top, description);
    const notes = [
      ...(isHidden(entry)
        ? ["not used: a skill added by hand has the same name"]
        : []),
      ...skill.warnings,
    ];
    for (const note of notes) {
      const line = document.createElement("div");
      line.textContent = `${skill.name}: ${note}`;
      line.style.cssText = quietTextStyle;
      row.append(line);
    }
    return row;
  }

  /** Shows the list of skills again */
  function render(): void {
    const all = [...entries, ...projectEntries].sort(
      (a, b) =>
        a.skill.name.localeCompare(b.skill.name) ||
        (a.origin === "added" ? -1 : 1),
    );
    list.replaceChildren(...all.map(makeRow));
    if (all.length === 0) {
      const empty = document.createElement("div");
      empty.textContent =
        "No skills yet. A skill is a folder with a SKILL.md file: instructions, and maybe scripts, for the assistant.";
      empty.style.cssText = quietTextStyle;
      list.append(empty);
    }
  }

  /**
   * Puts skills added by hand in the list, in place of those with the same
   * name
   *
   * @param added - The skills, with their state
   * @returns The names of the skills that were replaced
   */
  function put(added: SkillEntry[]): string[] {
    const replaced: string[] = [];
    for (const entry of added) {
      const index = entries.findIndex(
        (other) => other.skill.name === entry.skill.name,
      );
      if (index === -1) {
        entries.push(entry);
      } else {
        entries[index] = entry;
        replaced.push(entry.skill.name);
      }
    }
    return replaced;
  }

  /** Reads the skills of the folder chosen with "Add skill…" */
  async function addChosenFolder(): Promise<void> {
    const files: FolderFile[] = Array.from(folderInput.files ?? [], (file) => ({
      // the path from the chosen folder, which is part of it
      path: file.webkitRelativePath || file.name,
      content: file,
    }));
    // the same folder can be chosen again, e.g. after changing it
    folderInput.value = "";
    if (files.length === 0) {
      return;
    }
    const { skills, problems } = await readSkills(files, { signal });
    const added: SkillEntry[] = skills.map((skill) => ({
      skill,
      origin: "added",
      enabled: true,
      remembered: rememberBox.checked,
    }));
    const replaced = put(added);
    render();
    onChange();
    const lines: string[] = [];
    if (added.length > 0) {
      const names = added.map(({ skill }) => skill.name).join(", ");
      lines.push(
        replaced.length > 0
          ? `Added: ${names} (replacing the one with the same name).`
          : `Added: ${names}.`,
      );
    }
    for (const { folder, error } of problems) {
      lines.push(`Not added: ${folder || "the chosen folder"}: ${error}.`);
    }
    if (added.length === 0 && problems.length === 0) {
      lines.push(
        "No skill found: a skill is a folder with a SKILL.md file in it.",
      );
    }
    report(lines);
    // a skill added for this session only replaces a remembered one, also
    // in the browser
    for (const entry of added) {
      const problem = await store(entry);
      if (problem !== "") {
        lines.push(problem);
        report(lines);
        render();
      }
    }
  }

  /** Puts the skills the browser remembers in the list */
  async function loadRemembered(): Promise<void> {
    const lines: string[] = [];
    try {
      const stored = await storage.load({ signal });
      const loaded: SkillEntry[] = [];
      for (const { name, enabled, files, github } of stored) {
        const { skills, problems } = await readSkills(files, { signal });
        loaded.push(
          ...skills.map((skill) => ({
            skill,
            origin: "added" as const,
            enabled,
            remembered: true,
            ...(github === undefined ? {} : { github }),
          })),
        );
        for (const { error } of problems) {
          lines.push(`The remembered skill ${name} cannot be read: ${error}.`);
        }
      }
      // skills added while these were read are kept
      put(
        loaded.filter(
          ({ skill }) =>
            !entries.some((other) => other.skill.name === skill.name),
        ),
      );
    } catch (error) {
      signal?.throwIfAborted();
      lines.push(
        `The remembered skills cannot be read: ${(error as Error).message}.`,
      );
    }
    render();
    onChange();
    if (lines.length > 0) {
      report(lines);
    }
  }

  /**
   * Shows the skills of the project that was opened
   *
   * @param project - Its skills, `null` if it has none, or why they could
   * not be read
   */
  function setProjectSkills(project: ProjectSkills | Error | null): void {
    projectEntries.length = 0;
    projectWhere = "";
    const lines: string[] = [];
    if (project instanceof Error) {
      lines.push(
        `The skills of the project cannot be read: ${project.message}.`,
      );
    } else if (project !== null) {
      projectWhere = project.where;
      projectEntries.push(
        ...project.skills.map((skill) => ({
          skill,
          origin: "project" as const,
          enabled: true,
          remembered: false,
        })),
      );
      if (project.skills.length > 0) {
        lines.push(
          `From the project: ${project.skills.map(({ name }) => name).join(", ")}.`,
        );
      }
      for (const { folder, error } of project.problems) {
        lines.push(`Not added from the project: ${folder}: ${error}.`);
      }
    }
    render();
    onChange();
    if (lines.length > 0) {
      report(lines);
    }
  }

  /**
   * Adds skills, e.g. from GitHub, in place of those with the same name
   *
   * @param skills - The skills
   * @param options - Where they come from, and whether the browser remembers them
   * @returns What happened, for the user
   */
  async function addSkills(
    skills: Skill[],
    { github, remember }: { github?: GitHubOrigin; remember: boolean },
  ): Promise<string[]> {
    const added: SkillEntry[] = skills.map((skill) => ({
      skill,
      origin: "added",
      enabled: true,
      remembered: remember,
      ...(github === undefined ? {} : { github }),
    }));
    const replaced = put(added);
    render();
    onChange();
    const names = skills.map(({ name }) => name).join(", ");
    const lines = [
      replaced.length > 0
        ? `Added: ${names} (replacing the one with the same name).`
        : `Added: ${names}.`,
    ];
    report(lines);
    for (const entry of added) {
      const problem = await store(entry);
      if (problem !== "") {
        lines.push(problem);
        report(lines);
        render();
      }
    }
    return lines;
  }

  /**
   * Gives a skill added by hand or from GitHub another SKILL.md
   *
   * @param skill - The skill
   * @param skillMd - The text of the new SKILL.md
   * @returns The skill with its new instructions
   * @throws If the skill is not one added by hand, or SKILL.md is not valid
   */
  async function replaceSkillMd(skill: Skill, skillMd: string): Promise<Skill> {
    const entry = entries.find((candidate) => candidate.skill === skill);
    if (entry === undefined) {
      throw new Error(
        `the skill "${skill.name}" came with the project: only skills added by hand or from GitHub can get new instructions`,
      );
    }
    const { skills, problems } = await readSkills(
      [
        {
          path: `${skill.name}/SKILL.md`,
          content: new Blob([skillMd], { type: "text/plain" }),
        },
        ...[...skill.files].map(([path, content]) => ({
          path: `${skill.name}/${path}`,
          content,
        })),
      ],
      { signal },
    );
    const [renewed] = skills;
    if (renewed === undefined) {
      throw new Error(problems[0]?.error ?? "the new SKILL.md cannot be read");
    }
    entry.skill = renewed;
    render();
    onChange();
    const problem = entry.remembered ? await store(entry) : "";
    report([problem || `${skill.name} has new instructions.`]);
    return renewed;
  }

  folderInput.addEventListener("change", () => runSafely(addChosenFolder()));
  render();
  runSafely(loadRemembered());

  return {
    element,
    activeSkills: () => {
      const added = entries.filter(({ enabled }) => enabled);
      const fromProject = projectEntries.filter(
        (entry) => entry.enabled && !isHidden(entry),
      );
      return [...added, ...fromProject]
        .map(({ skill }) => skill)
        .sort((a, b) => a.name.localeCompare(b.name));
    },
    setProjectSkills,
    projectSourceOf: (skill) =>
      projectEntries.some((entry) => entry.skill === skill)
        ? projectWhere
        : undefined,
    originOf: (skill) => {
      if (projectEntries.some((entry) => entry.skill === skill)) {
        return { kind: "project", where: projectWhere };
      }
      const github = entries.find((entry) => entry.skill === skill)?.github;
      return github === undefined
        ? { kind: "added" }
        : { kind: "github", github };
    },
    addSkills,
    replaceSkillMd,
  };
}
