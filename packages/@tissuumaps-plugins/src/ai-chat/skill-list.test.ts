// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";

import { makeSkillList } from "./skill-list";
import type { SkillStorage, StoredSkill } from "./skill-storage";
import { createSkill } from "./skills";

/** A SKILL.md with a name and a description */
function skillMd(name: string, description = `What ${name} does.`): string {
  return `---\nname: ${name}\ndescription: ${description}\n---\n\nSteps.`;
}

/** A storage that keeps the skills in memory, like IndexedDB would */
function fakeStorage(remembered: StoredSkill[] = []) {
  const skills = new Map(remembered.map((skill) => [skill.name, skill]));
  const storage: SkillStorage = {
    load: vi.fn(() => Promise.resolve([...skills.values()])),
    save: vi.fn((skill: StoredSkill) => {
      skills.set(skill.name, skill);
      return Promise.resolve();
    }),
    remove: vi.fn((name: string) => {
      skills.delete(name);
      return Promise.resolve();
    }),
  };
  return { storage, skills };
}

/** Waits until the reads and the storage have answered */
function settle(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/** Creates the section, and waits for the remembered skills */
async function mountList(storage: SkillStorage) {
  const onChange = vi.fn();
  const list = makeSkillList(storage, onChange);
  document.body.replaceChildren(list.element);
  await settle();
  return { list, onChange, element: list.element };
}

/** Chooses a folder with "Add skill…", as the browser gives it */
async function chooseFolder(
  element: HTMLElement,
  files: Record<string, string>,
) {
  const input = element.querySelector<HTMLInputElement>("input[type=file]")!;
  const chosen = Object.entries(files).map(([path, text]) => {
    const file = new File([text], path.split("/").at(-1)!);
    Object.defineProperty(file, "webkitRelativePath", { value: path });
    return file;
  });
  Object.defineProperty(input, "files", { value: chosen, configurable: true });
  input.dispatchEvent(new Event("change"));
  await settle();
}

/** The checkbox with the given label */
function checkbox(element: HTMLElement, label: string, index = 0) {
  const labels = [...element.querySelectorAll("label")].filter(
    (candidate) => candidate.textContent === label,
  );
  return labels[index]!.querySelector("input")!;
}

/** Clicks a checkbox, and waits for the storage */
async function click(box: HTMLInputElement) {
  box.click();
  await settle();
}

/** The text of the line under the list */
function status(element: HTMLElement): string {
  return element.querySelector("[role=status]")!.textContent;
}

describe("makeSkillList", () => {
  it("says what a skill is while there are none", async () => {
    const { element, list } = await mountList(fakeStorage().storage);
    expect(element.textContent).toContain("No skills yet");
    expect(list.activeSkills()).toEqual([]);
    expect(checkbox(element, "Remember this skill").checked).toBe(true);
    expect(
      element.querySelector<HTMLInputElement>("input[type=file]")!
        .webkitdirectory,
    ).toBe(true);
  });

  it("adds the skills of a chosen folder, on and remembered", async () => {
    const { storage, skills } = fakeStorage();
    const { element, list, onChange } = await mountList(storage);
    await chooseFolder(element, {
      "skills/zeta/SKILL.md": skillMd("zeta"),
      "skills/zeta/scripts/run.js": "// code",
      "skills/alpha/SKILL.md": skillMd(
        "alpha",
        "Says hello. Use for greetings.",
      ),
    });
    expect(status(element)).toBe("Added: alpha, zeta.");
    expect(list.activeSkills().map((skill) => skill.name)).toEqual([
      "alpha",
      "zeta",
    ]);
    expect(onChange).toHaveBeenCalled();
    expect(element.textContent).toContain("Says hello. Use for greetings.");
    expect(checkbox(element, "alpha").checked).toBe(true);
    expect(checkbox(element, "Remember", 0).checked).toBe(true);

    const zeta = skills.get("zeta")!;
    expect(zeta.enabled).toBe(true);
    expect(zeta.files.map((file) => file.path)).toEqual([
      "zeta/SKILL.md",
      "zeta/scripts/run.js",
    ]);
    expect(await zeta.files[1]!.content.text()).toBe("// code");
  });

  it("keeps a skill for this session only, and remembers it later", async () => {
    const { storage, skills } = fakeStorage();
    const { element } = await mountList(storage);
    await click(checkbox(element, "Remember this skill"));
    await chooseFolder(element, { "hello/SKILL.md": skillMd("hello") });
    expect(skills.size).toBe(0);
    expect(checkbox(element, "Remember").checked).toBe(false);

    await click(checkbox(element, "Remember"));
    expect(skills.has("hello")).toBe(true);
    expect(status(element)).toBe("hello is remembered by this browser.");

    await click(checkbox(element, "Remember"));
    expect(skills.has("hello")).toBe(false);
    expect(status(element)).toBe("hello is kept for this session only.");
  });

  it("turns a skill off and on, also in the browser", async () => {
    const { storage, skills } = fakeStorage();
    const { element, list, onChange } = await mountList(storage);
    await chooseFolder(element, { "hello/SKILL.md": skillMd("hello") });
    onChange.mockClear();

    await click(checkbox(element, "hello"));
    expect(list.activeSkills()).toEqual([]);
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(skills.get("hello")!.enabled).toBe(false);

    await click(checkbox(element, "hello"));
    expect(list.activeSkills().map((skill) => skill.name)).toEqual(["hello"]);
    expect(skills.get("hello")!.enabled).toBe(true);
  });

  it("removes a skill, also from the browser", async () => {
    const { storage, skills } = fakeStorage();
    const { element, list } = await mountList(storage);
    await chooseFolder(element, { "hello/SKILL.md": skillMd("hello") });
    element
      .querySelector<HTMLButtonElement>("[aria-label='Remove hello']")!
      .click();
    await settle();
    expect(list.activeSkills()).toEqual([]);
    expect(skills.size).toBe(0);
    expect(status(element)).toBe("hello removed.");
    expect(element.textContent).toContain("No skills yet");
  });

  it("starts with the skills the browser remembers, on or off", async () => {
    const { storage } = fakeStorage([
      {
        name: "on",
        enabled: true,
        files: [{ path: "on/SKILL.md", content: new Blob([skillMd("on")]) }],
      },
      {
        name: "off",
        enabled: false,
        files: [{ path: "off/SKILL.md", content: new Blob([skillMd("off")]) }],
      },
    ]);
    const { element, list, onChange } = await mountList(storage);
    expect(list.activeSkills().map((skill) => skill.name)).toEqual(["on"]);
    expect(checkbox(element, "off").checked).toBe(false);
    expect(checkbox(element, "Remember", 0).checked).toBe(true);
    expect(onChange).toHaveBeenCalled();
  });

  it("replaces a skill with the same name, also a remembered one", async () => {
    const { storage, skills } = fakeStorage();
    const { element, list } = await mountList(storage);
    await chooseFolder(element, { "hello/SKILL.md": skillMd("hello", "Old.") });
    await click(checkbox(element, "Remember this skill"));
    await chooseFolder(element, { "hello/SKILL.md": skillMd("hello", "New.") });
    expect(status(element)).toBe(
      "Added: hello (replacing the one with the same name).",
    );
    expect(list.activeSkills().map((skill) => skill.description)).toEqual([
      "New.",
    ]);
    expect(skills.size).toBe(0);
  });

  it("says which folders are not skills", async () => {
    const { element, list } = await mountList(fakeStorage().storage);
    await chooseFolder(element, {
      "skills/Bad/SKILL.md": skillMd("Bad"),
      "skills/good/SKILL.md": skillMd("good"),
    });
    expect(status(element)).toBe(
      'Added: good.\nNot added: skills/Bad: the name "Bad" must be lowercase.',
    );
    expect(list.activeSkills()).toHaveLength(1);

    await chooseFolder(element, { "notes/README.md": "# Notes" });
    expect(status(element)).toBe(
      "No skill found: a skill is a folder with a SKILL.md file in it.",
    );
  });

  it("shows the warnings of a skill", async () => {
    const { element } = await mountList(fakeStorage().storage);
    await chooseFolder(element, {
      "hello/SKILL.md": "---\nname: hello\ndescription: d\nmodel: x\n---\n",
    });
    expect(element.textContent).toContain(
      "hello: ignored, not part of the Agent Skills format: model",
    );
  });

  it("keeps a skill for this session when the browser cannot remember it", async () => {
    const { storage } = fakeStorage();
    vi.mocked(storage.save).mockRejectedValue(
      new Error("this browser cannot remember skills (it has no IndexedDB)"),
    );
    const { element, list } = await mountList(storage);
    await chooseFolder(element, { "hello/SKILL.md": skillMd("hello") });
    expect(list.activeSkills()).toHaveLength(1);
    expect(checkbox(element, "Remember").checked).toBe(false);
    expect(status(element)).toBe(
      "Added: hello.\nhello is kept for this session only: this browser cannot remember skills (it has no IndexedDB)",
    );
  });

  it("says when the browser cannot forget a removed skill", async () => {
    const { storage } = fakeStorage();
    vi.mocked(storage.remove).mockRejectedValue(new Error("blocked"));
    const { element, list } = await mountList(storage);
    await chooseFolder(element, { "hello/SKILL.md": skillMd("hello") });
    element
      .querySelector<HTMLButtonElement>("[aria-label='Remove hello']")!
      .click();
    await settle();
    expect(list.activeSkills()).toEqual([]);
    expect(status(element)).toBe("The browser could not forget hello: blocked");
  });

  it("says when the remembered skills cannot be read", async () => {
    const { storage } = fakeStorage();
    vi.mocked(storage.load).mockRejectedValue(new Error("blocked"));
    const { element } = await mountList(storage);
    expect(status(element)).toBe(
      "The remembered skills cannot be read: blocked.",
    );
  });

  describe("the skills of the project", () => {
    /** The skills of a project, from SKILL.md files */
    function projectSkills(...names: string[]) {
      return {
        skills: names.map((name) =>
          createSkill(
            name,
            skillMd(name, `${name} of the project.`),
            new Map(),
          ),
        ),
        problems: [],
        where: "/xenium/skills",
      };
    }

    /** The tags of the rows of the list, e.g. "hello project" */
    function rows(element: HTMLElement): string[] {
      return [...element.querySelectorAll("label")]
        .filter((label) => !label.textContent.startsWith("Remember"))
        .map(
          (label) => `${label.textContent} ${label.nextSibling?.textContent}`,
        );
    }

    it("shows them apart from those added by hand, without Remember and Remove", async () => {
      const { element, list, onChange } = await mountList(
        fakeStorage().storage,
      );
      await chooseFolder(element, { "zeta/SKILL.md": skillMd("zeta") });
      onChange.mockClear();
      list.setProjectSkills(projectSkills("lab-colors"));
      expect(status(element)).toBe("From the project: lab-colors.");
      expect(rows(element)).toEqual(["lab-colors project", "zeta added"]);
      expect(
        element.querySelectorAll("button[aria-label^=Remove]"),
      ).toHaveLength(1);
      expect(
        [...element.querySelectorAll("label")].filter(
          (label) => label.textContent === "Remember",
        ),
      ).toHaveLength(1);
      expect(list.activeSkills().map((skill) => skill.name)).toEqual([
        "lab-colors",
        "zeta",
      ]);
      expect(onChange).toHaveBeenCalled();

      await click(checkbox(element, "lab-colors"));
      expect(list.activeSkills().map((skill) => skill.name)).toEqual(["zeta"]);
    });

    it("are hidden by a skill added by hand with the same name, while it is on", async () => {
      const { element, list } = await mountList(fakeStorage().storage);
      list.setProjectSkills(projectSkills("hello"));
      await chooseFolder(element, {
        "hello/SKILL.md": skillMd("hello", "Mine."),
      });
      expect(rows(element)).toEqual(["hello added", "hello project"]);
      expect(list.activeSkills().map((skill) => skill.description)).toEqual([
        "Mine.",
      ]);
      expect(element.textContent).toContain(
        "hello: not used: a skill added by hand has the same name",
      );

      await click(checkbox(element, "hello", 0));
      expect(list.activeSkills().map((skill) => skill.description)).toEqual([
        "hello of the project.",
      ]);
    });

    it("change with the project, and say why they cannot be read", async () => {
      const { element, list } = await mountList(fakeStorage().storage);
      list.setProjectSkills({
        ...projectSkills("hello"),
        problems: [
          { folder: "Bad", error: 'the name "Bad" must be lowercase' },
        ],
      });
      expect(status(element)).toBe(
        'From the project: hello.\nNot added from the project: Bad: the name "Bad" must be lowercase.',
      );
      list.setProjectSkills(null);
      expect(list.activeSkills()).toEqual([]);
      expect(element.textContent).toContain("No skills yet");

      list.setProjectSkills(new Error("permission denied"));
      expect(status(element)).toBe(
        "The skills of the project cannot be read: permission denied.",
      );
    });
  });

  it("opens the whole description with a click, and closes it again", async () => {
    const { element } = await mountList(fakeStorage().storage);
    await chooseFolder(element, {
      "hello/SKILL.md": skillMd("hello", "A very long description."),
      "zeta/SKILL.md": skillMd("zeta"),
    });
    const description = () =>
      [...element.querySelectorAll<HTMLElement>("[role=button]")].find(
        (candidate) => candidate.textContent === "A very long description.",
      )!;
    expect(description().style.whiteSpace).toBe("nowrap");
    expect(description().getAttribute("aria-expanded")).toBe("false");

    description().click();
    expect(description().style.whiteSpace).toBe("normal");
    expect(description().getAttribute("aria-expanded")).toBe("true");
    // it stays open when the list is shown again
    await click(checkbox(element, "zeta"));
    expect(description().getAttribute("aria-expanded")).toBe("true");

    description().dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    expect(description().style.whiteSpace).toBe("nowrap");
  });
});
