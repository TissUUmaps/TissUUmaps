import { describe, expect, it } from "vitest";

import {
  type FolderFile,
  createSkill,
  parseFrontmatter,
  readSkills,
} from "./skills";

/** A SKILL.md with the given frontmatter lines and body */
function skillMd(frontmatter: string[], body = "Do this, then that."): string {
  return ["---", ...frontmatter, "---", "", body].join("\n");
}

/** The files of a folder, from their paths and texts */
function folder(files: Record<string, string>): FolderFile[] {
  return Object.entries(files).map(([path, text]) => ({
    path,
    content: new Blob([text]),
  }));
}

describe("parseFrontmatter", () => {
  it("reads the examples of the Agent Skills specification", () => {
    const { fields, body } = parseFrontmatter(
      [
        "---",
        "name: pdf-processing",
        "description: Extract PDF text, fill forms, merge files. Use when handling PDFs.",
        "license: Apache-2.0",
        "metadata:",
        "  author: example-org",
        '  version: "1.0"',
        "allowed-tools: Bash(git:*) Bash(jq:*) Read",
        "---",
        "",
        "# PDF processing",
        "",
        "Run the extraction script:",
        "scripts/extract.py",
      ].join("\n"),
    );
    expect(Object.fromEntries(fields)).toEqual({
      name: "pdf-processing",
      description:
        "Extract PDF text, fill forms, merge files. Use when handling PDFs.",
      license: "Apache-2.0",
      metadata: { author: "example-org", version: "1.0" },
      "allowed-tools": "Bash(git:*) Bash(jq:*) Read",
    });
    expect(body).toBe(
      "# PDF processing\n\nRun the extraction script:\nscripts/extract.py",
    );
  });

  it("reads values over several lines, quotes and comments", () => {
    const { fields } = parseFrontmatter(
      skillMd([
        "# a comment",
        "folded: >",
        "  Computes statistics",
        "  on the columns.",
        "",
        "  Use it for tests.",
        "literal: |-",
        "  line one",
        "  line two",
        "plain: Computes statistics",
        "  on the columns # not part of it",
        'double: "say \\"hi\\"\\n\\u00e9" # a comment',
        "single: 'it''s # not a comment'",
        'quoted: "one',
        '  two"',
        "list:",
        "  - Read",
        "  - 'Bash(git:*)'",
        "empty:",
      ]),
    );
    expect(Object.fromEntries(fields)).toEqual({
      folded: "Computes statistics on the columns.\nUse it for tests.\n",
      literal: "line one\nline two",
      plain: "Computes statistics on the columns",
      double: 'say "hi"\né',
      single: "it's # not a comment",
      quoted: "one two",
      list: ["Read", "Bash(git:*)"],
      empty: "",
    });
  });

  it("accepts Windows line ends and a byte order mark", () => {
    const { fields, body } = parseFrontmatter(
      `${String.fromCharCode(0xfeff)}---\r\nname: a\r\ndescription: b\r\n---\r\nBody\r\n`,
    );
    expect(fields.get("name")).toBe("a");
    expect(body).toBe("Body");
  });

  it.each([
    ["no frontmatter", "# Title\n\nText", 'must start with a "---" line'],
    ["an open frontmatter", "---\nname: a\n", 'must end with a "---" line'],
    [
      "a line that is not a field",
      "---\njust text\n---\n",
      'line 2: expected "key: value"',
    ],
    [
      "a field given twice",
      "---\nname: a\nname: b\n---\n",
      'line 3: "name" is given twice',
    ],
    [
      "an open quote",
      '---\nname: "a\n---\n',
      'line 2: the quote " is not closed',
    ],
    [
      "text after a quote",
      "---\nname: 'a' b\n---\n",
      "line 2: unexpected text",
    ],
    [
      "a deeper level",
      "---\nmetadata:\n  a: 1\n    b: 2\n---\n",
      "only one level",
    ],
  ])("refuses %s", (_, text, error) => {
    expect(() => parseFrontmatter(text)).toThrow(error);
  });
});

describe("createSkill", () => {
  it("reads the fields and the instructions", () => {
    const files = new Map([["scripts/stats.js", new Blob(["// code"])]]);
    const text = skillMd([
      "name: data-statistics",
      "description: >",
      "  Statistics on table columns.",
      "  Use when the user asks for tests.",
      "license: MIT",
      "compatibility: Needs JavaScript",
      "metadata:",
      "  version: 1.0",
    ]);
    const skill = createSkill("data-statistics", text, files);
    expect(skill).toEqual({
      name: "data-statistics",
      description:
        "Statistics on table columns. Use when the user asks for tests.",
      license: "MIT",
      compatibility: "Needs JavaScript",
      metadata: { version: "1.0" },
      instructions: "Do this, then that.",
      skillMd: text,
      files,
      warnings: [],
    });
  });

  it("ignores unknown fields with a warning, and allowed-tools without", () => {
    const skill = createSkill(
      "a",
      skillMd([
        "name: a",
        "description: b",
        "allowed-tools: Read",
        "user-invocable: true",
        "model: some-model",
      ]),
      new Map(),
    );
    expect(skill.warnings).toEqual([
      "ignored, not part of the Agent Skills format: user-invocable, model",
    ]);
    expect(skill).not.toHaveProperty("license");
    expect(skill.metadata).toEqual({});
  });

  it("accepts lowercase letters of any alphabet, and an empty metadata", () => {
    const skill = createSkill(
      "données-2",
      skillMd(["name: données-2", "description: b", "metadata:"]),
      new Map(),
    );
    expect(skill.name).toBe("données-2");
  });

  it.each([
    ["PDF-Processing", "must be lowercase"],
    ["-pdf", "must not start or end with a hyphen"],
    ["pdf-", "must not start or end with a hyphen"],
    ["pdf--processing", "must not have two hyphens in a row"],
    ["pdf_processing", "may only have letters, digits and hyphens"],
    ["a".repeat(65), "longer than 64 characters"],
  ])("refuses the name %s", (name, error) => {
    expect(() =>
      createSkill(
        name,
        skillMd([`name: ${name}`, "description: b"]),
        new Map(),
      ),
    ).toThrow(error);
  });

  it.each([
    [["description: b"], "other", 'the "name" is missing'],
    [["name: a"], "a", 'the "description" is missing'],
    [["name: a", "description: '  '"], "a", 'the "description" is missing'],
    [["name: a", "description: b"], "b", 'must be the name of its folder, "b"'],
    [["name: a", `description: ${"d".repeat(1025)}`], "a", "longer than 1024"],
    [
      ["name: a", "description: b", `compatibility: ${"c".repeat(501)}`],
      "a",
      "longer than 500",
    ],
    [["name: a", "description:", "  - b"], "a", '"description" must be text'],
    [
      ["name: a", "description: b", "metadata: 1.0"],
      "a",
      '"metadata" must be "key: value" lines',
    ],
  ])("refuses %j in folder %s", (frontmatter, folderName, error) => {
    expect(() =>
      createSkill(folderName, skillMd(frontmatter), new Map()),
    ).toThrow(error);
  });

  it("names every problem at once", () => {
    expect(() => createSkill("b", skillMd(["name: A"]), new Map())).toThrow(
      'the name "A" must be lowercase; the name "A" must be the name of its folder, "b"; the "description" is missing',
    );
  });
});

describe("readSkills", () => {
  it("reads a chosen skill folder, with its files", async () => {
    const { skills, problems } = await readSkills(
      folder({
        "data-statistics/SKILL.md": skillMd([
          "name: data-statistics",
          "description: d",
        ]),
        "data-statistics/scripts/stats.js": "// code",
        "data-statistics/references/TESTS.md": "# Tests",
        "data-statistics/.DS_Store": "",
      }),
    );
    expect(problems).toEqual([]);
    expect(skills.map((skill) => skill.name)).toEqual(["data-statistics"]);
    expect([...skills[0]!.files.keys()]).toEqual([
      "scripts/stats.js",
      "references/TESTS.md",
    ]);
    expect(await skills[0]!.files.get("scripts/stats.js")!.text()).toBe(
      "// code",
    );
  });

  it("reads a folder of skills, by name, and keeps going after a problem", async () => {
    const { skills, problems } = await readSkills(
      folder({
        "skills\\zeta\\SKILL.md": skillMd(["name: zeta", "description: z"]),
        "skills/alpha/SKILL.md": skillMd(["name: alpha", "description: a"]),
        "skills/broken/SKILL.md": "no frontmatter",
        "skills/README.md": "not in a skill",
        "skills/.hidden/SKILL.md": skillMd(["name: .hidden", "description: h"]),
      }),
    );
    expect(skills.map((skill) => skill.name)).toEqual(["alpha", "zeta"]);
    expect(skills.every((skill) => skill.files.size === 0)).toBe(true);
    expect(problems).toEqual([
      {
        folder: "skills/broken",
        error:
          'SKILL.md must start with a "---" line, then the name and the description',
      },
    ]);
  });

  it("gives a skill inside another skill its own files", async () => {
    const { skills } = await readSkills(
      folder({
        "outer/SKILL.md": skillMd(["name: outer", "description: o"]),
        "outer/notes.md": "outer notes",
        "outer/inner/SKILL.md": skillMd(["name: inner", "description: i"]),
        "outer/inner/notes.md": "inner notes",
      }),
    );
    expect(
      skills.map((skill) => [skill.name, [...skill.files.keys()]]),
    ).toEqual([
      ["inner", ["notes.md"]],
      ["outer", ["notes.md"]],
    ]);
  });

  it("refuses two skills with the same name, and a SKILL.md without a folder", async () => {
    const { skills, problems } = await readSkills(
      folder({
        "SKILL.md": skillMd(["name: a", "description: a"]),
        "one/same/SKILL.md": skillMd(["name: same", "description: 1"]),
        "two/same/SKILL.md": skillMd(["name: same", "description: 2"]),
      }),
    );
    expect(skills.map((skill) => skill.description)).toEqual(["1"]);
    expect(problems).toEqual([
      {
        folder: "",
        error: "SKILL.md must be in a folder named like the skill",
      },
      { folder: "two/same", error: 'another skill is already named "same"' },
    ]);
  });

  it("stops when aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      readSkills(folder({}), { signal: controller.signal }),
    ).rejects.toThrow();
  });
});
