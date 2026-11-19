// Skills follow the Agent Skills format (https://agentskills.io/specification):
// a folder with a SKILL.md file, whose YAML frontmatter gives the name and the
// description of the skill and whose Markdown body gives its instructions,
// and any other files, e.g. scripts. The checks are those of the reference
// validator (skills-ref), except that unknown fields of the frontmatter are
// ignored with a warning instead of refusing the skill, since skills written
// for other agents often have some. Only the part of YAML that frontmatter
// uses is read, so that the plugin needs no YAML library.

/** A skill in the Agent Skills format */
export type Skill = {
  /** The name, which is also the name of its folder, e.g. "data-statistics" */
  name: string;
  /** What the skill does and when to use it, which the model sees first */
  description: string;
  /** The license of the skill, if it gives one */
  license?: string;
  /** What the skill needs to work, if it says */
  compatibility?: string;
  /** Other properties, e.g. the author or the version */
  metadata: Record<string, string>;
  /** The instructions: SKILL.md after its frontmatter */
  instructions: string;
  /** The whole text of SKILL.md, e.g. to remember the skill */
  skillMd: string;
  /** The other files, by path inside the skill's folder, e.g. "scripts/stats.js" */
  files: Map<string, Blob>;
  /** What was ignored in SKILL.md, for the panel */
  warnings: string[];
};

/** A file of a chosen folder, by its path, e.g. "skills/data-statistics/SKILL.md" */
export type FolderFile = { path: string; content: Blob };

/** A skill folder that could not be read, and why */
export type SkillProblem = { folder: string; error: string };

/** A value of the frontmatter: text, a list, or one level of key: value lines */
type FrontmatterValue = string | string[] | Record<string, string>;

/** The fields of the Agent Skills format */
const knownFields = [
  "name",
  "description",
  "license",
  "compatibility",
  "metadata",
  "allowed-tools",
];

/** The longest name, description and compatibility the format allows */
const maxLengths = { name: 64, description: 1024, compatibility: 500 };

/** The character that some editors put at the start of a file */
const byteOrderMark = 0xfeff;

/** A line that starts a field, e.g. "name: data-statistics" or "metadata:" */
const fieldLine = /^([\w.-]+):(?:[ \t]+(.*))?$/;

/**
 * Whether a line of YAML has nothing to read
 *
 * @param line - The line
 * @returns Whether it is empty or only a comment
 */
function isBlank(line: string): boolean {
  const text = line.trim();
  return text === "" || text.startsWith("#");
}

/**
 * Removes a comment from the end of a plain value, e.g. "1.0 # the version"
 *
 * @param text - The value
 * @returns The value without the comment
 */
function withoutComment(text: string): string {
  return text.replace(/(^|\s)#.*$/, "").trim();
}

/**
 * Reads a quoted value, which may go on over the next lines
 *
 * @param text - The value, from its opening quote, with its lines joined
 * @param line - The line number, for the error
 * @returns The value
 * @throws If the quote is not closed, or something follows it
 */
function readQuoted(text: string, line: number): string {
  const quote = text[0];
  let value = "";
  let index = 1;
  for (; index < text.length; index++) {
    const char = text[index]!;
    if (quote === "'" && char === "'") {
      // '' is a quote inside single quotes
      if (text[index + 1] === "'") {
        value += "'";
        index++;
        continue;
      }
      break;
    }
    if (quote === '"' && char === '"') {
      break;
    }
    if (quote === '"' && char === "\\") {
      const next = text[index + 1] ?? "";
      if (
        next === "u" &&
        /^[0-9a-fA-F]{4}$/.test(text.slice(index + 2, index + 6))
      ) {
        value += String.fromCharCode(
          parseInt(text.slice(index + 2, index + 6), 16),
        );
        index += 5;
        continue;
      }
      const escapes: Record<string, string> = {
        n: "\n",
        t: "\t",
        '"': '"',
        "\\": "\\",
        "/": "/",
      };
      if (!(next in escapes)) {
        throw new Error(`line ${line}: unknown escape "\\${next}"`);
      }
      value += escapes[next];
      index++;
      continue;
    }
    value += char;
  }
  if (index >= text.length) {
    throw new Error(`line ${line}: the quote ${quote} is not closed`);
  }
  if (withoutComment(text.slice(index + 1)) !== "") {
    throw new Error(`line ${line}: unexpected text after the closing quote`);
  }
  return value;
}

/**
 * Reads a block value, after "|" (lines kept) or ">" (lines joined)
 *
 * @param header - The "|" or ">", with "-" or "+" for the end of the value
 * @param lines - The indented lines of the value
 * @param line - The line number, for the error
 * @returns The value
 * @throws If the header is not supported
 */
function readBlock(header: string, lines: string[], line: number): string {
  const match = /^([|>])([+-]?)$/.exec(withoutComment(header));
  if (match === null) {
    throw new Error(
      `line ${line}: "${header}" is not supported, use "|" or ">"`,
    );
  }
  const [, style, chomping] = match;
  const indent = Math.min(
    ...lines
      .filter((text) => text.trim() !== "")
      .map((text) => text.length - text.trimStart().length),
  );
  const content = lines.map((text) => text.slice(indent).trimEnd());
  const value =
    style === "|"
      ? content.join("\n")
      : // ">" joins the lines of a paragraph, and an empty line ends it
        content.reduce((folded, text, index) => {
          if (index === 0) {
            return text;
          }
          if (text === "") {
            return `${folded}\n`;
          }
          return content[index - 1] === ""
            ? folded + text
            : `${folded} ${text}`;
        }, "");
  // "-" drops the final line break; the empty lines after the value are
  // already left out, so "+" keeps only one
  return chomping === "-" ? value : `${value}\n`;
}

/**
 * Reads the value of a field: on its own line, or on the indented lines that
 * follow it
 *
 * @param inline - What follows "key:" on the same line
 * @param lines - The indented lines that follow
 * @param line - The line number of the field, for the error
 * @returns The value
 * @throws If the value cannot be read
 */
function readValue(
  inline: string,
  lines: string[],
  line: number,
): FrontmatterValue {
  if (inline.startsWith("|") || inline.startsWith(">")) {
    return readBlock(inline, lines, line);
  }
  if (inline.startsWith('"') || inline.startsWith("'")) {
    const text = [inline, ...lines.map((text) => text.trim())].join(" ").trim();
    return readQuoted(text, line);
  }
  const plain = withoutComment(inline);
  const filled = lines.filter((text) => !isBlank(text));
  if (plain !== "" || filled.length === 0) {
    // a plain value may go on over the next lines, joined with spaces
    return [plain, ...filled.map(withoutComment)].join(" ").trim();
  }
  if (filled.every((text) => /^\s*-(\s|$)/.test(text))) {
    return filled.map((text) => {
      const item = text.trim().slice(1).trim();
      return /^["']/.test(item) ? readQuoted(item, line) : withoutComment(item);
    });
  }
  const indent = filled[0]!.length - filled[0]!.trimStart().length;
  const entries: Record<string, string> = {};
  for (const text of filled) {
    const match = fieldLine.exec(text.trim());
    if (match === null || text.length - text.trimStart().length !== indent) {
      throw new Error(
        `line ${line}: only one level of "key: value" lines is supported`,
      );
    }
    const [, key, value = ""] = match;
    entries[key!] = /^["']/.test(value)
      ? readQuoted(value, line)
      : withoutComment(value);
  }
  return entries;
}

/**
 * Splits SKILL.md into its frontmatter, read as YAML, and its body
 *
 * @param text - The text of SKILL.md
 * @returns The fields of the frontmatter, and the body
 * @throws If there is no frontmatter, or it cannot be read
 */
export function parseFrontmatter(text: string): {
  fields: Map<string, FrontmatterValue>;
  body: string;
} {
  const lines = (
    text.charCodeAt(0) === byteOrderMark ? text.slice(1) : text
  ).split(/\r?\n/);
  if (lines[0]?.trimEnd() !== "---") {
    throw new Error(
      'SKILL.md must start with a "---" line, then the name and the description',
    );
  }
  const end = lines.findIndex(
    (line, index) => index > 0 && line.trimEnd() === "---",
  );
  if (end === -1) {
    throw new Error('the frontmatter of SKILL.md must end with a "---" line');
  }
  const fields = new Map<string, FrontmatterValue>();
  for (let index = 1; index < end;) {
    const line = lines[index]!;
    if (isBlank(line)) {
      index++;
      continue;
    }
    const match = /^\s/.test(line) ? null : fieldLine.exec(line);
    if (match === null) {
      throw new Error(
        `line ${index + 1}: expected "key: value", found "${line.trim()}"`,
      );
    }
    const [, key, inline = ""] = match;
    if (fields.has(key!)) {
      throw new Error(`line ${index + 1}: "${key}" is given twice`);
    }
    let next = index + 1;
    while (
      next < end &&
      (lines[next]!.trim() === "" || /^\s/.test(lines[next]!))
    ) {
      next++;
    }
    // the empty lines before the next field are not part of the value
    let last = next;
    while (last > index + 1 && lines[last - 1]!.trim() === "") {
      last--;
    }
    fields.set(
      key!,
      readValue(inline.trim(), lines.slice(index + 1, last), index + 1),
    );
    index = next;
  }
  return {
    fields,
    body: lines
      .slice(end + 1)
      .join("\n")
      .trim(),
  };
}

/**
 * Checks the name of a skill like the reference validator does
 *
 * @param name - The name in SKILL.md
 * @param folder - The name of the skill's folder
 * @returns What is wrong, one text each
 */
function checkName(name: string, folder: string): string[] {
  const problems: string[] = [];
  if (name.length > maxLengths.name) {
    problems.push(`the name is longer than ${maxLengths.name} characters`);
  }
  if (name !== name.toLowerCase()) {
    problems.push(`the name "${name}" must be lowercase`);
  }
  if (!/^[\p{L}\p{N}-]+$/u.test(name)) {
    problems.push(
      `the name "${name}" may only have letters, digits and hyphens`,
    );
  }
  if (name.startsWith("-") || name.endsWith("-")) {
    problems.push(`the name "${name}" must not start or end with a hyphen`);
  }
  if (name.includes("--")) {
    problems.push(`the name "${name}" must not have two hyphens in a row`);
  }
  if (name !== folder.normalize("NFKC")) {
    problems.push(
      `the name "${name}" must be the name of its folder, "${folder}"`,
    );
  }
  return problems;
}

/**
 * Reads a skill from its SKILL.md and its other files
 *
 * @param folder - The name of the skill's folder
 * @param skillMd - The text of SKILL.md
 * @param files - The other files, by path inside the folder
 * @returns The skill
 * @throws If SKILL.md cannot be read or breaks the rules of the format
 */
export function createSkill(
  folder: string,
  skillMd: string,
  files: Map<string, Blob>,
): Skill {
  const { fields, body } = parseFrontmatter(skillMd);
  const problems: string[] = [];
  const text = (field: string): string | undefined => {
    const value = fields.get(field);
    if (value === undefined) {
      return undefined;
    }
    if (typeof value !== "string") {
      problems.push(`"${field}" must be text`);
      return undefined;
    }
    return value.trim();
  };

  const name = text("name")?.normalize("NFKC");
  if (name === undefined || name === "") {
    problems.push('the "name" is missing');
  } else {
    problems.push(...checkName(name, folder));
  }
  const description = text("description");
  if (description === undefined || description === "") {
    problems.push('the "description" is missing');
  } else if (description.length > maxLengths.description) {
    problems.push(
      `the description is longer than ${maxLengths.description} characters`,
    );
  }
  const compatibility = text("compatibility");
  if (
    compatibility !== undefined &&
    compatibility.length > maxLengths.compatibility
  ) {
    problems.push(
      `"compatibility" is longer than ${maxLengths.compatibility} characters`,
    );
  }
  const license = text("license");
  // "metadata:" with nothing after it has no entries
  const metadata = fields.get("metadata") || {};
  if (typeof metadata === "string" || Array.isArray(metadata)) {
    problems.push('"metadata" must be "key: value" lines');
  }
  if (problems.length > 0) {
    throw new Error(problems.join("; "));
  }

  const unknown = [...fields.keys()].filter(
    (field) => !knownFields.includes(field),
  );
  return {
    name: name!,
    description: description!,
    ...(license !== undefined && license !== "" && { license }),
    ...(compatibility !== undefined &&
      compatibility !== "" && { compatibility }),
    metadata: metadata as Record<string, string>,
    instructions: body,
    skillMd,
    files,
    warnings:
      unknown.length > 0
        ? [
            `ignored, not part of the Agent Skills format: ${unknown.join(", ")}`,
          ]
        : [],
  };
}

/**
 * Finds the skills in the files of a chosen folder: each folder with a
 * SKILL.md is a skill, e.g. a skill's own folder, or a "skills" folder with
 * one folder per skill. Hidden files, e.g. ".DS_Store", are left out.
 *
 * @param files - The files, by path from the chosen folder, which is part of it
 * @param options - Optional abort signal
 * @returns The skills, by name, and the folders that could not be read
 */
export async function readSkills(
  files: FolderFile[],
  options?: { signal?: AbortSignal },
): Promise<{ skills: Skill[]; problems: SkillProblem[] }> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const visible = files
    .map(({ path, content }) => ({
      path: path.replaceAll("\\", "/").replace(/^(\.?\/)+/, ""),
      content,
    }))
    .filter(
      ({ path }) => !path.split("/").some((part) => part.startsWith(".")),
    );
  const roots = visible
    .filter(({ path }) => path === "SKILL.md" || path.endsWith("/SKILL.md"))
    .map(({ path }) => path.slice(0, -"SKILL.md".length).replace(/\/$/, ""));

  const skills: Skill[] = [];
  const problems: SkillProblem[] = [];
  for (const root of roots.toSorted()) {
    const folder = root.split("/").at(-1)!;
    if (root === "") {
      problems.push({
        folder: "",
        error: "SKILL.md must be in a folder named like the skill",
      });
      continue;
    }
    // a file belongs to the innermost skill folder it is in
    const own = new Map<string, Blob>();
    let skillMd: Blob | undefined;
    for (const { path, content } of visible) {
      const inner = roots
        .filter((other) => path.startsWith(`${other}/`))
        .reduce((a, b) => (b.length > a.length ? b : a), "");
      if (inner !== root) {
        continue;
      }
      const inside = path.slice(root.length + 1);
      if (inside === "SKILL.md") {
        skillMd = content;
      } else {
        own.set(inside, content);
      }
    }
    try {
      // Blob.text() takes no abort signal
      const text = await skillMd!.text();
      signal?.throwIfAborted();
      const skill = createSkill(folder, text, own);
      const twin = skills.find((other) => other.name === skill.name);
      if (twin !== undefined) {
        throw new Error(`another skill is already named "${skill.name}"`);
      }
      skills.push(skill);
    } catch (error) {
      signal?.throwIfAborted();
      problems.push({ folder: root, error: (error as Error).message });
    }
  }
  skills.sort((a, b) => a.name.localeCompare(b.name));
  return { skills, problems };
}
