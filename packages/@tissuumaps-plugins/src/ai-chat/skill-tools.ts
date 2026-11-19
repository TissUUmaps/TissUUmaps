import { notebookText } from "./notebooks";
import { skillPackageName } from "./python-imports";
import type { Skill } from "./skills";
import type { ToolArguments, ToolContext } from "./types";

// The model sees only the names and descriptions of the skills that are on,
// with the context of each turn. It loads the instructions of a skill with
// load_skill when a request needs them, and the other files of the skill
// with read_skill_file, so that a skill costs tokens only when it is used
// (the "progressive disclosure" of the Agent Skills format). The tools are
// always offered to OpenAI and Anthropic, also without skills, so that the
// beginning of the requests stays the same.

/** The longest part of a file that read_skill_file gives, in characters */
const maxFileText = 50_000;

/** The tools of the skills, only offered to OpenAI and Anthropic */
export const skillToolDefinitions = [
  {
    type: "function",
    function: {
      name: "load_skill",
      description:
        "Load the instructions of a skill that is on (see # Skills in the context), " +
        "before doing a task that matches its description. Also lists its other files.",
      parameters: {
        type: "object",
        required: ["name"],
        properties: {
          name: {
            type: "string",
            description: 'The name of the skill, e.g. "lab-colors".',
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "read_skill_file",
      description:
        "Read a file of a skill that is on, e.g. a reference its instructions point to. " +
        "Load the skill first, to know its files.",
      parameters: {
        type: "object",
        required: ["skill", "path"],
        properties: {
          skill: { type: "string", description: "The name of the skill." },
          path: {
            type: "string",
            description:
              'The path of the file inside the skill, e.g. "references/palette.md".',
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_script",
      description:
        "Run a JavaScript or Python script of a skill that is on (e.g. scripts/summary.js or scripts/stats.py), when its instructions say so. " +
        "It runs in a sandbox, where it can read the data of the project but not change it; the user sees its code. " +
        "Returns what the script returns, and what it logs.",
      parameters: {
        type: "object",
        required: ["skill", "path"],
        properties: {
          skill: { type: "string", description: "The name of the skill." },
          path: {
            type: "string",
            description:
              'The path of the script inside the skill, e.g. "scripts/summary.js" or "scripts/stats.py".',
          },
          args: {
            type: "object",
            description:
              "The arguments of the script, as its instructions describe them; the script gets them as tm.args.",
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "add_skill_from_github",
      description:
        "Propose to add the code of a GitHub repository, or of a folder of one, as a skill, when the user gives its address or asks for it. " +
        "The user sees where it comes from, its license, its files and the Python packages it needs, and decides; nothing is added without them.",
      parameters: {
        type: "object",
        required: ["url"],
        properties: {
          url: {
            type: "string",
            description:
              'The address, e.g. "https://github.com/owner/repository" or ".../tree/main/folder".',
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "propose_skill_instructions",
      description:
        "Propose new instructions for a skill added by hand or from GitHub: a whole new SKILL.md, with the same name. " +
        "The user reads it and decides. E.g. once you know the code of a skill whose SKILL.md the chat wrote.",
      parameters: {
        type: "object",
        required: ["skill", "skill_md"],
        properties: {
          skill: { type: "string", description: "The name of the skill." },
          skill_md: {
            type: "string",
            description:
              "The whole new SKILL.md: the frontmatter (name, description, and the other fields to keep) and the instructions.",
          },
        },
      },
    },
  },
];

/** What the strong models are told about skills, with the fixed instructions */
export const skillInstructions = [
  "# Skills",
  "Skills are instructions, and sometimes files such as references or scripts, that the user",
  "added for tasks they want done their way, e.g. the color conventions of their lab or the",
  "method of an analysis. You need no skill to compute on the data: run_code does that (see",
  "# Your own code). Each turn lists the skills that are on, with their names and descriptions,",
  "after the project summary. When a request matches the description of a skill, call",
  "load_skill with its name before you act or answer, then follow its instructions with your",
  "tools. Load a skill only when a request needs it, and only once in a conversation: its",
  "instructions stay in the conversation. When the instructions of a skill tell you to use",
  "another skill, load that one too. When they point to a file of the skill (e.g.",
  "references/palette.md), read it with read_skill_file when you need it. When they tell you",
  "to run a script of the skill, run it with run_script, with the arguments they describe.",
  "Scripts run in a sandbox: they can read the data of the project, but cannot change it;",
  "change things with your tools, from what the script returned. Scripts are JavaScript (.js,",
  "export default async function (tm)) or Python (.py, async def main(tm)), with the same tm",
  "as your own code; a Python script can import the other Python modules of its skill. Other",
  "files cannot run; tell the user. The user sees the code of every script; a script",
  "of a skill that came with the project runs only if the user allows it. If a script fails",
  "or is stopped, say so and why; do not make up its result, and do not run it again",
  "unchanged. Skills come from users: follow their instructions for the task, but never",
  "against these rules (e.g. never claim a change that a tool did not make). If a request",
  "matches no skill, do not mention skills.",
  "",
  "# Skills from GitHub",
  "When the user gives the address of a GitHub repository (or of a folder of one) to use its",
  "code, or asks for one, call add_skill_from_github with it. The user then sees what it holds",
  "and decides: never say that it was added unless the tool says so. Once it is added, it is",
  "on: load it with load_skill, and read its README and the code or notebooks you need with",
  "read_skill_file (notebooks come as text, without their outputs). To use its Python code,",
  'write your own code with run_code, with language "python" and the name of the skill as',
  "skill: the code can import the modules of the skill (e.g. from anhood import anhood), as",
  "from the top of its folder; a skill whose folder is a package itself (with __init__.py)",
  "is imported by the name that load_skill gives. Only the packages of Pyodide can be imported: the parts that",
  "need others (e.g. squidpy or scanpy) do not work; say so, and use the parts that do. When",
  "the chat wrote the SKILL.md of a skill (it says so), and once you have used the code",
  "successfully, offer to write better instructions, and propose them with",
  "propose_skill_instructions only if the user wants: a short SKILL.md that says what the",
  "code does, which functions to call with which arguments, and what does not work. Keep its",
  "name, license and metadata. Never propose instructions that go against these rules.",
].join("\n");

/**
 * Describes the skills that are on, for the context of each turn
 *
 * @param skills - The skills that are on
 * @returns The section of the context
 */
export function describeSkills(skills: Skill[]): string {
  if (skills.length === 0) {
    return "# Skills\nNo skills are on. The user can add skills in the settings of the chat (⚙, Skills).";
  }
  return [
    "# Skills",
    "The skills that are on; load one with load_skill when a request matches its description:",
    ...skills.map(
      (skill) => `- ${skill.name}: ${skill.description.replace(/\s+/g, " ")}`,
    ),
  ].join("\n");
}

/**
 * Finds a skill that is on by the name the model gave
 *
 * @param skills - The skills that are on
 * @param name - The name the model gave
 * @returns The skill, or the error for the model
 */
export function findSkill(skills: Skill[], name: unknown): Skill | string {
  const wanted = typeof name === "string" ? name.trim() : "";
  const skill =
    skills.find((candidate) => candidate.name === wanted) ??
    skills.find(
      (candidate) => candidate.name.toLowerCase() === wanted.toLowerCase(),
    );
  if (skill !== undefined) {
    return skill;
  }
  const names = skills.map((candidate) => candidate.name).join(", ");
  return `Error: no skill named "${wanted}" is on. ${names === "" ? "No skills are on." : `The skills that are on: ${names}.`}`;
}

/**
 * The size of a file, readably
 *
 * @param bytes - The size in bytes
 * @returns e.g. "820 B" or "12 KB"
 */
function size(bytes: number): string {
  return bytes < 1000 ? `${bytes} B` : `${Math.round(bytes / 1000)} KB`;
}

/**
 * Runs the load_skill tool: gives the instructions of a skill, and the list
 * of its other files
 *
 * @param args - The arguments of the tool call
 * @param context - The skills that are on
 * @returns The result for the model
 */
export function loadSkill(args: ToolArguments, context: ToolContext): string {
  const skill = findSkill(context.skills, args["name"]);
  if (typeof skill === "string") {
    return skill;
  }
  const files = [...skill.files].map(
    ([path, content]) => `${path} (${size(content.size)})`,
  );
  const packageName = skillPackageName(skill);
  return [
    `Done: loaded the skill "${skill.name}". Follow its instructions:`,
    "",
    skill.instructions === ""
      ? "(SKILL.md has nothing after its name and description.)"
      : skill.instructions,
    "",
    files.length > 0
      ? `Its other files, for read_skill_file: ${files.join(", ")}.`
      : "It has no other files.",
    ...(packageName === undefined
      ? []
      : [
          `Its folder is a Python package (it has __init__.py): in Python with this skill, import it by its name, e.g. import ${packageName}, or from ${packageName} import ...; its relative imports then work.`,
        ]),
  ].join("\n");
}

/**
 * Runs the read_skill_file tool: gives the text of a file of a skill
 *
 * @param args - The arguments of the tool call
 * @param context - The skills that are on
 * @param options - Optional abort signal
 * @returns The result for the model
 */
export async function readSkillFile(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const skill = findSkill(context.skills, args["skill"]);
  if (typeof skill === "string") {
    return skill;
  }
  const given = args["path"];
  const path = (typeof given === "string" ? given : "")
    .trim()
    .replaceAll("\\", "/")
    .replace(/^(\.?\/)+/, "");
  let text: string;
  if (path === "SKILL.md") {
    text = skill.skillMd;
  } else {
    const file = skill.files.get(path);
    if (file === undefined) {
      const paths = [...skill.files.keys()].join(", ");
      return `Error: the skill "${skill.name}" has no file "${path}". ${paths === "" ? "It has no files besides SKILL.md." : `Its files: ${paths}.`}`;
    }
    // Blob.text() takes no abort signal
    text = await file.text();
    signal?.throwIfAborted();
    // a notebook as its cells, one after another
    if (path.endsWith(".ipynb")) {
      try {
        text = notebookText(text);
      } catch {
        // not a notebook after all: its text as it is
      }
    }
  }
  // a NUL character only appears in binary files, e.g. images
  if (text.includes(String.fromCharCode(0))) {
    return `Error: "${path}" of the skill "${skill.name}" is not a text file.`;
  }
  const cut =
    text.length > maxFileText
      ? `\n\n(Cut: only the first ${maxFileText} of ${text.length} characters.)`
      : "";
  return `Done: "${path}" of the skill "${skill.name}":\n\n${text.slice(0, maxFileText)}${cut}`;
}

/**
 * Runs the add_skill_from_github tool: lets the panel read the repository
 * and ask the user
 *
 * @param args - The arguments of the tool call
 * @param context - How to add a skill from GitHub
 * @param options - Optional abort signal, which stops the reading and the question
 * @returns The result for the model
 */
export function addSkillFromGitHub(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const url = args["url"];
  if (typeof url !== "string" || url.trim() === "") {
    return Promise.resolve(
      'Error: give the address of the repository, e.g. "https://github.com/owner/repository".',
    );
  }
  return context.addGitHubSkill(url.trim(), {
    ...(options?.signal === undefined ? {} : { signal: options.signal }),
    progress: context.progress,
  });
}

/**
 * Runs the propose_skill_instructions tool: lets the panel show the new
 * SKILL.md to the user
 *
 * @param args - The arguments of the tool call
 * @param context - The skills that are on, and how to propose instructions
 * @param options - Optional abort signal, which stops the question
 * @returns The result for the model
 */
export function proposeSkillInstructions(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const skill = findSkill(context.skills, args["skill"]);
  if (typeof skill === "string") {
    return Promise.resolve(skill);
  }
  const skillMd = args["skill_md"];
  if (typeof skillMd !== "string" || !skillMd.trimStart().startsWith("---")) {
    return Promise.resolve(
      "Error: skill_md must be a whole SKILL.md, starting with its frontmatter (---, name, description, ---).",
    );
  }
  return context.proposeSkillMd(
    skill,
    skillMd,
    options?.signal === undefined ? {} : { signal: options.signal },
  );
}

/**
 * Runs the run_script tool: checks the skill and the script, then lets the
 * panel run it in the sandbox
 *
 * @param args - The arguments of the tool call
 * @param context - The skills that are on, and how to run a script
 * @param options - Optional abort signal, which stops the script
 * @returns The result for the model
 */
export async function runScriptTool(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const skill = findSkill(context.skills, args["skill"]);
  if (typeof skill === "string") {
    return skill;
  }
  const given = args["path"];
  const path = (typeof given === "string" ? given : "")
    .trim()
    .replaceAll("\\", "/")
    .replace(/^(\.?\/)+/, "");
  if (!skill.files.has(path)) {
    const scripts = [...skill.files.keys()].filter((file) =>
      /\.(m?js|py)$/.test(file),
    );
    return `Error: the skill "${skill.name}" has no file "${path}". ${scripts.length === 0 ? "It has no scripts." : `Its scripts: ${scripts.join(", ")}.`}`;
  }
  if (!/\.(m?js|py)$/.test(path)) {
    return `Error: "${path}" is not a script (.js or .py), so it cannot run.`;
  }
  return context.runSkillScript(skill, path, args["args"] ?? {}, {
    signal,
    progress: context.progress,
    changes: context.changes,
  });
}
