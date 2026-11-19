import type { GitHubPreview } from "./github-skills";
import { makeButton, quietTextStyle, smallButtonStyle } from "./ui";

// Before a skill from GitHub is added, and before a skill gets instructions
// that the chat wrote, the user sees what would change, in the chat, and
// says yes or no. "Add skill from GitHub…" in the settings and the chat
// open the same confirmation.

/** A question in the chat, and its answer once the user gives it */
export type Confirmation<TAnswer> = {
  element: HTMLElement;
  answer: Promise<TAnswer>;
};

/** The look of the main button of a confirmation */
const mainButtonStyle =
  "background: var(--primary); color: var(--primary-foreground); border-color: var(--primary);";

/** The look of a warning */
export const warningStyle =
  "padding: 4px 6px; border-radius: 4px; border: 1px solid color-mix(in oklch, var(--destructive) 50%, transparent);" +
  "background: color-mix(in oklch, var(--destructive) 10%, transparent);";

/**
 * Creates a line of text, with a bold label
 *
 * @param label - e.g. "License"
 * @param content - The rest of the line
 * @returns The line
 */
export function line(
  label: string,
  ...content: (string | Node)[]
): HTMLElement {
  const element = document.createElement("div");
  const strong = document.createElement("strong");
  strong.textContent = `${label}: `;
  element.append(strong, ...content);
  return element;
}

/**
 * Creates a link that opens in another tab
 *
 * @param text - The text of the link
 * @param url - Where it goes
 * @returns The link
 */
export function link(text: string, url: string): HTMLAnchorElement {
  const element = document.createElement("a");
  element.textContent = text;
  element.href = url;
  element.target = "_blank";
  element.rel = "noopener noreferrer";
  element.style.color = "inherit";
  return element;
}

/**
 * A size in kilobytes or megabytes, readably
 *
 * @param bytes - The size
 * @returns e.g. "30 KB" or "2.4 MB"
 */
function megabytes(bytes: number): string {
  return bytes < 1e6
    ? `${Math.max(1, Math.round(bytes / 1e3))} KB`
    : `${(bytes / 1e6).toFixed(1)} MB`;
}

/**
 * Creates the buttons of a question; a click answers it once, and so does
 * the abort signal (as "no")
 *
 * @param labels - The yes and the no button
 * @param signal - Optional abort signal, e.g. Stop
 * @returns The row of buttons, and the answer
 */
export function makeAnswerButtons(
  labels: { yes: string; no: string },
  signal?: AbortSignal,
): { row: HTMLElement; answer: Promise<boolean>; status: HTMLElement } {
  const row = document.createElement("div");
  row.style.cssText =
    "display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 4px;";
  const status = document.createElement("span");
  status.style.cssText = quietTextStyle;
  let answer!: (yes: boolean) => void;
  const answered = new Promise<boolean>((resolve) => {
    answer = resolve;
  });
  const finish = (yes: boolean) => {
    yesButton.disabled = true;
    noButton.disabled = true;
    signal?.removeEventListener("abort", onAbort);
    answer(yes);
  };
  const onAbort = () => {
    status.textContent = "Stopped.";
    finish(false);
  };
  const yesButton = makeButton(
    labels.yes,
    () => finish(true),
    smallButtonStyle + mainButtonStyle,
  );
  const noButton = makeButton(labels.no, () => finish(false), smallButtonStyle);
  row.append(yesButton, noButton, status);
  if (signal?.aborted) {
    onAbort();
  } else {
    signal?.addEventListener("abort", onAbort, { once: true });
  }
  return { row, answer: answered, status };
}

/**
 * The frame of a question in the chat
 *
 * @param title - e.g. "Add a skill from GitHub?"
 * @returns The frame
 */
export function makeFrame(title: string): HTMLElement {
  const element = document.createElement("div");
  element.dataset.who = "Confirm";
  element.style.cssText =
    "display: flex; flex-direction: column; gap: 4px; padding: 6px 8px; border: 1px solid var(--border);" +
    "border-radius: 6px; overflow-wrap: anywhere;";
  const heading = document.createElement("strong");
  heading.textContent = title;
  element.append(heading);
  return element;
}

/**
 * Asks the user whether to add the skill that a repository would give:
 * where exactly it comes from, its license, its files, what its Python code
 * needs and what will not work, and that it is code written by others
 *
 * @param preview - What the skill would hold
 * @param options - Optional abort signal, which answers no
 * @returns The question, and the answer: whether to add it, and whether the
 * browser remembers it
 */
export function confirmGitHubSkill(
  preview: GitHubPreview,
  options?: { signal?: AbortSignal },
): Confirmation<{ add: boolean; remember: boolean }> {
  const { origin, python } = preview;
  const element = makeFrame("Add a skill from GitHub?");

  const where = `${origin.repo}${origin.path === "" ? "" : `/${origin.path}`}`;
  element.append(
    line("Repository", link(where, preview.url)),
    ...(preview.description === "" ? [] : [line("About", preview.description)]),
    line(
      "Commit",
      link(
        origin.commit,
        `https://github.com/${origin.repo}/commit/${origin.commit}`,
      ),
      " (the files exactly as they are in this commit)",
    ),
  );
  const fromTop =
    preview.topFiles.length === 0
      ? ""
      : `; its text (${preview.topFiles.join(", ")}) comes from the top of the repository, and is added to the skill`;
  if (origin.license !== undefined) {
    element.append(line("License", `${origin.license}${fromTop}`));
  } else {
    const warning = line(
      "License",
      "⚠ none. This repository gives no license, so the rights to use, change and share its code are not given to you: " +
        "you cannot take them for granted. Ask its authors before you use it for your work or share what you make with it.",
    );
    warning.style.cssText = warningStyle;
    element.append(warning);
  }

  const skipped = preview.skipped.length;
  element.append(
    line(
      "Files",
      `${preview.files.length} text and code files, ${megabytes(preview.bytes)}` +
        (preview.madeSkillMd ? " (with SKILL.md, written by the chat)" : "") +
        (skipped > 0 ? `; ${skipped} left out` : ""),
    ),
  );
  if (skipped > 0) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Files left out";
    summary.style.cssText = `${quietTextStyle}cursor: pointer;`;
    const list = document.createElement("div");
    list.style.cssText = `${quietTextStyle}white-space: pre-wrap;`;
    list.textContent = preview.skipped
      .slice(0, 100)
      .map(({ path, reason }) => `${path}: ${reason}`)
      .concat(skipped > 100 ? [`… and ${skipped - 100} more`] : [])
      .join("\n");
    details.append(summary, list);
    element.append(details);
  }

  element.append(
    line(
      "Skills",
      preview.skills.map(({ name }) => name).join(", ") || "none",
      preview.madeSkillMd
        ? ". The repository has no SKILL.md: the chat wrote a minimal one, which says where the code comes from. " +
            "The assistant can propose better instructions later, which you confirm."
        : "",
    ),
  );
  for (const { folder, error } of preview.problems) {
    element.append(line("Not added", `${folder}: ${error}`));
  }

  if (python.hasPython) {
    if (python.pyodideError !== undefined) {
      element.append(
        line(
          "Python packages",
          `which ones Pyodide has could not be checked (${python.pyodideError}).`,
        ),
      );
    }
    if (python.packages.length > 0) {
      const list = document.createElement("ul");
      list.style.cssText = "margin: 0; padding-left: 18px;";
      for (const needed of python.packages) {
        const item = document.createElement("li");
        const where =
          needed.files.length > 0
            ? ` (${needed.files.slice(0, 3).join(", ")}${needed.files.length > 3 ? ", …" : ""})`
            : "";
        const listed = needed.listed ? ", listed as needed" : "";
        item.textContent =
          python.pyodideError !== undefined
            ? `${needed.name}${listed}${where}`
            : needed.pyodide !== undefined
              ? `${needed.name}: in Pyodide ✓${needed.pyodide === needed.name ? "" : ` (${needed.pyodide})`}${listed}${where}`
              : `${needed.name}: not in Pyodide ✗${listed}${where}`;
        list.append(item);
      }
      element.append(line("Python packages", list));
    } else if (python.pyodideError === undefined) {
      element.append(
        line("Python packages", "only the standard library of Python ✓"),
      );
    }
    if (python.broken.length > 0) {
      const broken = line(
        "Will not work",
        "Python in the chat has only the packages of Pyodide, and cannot install others, so these files cannot run: " +
          python.broken
            .map(({ path, missing }) => `${path} (needs ${missing.join(", ")})`)
            .join("; ") +
          ". The rest can.",
      );
      broken.style.cssText = warningStyle;
      element.append(broken);
    }
  }

  const caution = document.createElement("div");
  caution.style.cssText = warningStyle;
  caution.textContent =
    "⚠ This is code written by other people. Its SKILL.md, README and other texts become instructions for the assistant, " +
    "and its scripts run in the sandbox without asking again: the sandbox keeps them away from the network, the page and your API key, " +
    "but they can read the data of the project. Add it only if you trust it.";
  element.append(caution);

  const remember = document.createElement("input");
  remember.type = "checkbox";
  remember.checked = true;
  remember.style.accentColor = "var(--primary)";
  const rememberLabel = document.createElement("label");
  rememberLabel.style.cssText = "display: flex; gap: 4px; align-items: center;";
  rememberLabel.title =
    "Keep the skill in this browser for the next sessions, in every project";
  rememberLabel.append(remember, "Remember this skill");
  const canAdd = preview.skills.length > 0;
  const { row, answer, status } = makeAnswerButtons(
    { yes: "Add skill", no: "Cancel" },
    options?.signal,
  );
  if (!canAdd) {
    row.querySelector("button")!.disabled = true;
    status.textContent = "There is no valid skill to add.";
  }
  element.append(rememberLabel, row);
  return {
    element,
    answer: answer.then((yes) => {
      remember.disabled = true;
      if (!status.textContent) {
        status.textContent = yes && canAdd ? "Added." : "Not added.";
      }
      return { add: yes && canAdd, remember: remember.checked };
    }),
  };
}

/**
 * Asks the user whether a skill may get the instructions that the chat
 * wrote: its new SKILL.md, shown in full
 *
 * @param name - The name of the skill
 * @param skillMd - The new SKILL.md
 * @param options - Optional abort signal, which answers no
 * @returns The question, and whether the user said yes
 */
export function confirmSkillMd(
  name: string,
  skillMd: string,
  options?: { signal?: AbortSignal },
): Confirmation<boolean> {
  const element = makeFrame(`New instructions for the skill ${name}?`);
  const explanation = document.createElement("div");
  explanation.style.cssText = quietTextStyle;
  explanation.textContent =
    "The assistant wrote this SKILL.md. With it, the assistant follows these instructions whenever it uses the skill.";
  const text = document.createElement("pre");
  text.textContent = skillMd;
  text.style.cssText =
    "margin: 0; max-height: 240px; overflow: auto; white-space: pre-wrap;" +
    "font-family: monospace; font-size: 12px; border: 1px solid var(--border); border-radius: 4px; padding: 4px;";
  const { row, answer, status } = makeAnswerButtons(
    { yes: "Use these instructions", no: "Keep the old ones" },
    options?.signal,
  );
  element.append(explanation, text, row);
  return {
    element,
    answer: answer.then((yes) => {
      if (!status.textContent) {
        status.textContent = yes ? "Used." : "Kept the old ones.";
      }
      return yes;
    }),
  };
}
