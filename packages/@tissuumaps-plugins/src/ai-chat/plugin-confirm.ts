import type { ChatPluginEntry } from "./chat-plugins";
import type { PluginCodeReport } from "./plugin-code";
import {
  type Confirmation,
  line,
  link,
  makeAnswerButtons,
  makeFrame,
  warningStyle,
} from "./skill-confirm";
import { makeCopyButton, quietTextStyle } from "./ui";

// Before a plugin made by the chat runs, the user sees it in the chat: what
// it is, where it comes from, its code (which opens and can be copied), what
// it uses and a warning for each delicate thing it does, and says yes or no.
// The same question comes for a plugin the browser remembers, the first time
// it is opened after the page was loaded, and for a plugin from a file.

/** Why the question is asked */
export type PluginQuestionKind = "proposed" | "remembered" | "file";

/** The titles of the question, by why it is asked */
const titles: Record<PluginQuestionKind, string> = {
  proposed: "Run a plugin made by the chat?",
  remembered: "Run a plugin that the chat made before?",
  file: "Run a plugin from a file?",
};

/**
 * Asks the user whether to run a plugin made by the chat
 *
 * @param entry - The plugin
 * @param report - What its code uses, and the delicate things it does
 * @param options - Why the question is asked; the plugin it replaces, if any
 * with the same id; whether the browser remembers it already; an optional
 * abort signal, which answers no
 * @returns The question, and the answer: whether to run it, and whether
 * the browser remembers it; and the line that says what happened, for the
 * panel to tell whether the plugin ran
 */
export function confirmPlugin(
  entry: ChatPluginEntry,
  report: PluginCodeReport,
  options: {
    kind: PluginQuestionKind;
    replaces?: ChatPluginEntry;
    remembered?: boolean;
    signal?: AbortSignal;
  },
): Confirmation<{ run: boolean; remember: boolean }> & {
  status: HTMLElement;
} {
  const element = makeFrame(titles[options.kind]);
  element.append(
    line("Plugin", `${entry.name} (${entry.id})`),
    line("What it does", entry.description),
  );
  if (options.replaces !== undefined) {
    const lines = (code: string) => code.split("\n").length;
    element.append(
      line(
        "Replaces",
        `the version of "${options.replaces.name}" (${entry.id}) made before, which closes: ` +
          `its code had ${lines(options.replaces.code)} lines, this one has ${lines(entry.code)}`,
      ),
    );
  }

  const origin = entry.origin;
  if (origin !== undefined) {
    const github = origin.github;
    if (github === undefined) {
      element.append(
        line(
          "Based on",
          `the skill ${origin.skill}${origin.project === undefined ? "" : `, which came with the project (${origin.project})`}`,
        ),
      );
    } else {
      const where = `${github.repo}${github.path === "" ? "" : `/${github.path}`}`;
      element.append(
        line(
          "Based on",
          `the skill ${origin.skill}, from `,
          link(where, `https://github.com/${github.repo}`),
          ", commit ",
          link(
            github.commit.slice(0, 7),
            `https://github.com/${github.repo}/commit/${github.commit}`,
          ),
        ),
      );
      if (github.license !== undefined) {
        element.append(
          line(
            "License",
            `${github.license}: the code of the plugin should say so, with the copyright of the authors`,
          ),
        );
      } else {
        const warning = line(
          "License",
          "⚠ none. The repository that the plugin is based on gives no license, so the rights to use, change and share its code are not given to you: " +
            "you cannot take them for granted. Ask its authors before you use the plugin for your work or share it.",
        );
        warning.style.cssText = warningStyle;
        element.append(warning);
      }
    }
  }

  // the code, which opens with a click and can be copied
  const lines = entry.code.split("\n").length;
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `Code (${lines} lines of JavaScript)`;
  summary.style.cssText = "cursor: pointer;";
  const copyButton = makeCopyButton(() => entry.code, "the code");
  copyButton.style.alignSelf = "flex-end";
  const pre = document.createElement("pre");
  pre.dataset.code = "true";
  pre.textContent = entry.code;
  pre.style.cssText =
    "margin: 0; max-height: 320px; overflow: auto; white-space: pre;" +
    "font-family: monospace; font-size: 12px; border: 1px solid var(--border); border-radius: 4px; padding: 4px;";
  const codeBody = document.createElement("div");
  codeBody.style.cssText =
    "display: flex; flex-direction: column; gap: 4px; margin-top: 4px;";
  codeBody.append(copyButton, pre);
  details.append(summary, codeBody);
  element.append(details);

  if (report.uses.length > 0) {
    const list = document.createElement("ul");
    list.style.cssText = "margin: 0; padding-left: 18px;";
    for (const use of report.uses) {
      const item = document.createElement("li");
      item.textContent = use;
      list.append(item);
    }
    element.append(line("It uses", list));
  }
  for (const { text, found } of report.warnings) {
    const warning = document.createElement("div");
    warning.dataset.warning = "true";
    warning.style.cssText = warningStyle;
    warning.textContent = `⚠ ${text}`;
    if (found.length > 0) {
      const what = document.createElement("div");
      what.style.cssText = `${quietTextStyle}font-family: monospace; font-size: 11px;`;
      what.textContent = `Found: ${found.join(", ")}`;
      warning.append(what);
    }
    element.append(warning);
  }

  const caution = document.createElement("div");
  caution.style.cssText = quietTextStyle;
  caution.textContent =
    "A plugin runs in the page, with the same access to TissUUmaps and to the data it has opened as the plugins you load from a file. " +
    "What it uses was found by reading its code, which may miss things: run it only if you trust it. " +
    'What it changes in the project is not taken back by "Undo" in the chat. ' +
    "The JavaScript and Python that it runs in the sandbox of the chat have no network and cannot change the project.";
  element.append(caution);

  const remember = document.createElement("input");
  remember.type = "checkbox";
  remember.checked = options.remembered === true;
  remember.style.accentColor = "var(--primary)";
  const rememberLabel = document.createElement("label");
  rememberLabel.style.cssText = "display: flex; gap: 4px; align-items: center;";
  rememberLabel.title =
    "Keep the plugin in this browser for the next sessions; you are asked again before it runs";
  rememberLabel.append(remember, "Remember this plugin");
  const { row, answer, status } = makeAnswerButtons(
    { yes: "Run plugin", no: "Cancel" },
    options.signal,
  );
  element.append(rememberLabel, row);
  return {
    element,
    status,
    answer: answer.then((yes) => {
      remember.disabled = true;
      if (!status.textContent) {
        status.textContent = yes ? "Running…" : "Not run.";
      }
      return { run: yes, remember: remember.checked };
    }),
  };
}
