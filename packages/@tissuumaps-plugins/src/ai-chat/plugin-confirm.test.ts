// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import type { ChatPluginEntry } from "./chat-plugins";
import { confirmPlugin } from "./plugin-confirm";

/** A plugin based on the skill anhood, from a repository */
function anhoodPlugin(license?: string): ChatPluginEntry {
  return {
    id: "chat-anhood",
    name: "Neighborhood enrichment",
    description: "Runs anhood on the clusters.",
    code: "export default () => ({ mount() {} });",
    made: "2026-11-18",
    origin: {
      skill: "anhood",
      github: {
        repo: "lab/anhood",
        path: "",
        commit: "a7e9fa555837a46ee47a02d7f6e4a70cb3892b0a",
        ...(license === undefined ? {} : { license }),
      },
    },
  };
}

describe("confirmPlugin", () => {
  it("says where a plugin comes from and its license", async () => {
    const stop = new AbortController();
    const question = confirmPlugin(
      anhoodPlugin("MIT"),
      { uses: [], warnings: [] },
      { kind: "remembered", remembered: true, signal: stop.signal },
    );
    const text = question.element.textContent;
    expect(text).toContain("Run a plugin that the chat made before?");
    expect(text).toContain(
      "Based on: the skill anhood, from lab/anhood, commit a7e9fa5",
    );
    expect(text).toContain(
      "License: MIT: the code of the plugin should say so, with the copyright of the authors",
    );
    expect(text).not.toContain("⚠");
    const links = [...question.element.querySelectorAll("a")].map(
      (a) => a.href,
    );
    expect(links).toEqual([
      "https://github.com/lab/anhood",
      "https://github.com/lab/anhood/commit/a7e9fa555837a46ee47a02d7f6e4a70cb3892b0a",
    ]);
    // remembered already: the box is on; Stop answers no
    expect(question.element.querySelector("input")?.checked).toBe(true);
    stop.abort();
    expect(await question.answer).toEqual({ run: false, remember: true });
  });

  it("warns clearly about a plugin based on a repository without a license", () => {
    const question = confirmPlugin(
      anhoodPlugin(),
      { uses: [], warnings: [] },
      {
        kind: "proposed",
        replaces: {
          ...anhoodPlugin(),
          code: "// old\nexport default 1;\n// end",
        },
      },
    );
    const text = question.element.textContent;
    expect(text).toContain(
      'Replaces: the version of "Neighborhood enrichment" (chat-anhood) made before, which closes: its code had 3 lines, this one has 1',
    );
    expect(text).toContain(
      "License: ⚠ none. The repository that the plugin is based on gives no license, so the rights to use, change and share its code are not given to you",
    );
    expect(question.element.querySelector("input")?.checked).toBe(false);
  });
});
