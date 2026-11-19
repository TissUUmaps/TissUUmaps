import { type ChatPluginState, pluginCredit, pluginId } from "./chat-plugins";
import { findSkill } from "./skill-tools";
import type { ToolArguments, ToolContext } from "./types";

// The strong models can make plugins for TissUUmaps: a panel with buttons,
// sliders, interactive charts, tools, which stays and can be used again.
// The model writes the code and calls propose_plugin; the user reads it in
// the chat, with what it uses and what is delicate in it, and runs it or not
// (see plugin-confirm.ts and chat-plugins.ts).

/** The longest name of a plugin, in characters */
const maxName = 60;

/** The longest description of a plugin, in characters */
const maxDescription = 500;

/** The longest code of a plugin, in characters */
const maxCode = 200_000;

/** The propose_plugin tool, only offered to OpenAI and Anthropic */
export const proposePluginDefinition = {
  type: "function",
  function: {
    name: "propose_plugin",
    description:
      "Propose a plugin for TissUUmaps that you wrote: a panel of its own with buttons, fields, sliders, interactive charts or tools, which stays and can be used again. " +
      "The user sees its code, what it uses and its delicate parts, and runs it or not; then its panel opens. " +
      "Before you call it, tell the user what the plugin does and every delicate thing it does. To change a plugin you made, propose it again with the same id.",
    parameters: {
      type: "object",
      required: ["id", "name", "description", "code"],
      properties: {
        id: {
          type: "string",
          description:
            'A short id of small letters, digits and dashes, e.g. "anhood"; the chat adds "chat-" before it. The same id replaces the plugin made before with it.',
        },
        name: {
          type: "string",
          description:
            'The name, shown in the plugins menu and on its panel, e.g. "Neighborhood enrichment".',
        },
        description: {
          type: "string",
          description:
            "What the plugin does, in one or two sentences, for the user.",
        },
        code: {
          type: "string",
          description:
            "The plugin: a JavaScript module, export default function createPlugin(chat) { return { mount(container, stores) { ... } }; }",
        },
        skill: {
          type: "string",
          description:
            "Only when the plugin uses the code or the method of a skill that is on (e.g. one from GitHub): its name. The user then sees where it comes from and its license.",
        },
      },
    },
  },
};

/** What the strong models are told about plugins, with the fixed instructions */
export const pluginInstructions = [
  "# Plugins that you make",
  "With propose_plugin you can make a plugin for TissUUmaps: a panel of its own, opened from",
  "the plugins menu, with buttons, fields, sliders, interactive charts or tools, that stays",
  "and can be used again, also with the chat closed. Make one when the user asks for a",
  "plugin, a panel, a tool or an interactive view to keep; for a single result, use",
  "run_code instead. A plugin can do anything that science needs: read the project and",
  "its tables, compute in JavaScript or Python in the sandbox of the chat (also with the",
  "modules of a skill, e.g. one from GitHub), and show the results.",
  "",
  "Before you call propose_plugin, always tell the user in your message what the plugin",
  "will do, and every delicate thing that it does: using the network, the memory of the",
  "browser (where the API keys may be), changing the project (Undo in the chat does not take",
  "that back), loading code from the internet; and, if it is based on a repository without",
  "a license, that the rights to use and share that code are not given. Do these delicate",
  "things only when the user asks for them. The user then sees the code, a summary of what",
  "it uses with warnings, and decides; you get the answer as the result.",
  "",
  'To fix or improve a plugin made before (the context lists them, under "# Plugins made by',
  'the chat", with their ids), never make a new one: read its current code with read_plugin',
  "(unless this conversation has its latest code), start from that code, change only what",
  "the user asks, keep every feature it had, and propose it with the SAME id and name. It",
  'then replaces the old version, and the user sees "Replaces". A new id is only for a',
  "plugin that does something else. A chart that it draws itself (on a canvas or in SVG)",
  "is replaced by the ready-made piece that fits (see below), keeping what it did (e.g.",
  "the click on a cell): this is how to fix missing numbers, colors or theme in a chart.",
  "",
  "The code is one JavaScript module (no imports, no build step, no frameworks):",
  "  export default function createPlugin(chat) {",
  "    return {",
  "      mount(container, stores) {",
  "        // build the panel in container; return a function that cleans up",
  "        return () => { /* unsubscribe, stop timers */ };",
  "      },",
  "    };",
  "  }",
  'The chat gives it the id (with "chat-" before yours) and the name. mount builds the panel',
  "each time the user opens it, in container, a fresh element of the panel that is not in",
  'the page yet when mount runs: make it a column (container.style.cssText = "display: flex;',
  "flex-direction: column; gap: 8px; padding: 8px; height: 100%; overflow: auto;",
  'box-sizing: border-box;"). Build the elements with document.createElement and',
  "textContent; never put data or names into innerHTML. Style the elements inline with the",
  "colors of the theme as CSS variables: var(--background), var(--foreground), var(--muted),",
  "var(--muted-foreground), var(--primary), var(--primary-foreground), var(--accent),",
  "var(--border), var(--card), var(--destructive); they follow the theme by themselves.",
  "A canvas does not understand var(--…) (it draws black instead): use chat.theme(), which",
  "gives the same colors, resolved, as { background, foreground, muted, mutedForeground,",
  "primary, primaryForeground, accent, border, card, destructive, dark }, and draw again",
  "with chat.onThemeChange((theme) => draw()), which is called once the page shows the new",
  "theme; call the function that it returns in the cleanup.",
  "stores has projectStore, dataStore, appStore and settingsStore, the stores of the app",
  "state described above: getState() gives the state, subscribe(listener) returns a function",
  "that unsubscribes. Read them freely; change the project only when the user asks.",
  "",
  "chat lets the plugin use the sandbox of the chat:",
  "- await chat.run({ language, code, args, skill, onProgress, signal, timeLimit }): runs",
  "  code exactly like run_code, with tm.args = args. The language must match the code:",
  '  language "python" with async def main(tm), or language "javascript" (the default) with',
  "  export default async function (tm) { … }; code in the other language is refused. With",
  "  skill (the name of a skill that is on), Python imports its modules. It returns",
  "  { result, text, logs, charts, files, saved }: result is what the code returned, parsed",
  '  from JSON, so return plain dicts and lists (e.g. frame.to_dict(orient="split"), numbers',
  "  as floats, NaN as None), never a DataFrame as text; charts are the PNG images of the",
  "  matplotlib figures ({ png, width, height, description }); files are those given with",
  "  tm.download. It throws an Error with the message and error.logs when the code fails.",
  "  onProgress(text) gets the progress (downloading and starting Python, tm.progress); each",
  "  plugin has its own Python, which starts the first time (some seconds). For example:",
  "    const PYTHON = `",
  "    import numpy as np",
  "    from anhood import anhood",
  "",
  "    async def main(tm):",
  '        table, k = tm.args["table"], int(tm.args["k"])',
  '        x = await tm.read(table, "x")',
  '        y = await tm.read(table, "y")',
  '        groups = [str(g) for g in (await tm.read(table, "leiden")).tolist()]',
  "        z, counts = anhood(np.column_stack([x, y]), groups, k=k)",
  '        return {"groups": list(z.index), "z": z.to_numpy().tolist()}',
  "    `;",
  "    try {",
  '      const { result } = await chat.run({ language: "python", skill: "anhood", code: PYTHON,',
  '        args: { table: "cells", k: 10 }, onProgress: (text) => (status.textContent = text) });',
  "      draw(result.groups, result.z);",
  "    } catch (error) {",
  '      status.textContent = "Error: " + error.message;',
  "    }",
  "  Inside a JavaScript template string, write the Python as it is, without ` or ${.",
  "  Columns of groups may be read as numbers (e.g. leiden from a CSV gives 3.0): name",
  '  such groups as whole numbers ("3", not "3.0"), and sort them as numbers.',
  "- await chat.tables(), chat.columns(table), chat.read(table, column): as tm in the code.",
  "- await chat.saveColumn(table, name, values, { ids }): as tm.saveColumn; Undo in the chat",
  "  does not take it back.",
  "- chat.skills(): the skills that are on, as [{ name, description }].",
  "- chat.theme() and chat.onThemeChange(listener): the colors of the theme, see above.",
  "- chat.heatmap, chat.bars, chat.scatter, chat.legend: ready-made charts, see below.",
  "- chat.showShapes({ shapes, like, color, fill, name }): shows temporary shapes on the",
  "  image, drawn by TissUUmaps, in the coordinates of the points (the same as the x and y",
  "  columns of their table): shapes is a list of { rect: [x, y, width, height] },",
  "  { circle: [x, y, radius] } or { polygon: [[x, y], ...] }; like is the id or name of the",
  "  points or labels whose coordinates they use (default: the first points); color a name",
  '  or "#rrggbb" (default magenta); fill: true fills them lightly. It returns an id;',
  "  chat.hideShapes(id) removes them, chat.hideShapes() all of them; they go away with the",
  "  plugin too. E.g. the size of a pixel or of a smoothing radius of a method, shown where",
  "  the cells are. They are listed under Shapes in TissUUmaps while they are shown.",
  "The sandbox has no network and cannot change the project; the plugin gets no more from",
  "it than your own code does.",
  "",
  "Make it good to use: labels and short help texts, sensible defaults (e.g. the cluster",
  "column from the summary), buttons disabled while it computes, the progress, and errors",
  "shown in the panel (with error.message), never only in the console.",
  "Charts: never draw a chart yourself (no canvas, no SVG of your own) when one of these",
  "ready-made pieces fits; they are made and tested for TissUUmaps, and they have the",
  "numbers on the axes (groups named by numbers in number order), a color bar with its",
  "numbers (centered on 0, from minus to plus the largest absolute value, for values with",
  "a sign), the colors of the light and dark theme (drawn again when it changes), the",
  "values under the mouse, and buttons Save PNG and Save CSV. Each takes an empty element",
  "of the panel of its own (a div), and returns { update(options), png(), savePng(),",
  "csv(), saveCsv(), destroy() }; calling it again on the same element replaces the chart,",
  "and update(options) changes only the options given:",
  "- chat.heatmap(element, { rows, columns, values, title, rowTitle, columnTitle,",
  "  valueTitle, signed, limit, ignoreDiagonal, tooltip(cell), onClick(cell), fileName }):",
  "  rows and columns are the names (columns default to rows), values a list of rows of",
  "  numbers (null where missing), in the order of rows and columns; signed (default: when",
  "  a value is below 0) gives the scale centered on 0; limit fixes the end of the scale;",
  "  ignoreDiagonal: true makes the scale without the cells of a group with itself, which",
  "  are much larger in a neighborhood enrichment or a co-occurrence (else every other cell",
  "  looks pale): use it for those, they then show the end color; cell is { row, column,",
  "  value, rowIndex, columnIndex }; tooltip returns more lines (e.g. the counts) for the",
  "  value under the mouse; onClick gets the cell that the user clicked, outlined.",
  "- chat.bars(element, { labels, values, title, labelTitle, valueTitle, colors,",
  "  tooltip(bar), onClick(bar), fileName }): bar is { label, value, index }.",
  "- chat.scatter(element, { x, y, groups, labels, title, xTitle, yTitle, groupTitle,",
  "  colors, tooltip(point), onClick(point), fileName }): at most 20,000 points (sample",
  "  more); groups colors the points, with a legend; point is { index, x, y, group, label }.",
  "- chat.legend(element, { title, items, onReset, resetText, note }): the colors that the",
  '  plugin gave on the image, items as [{ label, color }] (e.g. { label: "3", color:',
  '  "#e34948" }), with a "Reset colors" button that calls onReset; returns { update,',
  "  destroy }.",
  "Pass the numbers as they are (e.g. the z-scores of Python): the pieces compute the scale.",
  "For example: const heat = chat.heatmap(chartDiv, { rows: result.groups, values: result.z,",
  '  title: "Neighborhood enrichment", rowTitle: "Cluster", columnTitle: "Neighbor cluster",',
  '  valueTitle: "z-score", onClick: (cell) => colorPair(cell.row, cell.column) });',
  "Only for a chart that none of them can show, draw your own, with the same features.",
  "When the plugin colors the cells or points on the image (e.g. the two clusters of a cell",
  "of a heatmap), it keeps their colors from before, shows them with chat.legend, whose",
  "onReset puts the colors from before back; tell the user that Undo in the chat does not",
  "take these colors back.",
  "",
  "When the plugin uses the code or the method of a skill, give the skill to",
  "propose_plugin, and start the code with a comment that says what the plugin is, where",
  "the code comes from (the repository and the commit) and its license, with the",
  "copyright line of its authors (read the LICENSE file of the skill with",
  "read_skill_file), as the license asks. The chat adds a thin bar at the bottom of the",
  `panel itself: "${pluginCredit}"; do not add it again.`,
  "",
  "After the result, tell the user where to find the plugin: in the plugins menu of",
  'TissUUmaps, and in the settings (⚙) of the chat, under "Plugins made by the chat",',
  "where it can be opened, kept for the next sessions, removed or exported as a .js file.",
  "Tell the user that what the plugin changes is not taken back by Undo in the chat. If it",
  "could not run, read why, fix the code and propose it again with the same id (at most",
  "twice). Keep the code clear and commented, in English.",
].join("\n");

/**
 * Runs the propose_plugin tool: checks the plugin, then lets the panel show
 * it to the user and run it if the user says yes
 *
 * @param args - The arguments of the tool call
 * @param context - How to propose a plugin
 * @param options - Optional abort signal, which answers no
 * @returns The result for the model
 */
export async function proposePluginTool(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const text = (value: unknown) =>
    typeof value === "string" ? value.trim() : "";
  const name = text(args["name"]).replace(/\s+/g, " ");
  if (name === "" || name.length > maxName) {
    return `Error: give the plugin a name of at most ${maxName} characters, e.g. "Neighborhood enrichment".`;
  }
  const given = pluginId(args["id"], name);
  if (typeof given !== "string") {
    return `Error: ${given.error}.`;
  }
  // a plugin with the name of one made before replaces it, under its id,
  // so that the list never holds two of them
  const plugins = context.chatPlugins();
  const sameName = plugins.find(
    ({ entry }) =>
      entry.id !== given && entry.name.toLowerCase() === name.toLowerCase(),
  );
  const id =
    plugins.some(({ entry }) => entry.id === given) || sameName === undefined
      ? given
      : sameName.entry.id;
  const description = text(args["description"]);
  if (description === "" || description.length > maxDescription) {
    return `Error: give a description of what the plugin does, of at most ${maxDescription} characters.`;
  }
  const code = typeof args["code"] === "string" ? args["code"] : "";
  if (!/\bexport\s+default\b/.test(code)) {
    return "Error: code is missing or has no export default: give a module, export default function createPlugin(chat) { return { mount(container, stores) { ... } }; }";
  }
  if (code.length > maxCode) {
    return `Error: the code is too long (${code.length} characters, at most ${maxCode}).`;
  }
  let skill;
  if (args["skill"] !== undefined && args["skill"] !== "") {
    const found = findSkill(context.skills, args["skill"]);
    if (typeof found === "string") {
      return found;
    }
    skill = found;
  }
  return context.proposePlugin(
    { id, name, description, code },
    {
      ...(skill === undefined ? {} : { skill }),
      ...(signal === undefined ? {} : { signal }),
    },
  );
}

/** The read_plugin tool, only offered to OpenAI and Anthropic */
export const readPluginDefinition = {
  type: "function",
  function: {
    name: "read_plugin",
    description:
      'Read the current code of a plugin made by the chat (listed in the context under "# Plugins made by the chat"), before you fix or improve it.',
    parameters: {
      type: "object",
      required: ["id"],
      properties: {
        id: {
          type: "string",
          description: 'Its id, e.g. "chat-anhood".',
        },
      },
    },
  },
};

/**
 * Finds a plugin made by the chat by its id, with or without "chat-", or by
 * its name
 *
 * @param plugins - The plugins made by the chat
 * @param given - The id or name that the model gave
 * @returns The plugin, if there is one
 */
function findPlugin(
  plugins: ChatPluginState[],
  given: string,
): ChatPluginState | undefined {
  const id = pluginId(given, given);
  const name = given.trim().toLowerCase();
  return (
    plugins.find(({ entry }) => entry.id === id) ??
    plugins.find(({ entry }) => entry.name.toLowerCase() === name)
  );
}

/**
 * Runs the read_plugin tool: gives the model the current code of a plugin
 * made by the chat
 *
 * @param args - The arguments of the tool call
 * @param context - The plugins made by the chat
 * @returns The result for the model
 */
export function readPluginTool(
  args: ToolArguments,
  context: ToolContext,
): string {
  const given = typeof args["id"] === "string" ? args["id"] : "";
  const plugins = context.chatPlugins();
  const found = findPlugin(plugins, given);
  if (found === undefined) {
    const ids = plugins.map(({ entry }) => entry.id).join(", ");
    return `Error: there is no plugin "${given}" made by the chat. ${ids === "" ? "The chat has made none yet." : `The plugins made by the chat: ${ids}.`}`;
  }
  const { entry } = found;
  return [
    `Done: the plugin "${entry.name}" (id ${entry.id}), made ${entry.made}${entry.origin === undefined ? "" : `, based on the skill ${entry.origin.skill}`}.`,
    `Description: ${entry.description}`,
    "To change it, start from this code, keep every feature, and propose it with the same id and name. Replace the charts that it draws itself (canvas or SVG) with chat.heatmap, chat.bars, chat.scatter or chat.legend.",
    "Code:",
    entry.code,
  ].join("\n");
}

/**
 * The plugins made by the chat, for the context of each turn
 *
 * @param plugins - The plugins
 * @returns The section of the context
 */
export function describePlugins(plugins: ChatPluginState[]): string {
  const lines = plugins.map(
    ({ entry, running, remembered }) =>
      `- ${entry.id}: "${entry.name}", ${entry.description} (${running ? "running" : "not running"}${remembered ? ", remembered" : ""})`,
  );
  return [
    "# Plugins made by the chat",
    ...(lines.length === 0 ? ["None yet."] : lines),
  ].join("\n");
}
