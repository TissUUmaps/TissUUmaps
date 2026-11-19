import { maxCharts } from "./charts";
import { languageProblem } from "./code-language";
import { findSkill } from "./skill-tools";
import type { ToolArguments, ToolContext } from "./types";

// The strong models can write their own JavaScript or Python for what their
// tools and the summary cannot answer, e.g. the mean of a gene in each
// cluster. The code runs in a sandbox like the scripts of skills (Python in
// Pyodide, see python-sandbox.ts), and the user sees it in the chat before it
// runs.

/** The longest title of the code, as the chat shows it, in characters */
const maxTitle = 80;

/** The languages that run_code runs */
const languages = ["javascript", "python"] as const;

/** The run_code tool, only offered to OpenAI and Anthropic */
export const runCodeDefinition = {
  type: "function",
  function: {
    name: "run_code",
    description:
      "Run JavaScript or Python that you write, to compute anything on the data of the tables that the summary does not answer: " +
      "statistics, comparisons, correlations, thresholds, spatial analyses with the coordinates. No skill is needed. " +
      "It runs in a sandbox, where it can read the data of the project but not change it; the user sees its code. " +
      "Returns what the code returns, and what it logs or prints; in Python, the matplotlib figures it draws are shown to the user.",
    parameters: {
      type: "object",
      required: ["title", "code"],
      properties: {
        title: {
          type: "string",
          description:
            'What the code computes, in a few words, shown to the user, e.g. "cells per cluster".',
        },
        language: {
          type: "string",
          enum: languages,
          description:
            'The language of the code: "javascript" (the default) or "python".',
        },
        code: {
          type: "string",
          description:
            "JavaScript: a module, export default async function (tm) { ... }. " +
            "Python: a function async def main(tm): ... Either returns a small result.",
        },
        skill: {
          type: "string",
          description:
            "Only to use the files of a skill that is on, e.g. one from GitHub: its name. Python can then import its modules, and tm.read_file (tm.readFile) reads its files.",
        },
      },
    },
  },
};

/** The save_computed_table tool, only offered to OpenAI and Anthropic */
export const saveComputedTableDefinition = {
  type: "function",
  function: {
    name: "save_computed_table",
    description:
      'Keep a table computed by the chat (e.g. "cells (chat)"), which is temporary, as a CSV file next to the project file, ' +
      "after the user confirms; or download it, when the project was not opened from a connected folder. " +
      "Only when the user asks to keep or save it.",
    parameters: {
      type: "object",
      required: ["table"],
      properties: {
        table: {
          type: "string",
          description: 'The name of the table, e.g. "cells (chat)".',
        },
        file_name: {
          type: "string",
          description:
            'Only if the user names the file, e.g. "neighbors_cluster3.csv". Without it, the name says the table, its columns and the day.',
        },
      },
    },
  },
};

/** The look_at_chart tool, only offered to the services that see images */
export const lookAtChartDefinition = {
  type: "function",
  function: {
    name: "look_at_chart",
    description:
      'Look at a chart that Python drew in this chat, e.g. "Chart 2": its image comes with the result, for this turn only. ' +
      "Use it when the user asks you to look at a chart or what it shows, or when its description is not enough to answer. " +
      "Each image costs about a thousand tokens.",
    parameters: {
      type: "object",
      properties: {
        chart: {
          type: "integer",
          description:
            'The number of the chart, as the chat shows it ("Chart 2" is 2). Without it: the last chart.',
        },
      },
    },
  },
};

/** What the strong models are told about run_code, with the fixed instructions */
export const codeInstructions = [
  "# Your own code",
  "With run_code you can compute anything on the data of the tables yourself: you write",
  "JavaScript or Python, and it runs at once. You need no skill for it. Use it whenever a",
  "question needs a computation that the summary does not answer, e.g.:",
  "- statistics: counts, sums, means, medians, standard deviations, quantiles, for all the",
  "  rows or for each group (e.g. each cluster or cell type);",
  "- comparisons between groups, columns or genes, e.g. which genes are higher in one",
  "  cluster than in the others, with a statistical test;",
  "- correlations between columns or genes;",
  "- thresholds: e.g. how many cells have a value above a limit, and in which groups;",
  "- spatial analyses with the coordinates of the cells or points: distances, neighbors,",
  "  density, which groups lie near each other. The summary gives the coordinates of each",
  '  object ("coordinates (x, y)"), e.g. x and y, geometry[x] and geometry[y], or the two',
  '  columns of the matrix obsm/spatial, "obsm/spatial[0]" and "obsm/spatial[1]": read',
  '  exactly those names; "obsm/spatial" alone is a matrix and reads nothing.',
  "Skills are useful but not needed: when a skill that is on does the task, use it, since it",
  "holds the method the user wants; otherwise write the code yourself. When the user asks",
  "what analyses are possible, tell them all this, and never that a skill is needed first.",
  "",
  "JavaScript or Python:",
  '- JavaScript (language "javascript", the default) starts at once. Use it for counts,',
  "  sums, means and other simple computations. It has no libraries: write what you need.",
  '- Python (language "python") has numpy, scipy, pandas, scikit-learn, statsmodels and the',
  "  other packages of Pyodide. Use it for statistical tests (e.g. scipy.stats), models,",
  "  clustering, or neighbors with scipy.spatial.cKDTree. The first Python code of a chat",
  "  takes some seconds to start, and the first time also to download Python (tens of MB):",
  "  do not use it for what JavaScript does as well.",
  "",
  "JavaScript must be a module: export default async function (tm) { ... }. Python must",
  "define async def main(tm): ... With tm it can read the data:",
  "- await tm.tables(): the tables, as [{ id, name }] (dicts in Python)",
  "- await tm.columns(table): the columns of a table, as [{ name, matrix }]; coordinates is",
  "  true on the columns of the x and the y of the items (or on their matrix). A matrix, such",
  '  as "X" or "obsm/spatial", has matrix true, its size, its first columns (e.g.',
  '  "obsm/spatial[0]", "X[EPCAM]") and how to read them: one column at a time, never by the',
  '  name of the matrix alone; await tm.columns(table, "X") lists all its columns',
  '- await tm.read(table, column): the values of a column (e.g. "obs/leiden" or "X[EPCAM]"),',
  "  one per row, in the order of the rows of the table, so that the values of two columns",
  "  of the same table belong together row by row; in Python, a numpy array (of objects,",
  "  for text such as cluster names)",
  "- tm.project: a copy of the app state, the same JSON as in the context",
  "- with skill (the name of a skill that is on): await tm.readFile(path) in JavaScript,",
  "  await tm.read_file(path) in Python, reads a file of the skill; Python imports its modules",
  "- tm.progress(text): shows how far it is, for long computations",
  "- await tm.download(name, text): gives the user a file of text (.csv, .tsv, .txt, .json",
  '  or .md, e.g. "zscores.csv" with frame.to_csv()), which the chat shows with Save; only',
  "  when the user asks for a file, e.g. of a table that is not one value per row",
  "- await tm.saveColumn(table, name, values) in JavaScript, await tm.save_column(table,",
  "  name, values) in Python: saves a result as a column, one value per row of the table, in",
  "  the order of tm.read (or with the item IDs of the values: { ids } in JavaScript, ids=",
  '  in Python, for some of the items only), into the table "<table> (chat)", which the',
  "  chat makes for it",
  "",
  "# Results on the image",
  "To show a result on the image, e.g. to color the cells by the number of their",
  "neighbors, or to show only the cells near a cluster, save it with tm.saveColumn (or",
  "tm.save_column) in the same code, then use your tools with that column like any other,",
  "e.g. color_by_column with its name: numbers get a color scale, text gets one color per",
  'group. To filter (show_only_groups), save groups as text, e.g. "near" and "far". A',
  "saved column never changes the data of the project, and Undo takes it back. It is",
  "temporary: it is lost when the page is reloaded, unless the user keeps the table with",
  "save_computed_table, which writes it as a CSV file next to the project file (or",
  "downloads it); if the user names the file, give that name as file_name. Say so when",
  "you save one. At most 20 columns can be saved in all.",
  "",
  "Use the table ids or names and the column names of the summary, or list them with",
  "tm.tables() and tm.columns() first. Tell the user which method you used, e.g. which",
  "test. The code may run for at most two minutes (in Python, the start and the download do",
  "not count): in big tables go over the rows only a few times, and for neighbors use a",
  "grid or a k-d tree instead of comparing every pair.",
  "",
  "# Groups named by numbers",
  'Groups whose names are numbers written as text, e.g. the leiden clusters "0", "1", …',
  '"15", are numbers to the user: always sort them as numbers (0, 1, 2, …, 9, 10, 11), never',
  "as text (0, 1, 10, 11, …, 2), in results, tables, files, charts (e.g. the rows and columns",
  "of a heatmap) and your answers. Results of a skill or a library often come in text order",
  "(e.g. from numpy.unique, sorted() or pandas): put them in number order before you return,",
  "save or draw them. In Python, e.g. order = sorted(names, key=lambda n: (0, float(n)) if",
  "str(n).replace('.', '', 1).lstrip('-').isdigit() else (1, str(n))), then",
  "frame.reindex(index=order, columns=order); in JavaScript, names.sort((a, b) =>",
  "a.localeCompare(b, undefined, { numeric: true })).",
  "",
  "# Charts",
  "In Python you can draw charts with matplotlib (import matplotlib.pyplot as plt): every",
  'figure still open when the code ends is shown to the user in the chat ("Chart 1", "Chart',
  '2", …), as an image they can copy and save. Usually you do not see it: you get a short',
  "description (titles, axes, what each plot holds), so say what it shows only from that",
  'and from the numbers. If the context says "Charts: you see them", the images come with',
  'the result of the code. When the user asks you to look at a chart (e.g. "look at the',
  'chart and tell me what you see"), call look_at_chart with its number: you see the image',
  "in this turn only. Never say that you see a chart whose image you did not get. Give each",
  "chart a title and labeled axes; do not call plt.show() or plt.savefig(). Draw one when",
  `the user asks for a chart, or when it shows a result much better than numbers; at most`,
  `${maxCharts} per code. JavaScript cannot draw charts. Also give the main numbers in your answer.`,
  "",
  "What you cannot do yet: read the pixels of images. The code has no internet:",
  "it cannot download anything, and in Python only the packages of Pyodide can be imported",
  "(no pip or micropip, and no packages such as squidpy or scanpy). It cannot change the",
  "data of the project or save files itself: it can only save columns with tm.saveColumn,",
  "and give the user files with tm.download. When a",
  "request needs one of these, say so, and offer what you can do instead.",
  "Return a small result (counts, summaries, a small DataFrame, at most a few hundred",
  "values), never whole columns: it comes back to you as text, and what the code logs with",
  "console.log or print comes along. The code runs in a sandbox: no network, and it cannot",
  "change the project. It is stopped after two minutes, or when the user clicks Stop. The",
  "user sees the code: keep it short and clear. If it fails, fix it once; if it fails",
  "again, tell the user why. Never make up a result that the code did not return.",
].join("\n");

/**
 * Runs the run_code tool: checks the code, then lets the panel run it in the
 * sandbox
 *
 * @param args - The arguments of the tool call
 * @param context - How to run code
 * @param options - Optional abort signal, which stops the code
 * @returns The result for the model
 */
export async function runCodeTool(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const language = args["language"] ?? "javascript";
  if (!languages.some((known) => known === language)) {
    return `Error: run_code runs "javascript" or "python", not ${JSON.stringify(language)}.`;
  }
  const python = language === "python";
  const code = args["code"];
  if (typeof code !== "string" || code.trim() === "") {
    return python
      ? "Error: code is missing: give Python with a function async def main(tm): ..."
      : "Error: code is missing: give a JavaScript module, export default async function (tm) { ... }.";
  }
  const mismatch = languageProblem(code, python ? "python" : "javascript");
  if (mismatch !== undefined) {
    return `Error: ${mismatch}.`;
  }
  let skill;
  if (args["skill"] !== undefined && args["skill"] !== "") {
    const found = findSkill(context.skills, args["skill"]);
    if (typeof found === "string") {
      return found;
    }
    skill = found;
  }
  const given = args["title"];
  const title =
    (typeof given === "string" ? given.trim().replace(/\s+/g, " ") : "") ||
    "code";
  return context.runCode(
    title.length > maxTitle ? `${title.slice(0, maxTitle - 1)}…` : title,
    code,
    python ? "python" : "javascript",
    {
      signal,
      progress: context.progress,
      changes: context.changes,
      ...(skill === undefined ? {} : { skill }),
    },
  );
}

/**
 * Runs the look_at_chart tool: the panel gives the image of the chart with
 * the result
 *
 * @param args - The arguments of the tool call
 * @param context - How to look at a chart
 * @returns The result for the model
 */
export function lookAtChartTool(
  args: ToolArguments,
  context: ToolContext,
): string {
  const chart = args["chart"];
  if (chart === undefined || chart === null) {
    return context.lookAtChart();
  }
  const number = typeof chart === "string" ? Number(chart) : chart;
  if (typeof number !== "number" || !Number.isInteger(number)) {
    return 'Error: give the number of the chart, e.g. 2 for "Chart 2".';
  }
  return context.lookAtChart(number);
}

/**
 * Runs the save_computed_table tool: lets the panel keep a table computed by
 * the chat as a CSV file, or download it
 *
 * @param args - The arguments of the tool call
 * @param context - How to keep a table
 * @returns The result for the model
 */
export function saveComputedTableTool(
  args: ToolArguments,
  context: ToolContext,
): Promise<string> {
  const table = args["table"];
  if (typeof table !== "string" || table.trim() === "") {
    return Promise.resolve(
      'Error: give the name of the table, e.g. "cells (chat)".',
    );
  }
  const fileName = args["file_name"];
  return context.keepComputedTable(
    table.trim(),
    typeof fileName === "string" && fileName.trim() !== "" ? { fileName } : {},
  );
}
