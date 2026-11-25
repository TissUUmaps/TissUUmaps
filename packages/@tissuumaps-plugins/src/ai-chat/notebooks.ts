// Jupyter notebooks (.ipynb) often show how to use the code of a repository.
// Their outputs (images, tables, logs) are most of their size and of no use
// to the model, and they could hold data that was not meant to be shared, so
// notebooks from GitHub are kept without them: only the cells of code and of
// text. The model reads a notebook as plain text, one cell after another.

/** A cell of a notebook, as nbformat 4 writes it */
type NotebookCell = {
  cell_type?: string;
  source?: string | string[];
  outputs?: unknown[];
  execution_count?: number | null;
  attachments?: unknown;
  metadata?: unknown;
};

/** A notebook, as nbformat 4 writes it */
type Notebook = {
  cells?: NotebookCell[];
  metadata?: Record<string, unknown>;
  nbformat?: number;
  nbformat_minor?: number;
};

/**
 * Reads a notebook
 *
 * @param text - The text of the .ipynb file
 * @returns The notebook
 * @throws If it is not a notebook
 */
function readNotebook(text: string): Notebook {
  let notebook: unknown;
  try {
    notebook = JSON.parse(text);
  } catch {
    throw new Error("it is not a notebook (not JSON)");
  }
  if (
    typeof notebook !== "object" ||
    notebook === null ||
    !Array.isArray((notebook as Notebook).cells)
  ) {
    throw new Error("it is not a notebook (it has no cells)");
  }
  return notebook;
}

/**
 * The text of a cell
 *
 * @param cell - The cell
 * @returns Its source, as one text
 */
function cellText(cell: NotebookCell): string {
  return Array.isArray(cell.source)
    ? cell.source.join("")
    : (cell.source ?? "");
}

/**
 * A notebook without its outputs: the cells of code and of text, with no
 * output, execution count or attachment, and only the language of the
 * notebook in its metadata
 *
 * @param text - The text of the .ipynb file
 * @returns The text of the notebook without outputs, still a notebook
 * @throws If it is not a notebook
 */
export function withoutOutputs(text: string): string {
  const notebook = readNotebook(text);
  const language = (
    notebook.metadata?.["language_info"] as { name?: unknown } | undefined
  )?.name;
  const cells = (notebook.cells ?? [])
    .filter(
      (cell) => cell.cell_type === "code" || cell.cell_type === "markdown",
    )
    .map((cell) =>
      cell.cell_type === "code"
        ? {
            cell_type: "code",
            execution_count: null,
            metadata: {},
            outputs: [],
            source: cellText(cell),
          }
        : { cell_type: "markdown", metadata: {}, source: cellText(cell) },
    );
  return JSON.stringify(
    {
      cells,
      metadata:
        typeof language === "string"
          ? { language_info: { name: language } }
          : {},
      nbformat: notebook.nbformat ?? 4,
      nbformat_minor: notebook.nbformat_minor ?? 5,
    },
    null,
    1,
  );
}

/**
 * A notebook as plain text, for the model: its cells one after another,
 * each after a line such as "# %% [code]" (the "percent" format that
 * editors know)
 *
 * @param text - The text of the .ipynb file
 * @returns The cells as text
 * @throws If it is not a notebook
 */
export function notebookText(text: string): string {
  const notebook = readNotebook(text);
  return (notebook.cells ?? [])
    .filter(
      (cell) => cell.cell_type === "code" || cell.cell_type === "markdown",
    )
    .map((cell) => `# %% [${cell.cell_type}]\n${cellText(cell).trimEnd()}`)
    .join("\n\n");
}

/**
 * The code of a notebook, for finding what it imports: its cells of code,
 * without the lines for IPython (e.g. "%load_ext" or "!pip install")
 *
 * @param text - The text of the .ipynb file
 * @returns The code, or "" if it is not a notebook
 */
export function notebookCode(text: string): string {
  try {
    return (readNotebook(text).cells ?? [])
      .filter((cell) => cell.cell_type === "code")
      .map((cell) =>
        cellText(cell)
          .split("\n")
          .filter((line) => !/^\s*[%!]/.test(line))
          .join("\n"),
      )
      .join("\n\n");
  } catch {
    return "";
  }
}
