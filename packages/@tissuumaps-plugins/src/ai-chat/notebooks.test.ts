import { describe, expect, it } from "vitest";

import { notebookCode, notebookText, withoutOutputs } from "./notebooks";

/** A notebook with outputs, as Jupyter saves one */
const notebook = JSON.stringify({
  cells: [
    {
      cell_type: "markdown",
      metadata: {},
      source: ["# Example\n", "How to use it."],
    },
    {
      cell_type: "code",
      execution_count: 3,
      metadata: { scrolled: true },
      outputs: [
        { output_type: "display_data", data: { "image/png": "iVBORw0KG..." } },
      ],
      source: [
        "%load_ext autoreload\n",
        "from anhood import anhood\n",
        "!pip install squidpy",
      ],
    },
    { cell_type: "raw", metadata: {}, source: "raw text" },
  ],
  metadata: {
    kernelspec: { name: "omics" },
    language_info: { name: "python", version: "3.10.14" },
  },
  nbformat: 4,
  nbformat_minor: 4,
});

describe("notebooks", () => {
  it("removes the outputs, and keeps the cells of code and of text", () => {
    const kept = JSON.parse(withoutOutputs(notebook)) as {
      cells: Record<string, unknown>[];
      metadata: unknown;
      nbformat: number;
    };
    expect(kept.cells).toEqual([
      {
        cell_type: "markdown",
        metadata: {},
        source: "# Example\nHow to use it.",
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source:
          "%load_ext autoreload\nfrom anhood import anhood\n!pip install squidpy",
      },
    ]);
    expect(kept.metadata).toEqual({ language_info: { name: "python" } });
    expect(kept.nbformat).toBe(4);
    expect(withoutOutputs(notebook)).not.toContain("iVBORw0KG");
  });

  it("gives a notebook as text, and its code without the lines for IPython", () => {
    expect(notebookText(notebook)).toBe(
      "# %% [markdown]\n# Example\nHow to use it.\n\n" +
        "# %% [code]\n%load_ext autoreload\nfrom anhood import anhood\n!pip install squidpy",
    );
    expect(notebookCode(notebook)).toBe("from anhood import anhood");
  });

  it("refuses what is not a notebook", () => {
    expect(() => withoutOutputs("{not json")).toThrow(
      "it is not a notebook (not JSON)",
    );
    expect(() => notebookText('{"cells": 3}')).toThrow(
      "it is not a notebook (it has no cells)",
    );
    expect(notebookCode("[]")).toBe("");
  });
});
