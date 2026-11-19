import { describe, expect, it } from "vitest";

import type { TableData } from "@tissuumaps/core";

import { describeMatrixReading, findCoordinates } from "./tables";

/** A fake table that only knows the columns of its matrices */
function tableWith(matrices: Record<string, string[]>): TableData {
  return {
    suggestColumnQueries: (query: string) =>
      Promise.resolve(
        (matrices[query] ?? []).map((name) => ({ query: `${query}[${name}]` })),
      ),
  } as unknown as TableData;
}

/** Columns that are not matrices */
function plain(...names: string[]) {
  return names.map((name) => ({ name, matrix: false }));
}

describe("findCoordinates", () => {
  it("takes the first two columns of obsm/spatial first", async () => {
    const data = tableWith({ "obsm/spatial": ["0", "1", "2"] });
    expect(
      await findCoordinates(data, [
        ...plain("obs/x", "obs/y"),
        { name: "obsm/spatial", matrix: true },
      ]),
    ).toEqual(["obsm/spatial[0]", "obsm/spatial[1]"]);
  });

  it("finds two columns named alike", async () => {
    const data = tableWith({});
    expect(
      await findCoordinates(
        data,
        plain("radius", "geometry[x]", "geometry[y]"),
      ),
    ).toEqual(["geometry[x]", "geometry[y]"]);
    expect(
      await findCoordinates(data, plain("cell_id", "x_centroid", "y_centroid")),
    ).toEqual(["x_centroid", "y_centroid"]);
    expect(await findCoordinates(data, plain("obs/X", "obs/Y"))).toEqual([
      "obs/X",
      "obs/Y",
    ]);
    expect(
      await findCoordinates(data, plain("centroid_x", "centroid_y")),
    ).toEqual(["centroid_x", "centroid_y"]);
  });

  it("does not take embeddings or words that only contain an x", async () => {
    const data = tableWith({ "obsm/X_umap": ["0", "1"] });
    expect(
      await findCoordinates(data, [
        ...plain("umap_x", "umap_y", "max", "may", "x_only"),
        { name: "obsm/X_umap", matrix: true },
      ]),
    ).toBeUndefined();
  });
});

describe("describeMatrixReading", () => {
  it("names all the columns of a small matrix, and some of a big one", () => {
    expect(
      describeMatrixReading("obsm/spatial", [
        "obsm/spatial[0]",
        "obsm/spatial[1]",
      ]),
    ).toBe(
      '"obsm/spatial" is a matrix with 2 columns: read "obsm/spatial[0]" and "obsm/spatial[1]", one at a time',
    );
    expect(
      describeMatrixReading("X", ["X[A]", "X[B]", "X[C]", "X[D]", "X[E]"]),
    ).toBe(
      '"X" is a matrix with 5 columns: read them one at a time, e.g. "X[A]"; tm.columns(table, "X") lists them all',
    );
  });
});
