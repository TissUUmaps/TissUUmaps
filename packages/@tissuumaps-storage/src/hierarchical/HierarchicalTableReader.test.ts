import { describe, expect, it } from "vitest";

import type {
  HierarchicalStore,
  HierarchicalStoreDataType,
  HierarchicalStoreNode,
  HierarchicalStoreValues,
} from "./HierarchicalStore";
import { HierarchicalTableReader } from "./HierarchicalTableReader";

type MemoryNode =
  | {
      kind: "group";
      attrs: Record<string, unknown>;
      children: Record<string, MemoryNode>;
    }
  | {
      kind: "array";
      values: HierarchicalStoreValues;
      shape: number[];
      dataType: HierarchicalStoreDataType;
    };

function group(
  children: Record<string, MemoryNode>,
  attrs: Record<string, unknown> = {},
): MemoryNode {
  return { kind: "group", attrs, children };
}

function array(
  values: HierarchicalStoreValues,
  shape: number[] = [values.length],
  dataType?: HierarchicalStoreDataType,
): MemoryNode {
  let inferredDataType: HierarchicalStoreDataType = "integer";
  if (values instanceof Float32Array || values instanceof Float64Array) {
    inferredDataType = "float";
  } else if (typeof values[0] === "string") {
    inferredDataType = "string";
  } else if (typeof values[0] === "boolean") {
    inferredDataType = "boolean";
  }
  return {
    kind: "array",
    values,
    shape,
    dataType: dataType ?? inferredDataType,
  };
}

function dataFrame(
  index: string,
  columns: Record<string, MemoryNode>,
): MemoryNode {
  return group(columns, { "encoding-type": "dataframe", _index: index });
}

function categorical(
  codes: HierarchicalStoreValues,
  categories: HierarchicalStoreValues,
): MemoryNode {
  return group(
    { codes: array(codes), categories: array(categories) },
    { "encoding-type": "categorical" },
  );
}

function sparse(
  encodingType: "csc_matrix" | "csr_matrix",
  shape: [number, number],
  data: HierarchicalStoreValues,
  indices: HierarchicalStoreValues,
  indptr: HierarchicalStoreValues,
): MemoryNode {
  return group(
    { data: array(data), indices: array(indices), indptr: array(indptr) },
    { "encoding-type": encodingType, shape },
  );
}

/** A {@link HierarchicalStore} over nested objects, with row-major 2-D arrays */
class MemoryStore implements HierarchicalStore {
  closed = false;
  private readonly _root: MemoryNode;

  constructor(root: MemoryNode) {
    this._root = root;
  }

  get(
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<HierarchicalStoreNode | null> {
    if (options?.signal?.aborted) {
      return Promise.reject(options.signal.reason as Error);
    }
    let node: MemoryNode | undefined = this._root;
    for (const name of path.split("/").filter((name) => name !== "")) {
      node = node.kind === "group" ? node.children[name] : undefined;
      if (node === undefined) {
        return Promise.resolve(null);
      }
    }
    if (node.kind === "group") {
      return Promise.resolve({
        kind: "group",
        attrs: node.attrs,
        keys: Object.keys(node.children),
      });
    }
    const { values, shape, dataType } = node;
    return Promise.resolve({
      kind: "array",
      shape,
      dataType,
      read: () => Promise.resolve(values),
      slice: (ranges) => {
        const [rowRange, columnRange] = ranges;
        if (shape.length === 1) {
          const [start, end] = rowRange ?? [0, values.length];
          return Promise.resolve(values.slice(start, end));
        }
        const column = columnRange![0];
        return Promise.resolve(
          Array.from(
            { length: shape[0]! },
            (_, row) => values[row * shape[1]! + column],
          ),
        );
      },
    });
  }

  close(): void {
    this.closed = true;
  }
}

function annData(children: Record<string, MemoryNode>): MemoryNode {
  return group(
    {
      obs: dataFrame("_index", {
        _index: array(["c0", "c1", "c2"]),
        area: array(new Float32Array([1.5, 2.5, 3.5])),
        ids: array(new BigInt64Array([1n, 2n, 3n])),
        cell_type: categorical(new Int8Array([0, -1, 1]), ["T", "B"]),
        batch: categorical(
          new Int8Array([1, 0, -1]),
          new Float64Array([10, 20]),
        ),
        cluster: categorical(
          new Int8Array([1, 1, 0]),
          new BigInt64Array([5n, 7n]),
        ),
        score: group(
          {
            values: array(new Int32Array([1, 2, 3])),
            mask: array(new Uint8Array([0, 1, 0])),
          },
          { "encoding-type": "nullable-integer" },
        ),
        flag: group(
          {
            values: array([true, false, true]),
            mask: array([false, false, true]),
          },
          { "encoding-type": "nullable-boolean" },
        ),
        label: group(
          {
            values: array(["a", "b", "c"]),
            mask: array([false, true, false]),
          },
          { "encoding-type": "nullable-string-array" },
        ),
      }),
      var: dataFrame("gene", { gene: array(["CD3", "CD4"]) }),
      X: sparse(
        "csc_matrix",
        [3, 2],
        new Float32Array([5, 7]),
        new Int32Array([0, 2]),
        new Int32Array([0, 2, 2]),
      ),
      obsm: group({
        spatial: array(new Float64Array([0, 1, 2, 3, 4, 5]), [3, 2]),
      }),
      ...children,
    },
    { "encoding-type": "anndata" },
  );
}

async function openReader(root: MemoryNode): Promise<HierarchicalTableReader> {
  return await HierarchicalTableReader.open(new MemoryStore(root));
}

describe("HierarchicalTableReader", () => {
  describe("open", () => {
    it("lists arrays and encoded groups in path order", async () => {
      const reader = await openReader(
        group({
          b: array(new Int32Array([1, 2])),
          a: group({
            matrix: array(new Float32Array(4), [2, 2]),
            cube: array(new Float32Array(8), [2, 2, 2]),
            scalar: array(new Float32Array(1), []),
            compound: array([{}, {}], [2], "other"),
            __categories: group({ labels: array(["x", "y"]) }),
            "bracketed[0]": array(new Int32Array(2)),
          }),
          csc: sparse("csc_matrix", [2, 3], [], [], [0, 0, 0, 0]),
          noShape: group({}, { "encoding-type": "csc_matrix" }),
        }),
      );
      expect(reader.columns).toEqual([
        { kind: "matrix", path: "a/matrix", numColumns: 2 },
        { kind: "dataset", path: "b" },
        { kind: "matrix", path: "csc", numColumns: 3 },
      ]);
    });

    it("takes the row count from the obs index of an AnnData object", async () => {
      const reader = await openReader(annData({}));
      expect(reader.numRows).toBe(3);
    });

    it("takes the row count from a nullable string obs index", async () => {
      const reader = await openReader(
        group(
          {
            obs: dataFrame("_index", {
              _index: group(
                {
                  values: array(["c0", "c1"]),
                  mask: array([false, false]),
                },
                { "encoding-type": "nullable-string-array" },
              ),
            }),
            a: array(new Int32Array(5)),
          },
          { "encoding-type": "anndata" },
        ),
      );
      expect(reader.numRows).toBe(2);
    });

    it("falls back to the length of the first column", async () => {
      const reader = await openReader(
        group({ b: array(new Int32Array(2)), a: array(new Int32Array(4)) }),
      );
      expect(reader.numRows).toBe(4);
    });

    it("rejects stores with more than one AnnData object", async () => {
      const store = new MemoryStore(
        group({ tables: group({ a: annData({}), b: annData({}) }) }),
      );
      await expect(HierarchicalTableReader.open(store)).rejects.toThrow(
        '2 AnnData objects ("/tables/a", "/tables/b")',
      );
      expect(store.closed).toBe(true);
    });

    it("rejects an obs index that is not a column", async () => {
      await expect(
        openReader(
          group(
            {
              obs: dataFrame("_index", { _index: group({}) }),
              a: array(new Int32Array(2)),
            },
            { "encoding-type": "anndata" },
          ),
        ),
      ).rejects.toThrow('"obs/_index" is a group, not a column');
    });

    it("rejects stores without columns and closes them", async () => {
      const store = new MemoryStore(group({ empty: group({}) }));
      await expect(HierarchicalTableReader.open(store)).rejects.toThrow(
        "No columns",
      );
      expect(store.closed).toBe(true);
    });

    it("gives expression matrices the selectors of their var index", async () => {
      const reader = await openReader(
        annData({
          layers: group({
            counts: array(new Float32Array(6), [3, 2]),
          }),
          raw: group({
            X: array(new Float32Array(9), [3, 3]),
            var: dataFrame("gene", { gene: array(["CD3", "CD4", "CD8"]) }),
          }),
          varm: group({ pca: array(new Float32Array(4), [2, 2]) }),
        }),
      );
      const selectors = Object.fromEntries(
        reader.columns.flatMap((column) =>
          column.kind === "matrix" ? [[column.path, column.selectors]] : [],
        ),
      );
      expect(selectors).toEqual({
        X: ["CD3", "CD4"],
        "layers/counts": ["CD3", "CD4"],
        "obsm/spatial": undefined,
        "raw/X": ["CD3", "CD4", "CD8"],
        "varm/pca": undefined,
      });
    });

    it("selects numeric and duplicate variable names by index", async () => {
      const reader = await openReader(
        annData({
          var: dataFrame("gene", { gene: array(["7157", "7157"]) }),
        }),
      );
      const x = reader.columns.find((column) => column.path === "X");
      expect(x).toMatchObject({ selectors: ["0", "1"] });
    });

    it("leaves a matrix unnamed if its var index has another length", async () => {
      const reader = await openReader(
        annData({
          var: dataFrame("gene", { gene: array(["CD3", "CD4", "CD8"]) }),
        }),
      );
      const x = reader.columns.find((column) => column.path === "X");
      expect(x).not.toHaveProperty("selectors");
    });

    it("names the matrices of an AnnData object below the root", async () => {
      const reader = await openReader(
        group({ tables: group({ adata: annData({}) }) }),
      );
      expect(reader.numRows).toBe(3);
      expect(
        reader.columns.find((column) => column.path === "tables/adata/X"),
      ).toMatchObject({ selectors: ["CD3", "CD4"] });
    });
  });

  describe("readColumn", () => {
    it("reads numeric and string arrays", async () => {
      const reader = await openReader(annData({}));
      expect(await reader.readColumn("obs/area")).toEqual(
        new Float32Array([1.5, 2.5, 3.5]),
      );
      expect(await reader.readColumn("obs/_index")).toEqual(["c0", "c1", "c2"]);
    });

    it("reads 64-bit integers as numbers", async () => {
      const reader = await openReader(annData({}));
      expect(await reader.readColumn("obs/ids")).toEqual(
        new Float64Array([1, 2, 3]),
      );
    });

    it("rejects 64-bit integers outside the safe integer range", async () => {
      const reader = await openReader(
        group({ big: array(new BigInt64Array([2n ** 60n])) }),
      );
      await expect(reader.readColumn("big")).rejects.toThrow();
    });

    it("reads one column of a 2-D array", async () => {
      const reader = await openReader(annData({}));
      expect(await reader.readColumn("obsm/spatial[1]")).toEqual([1, 3, 5]);
    });

    it("decodes string, numeric and 64-bit integer categoricals", async () => {
      const reader = await openReader(annData({}));
      expect(await reader.readColumn("obs/cell_type")).toEqual(["T", "", "B"]);
      expect(await reader.readColumn("obs/batch")).toEqual(
        new Float64Array([20, 10, NaN]),
      );
      expect(await reader.readColumn("obs/cluster")).toEqual(
        new Float64Array([7, 7, 5]),
      );
    });

    it("decodes nullable integers, booleans and strings", async () => {
      const reader = await openReader(annData({}));
      expect(await reader.readColumn("obs/score")).toEqual(
        new Float64Array([1, NaN, 3]),
      );
      expect(await reader.readColumn("obs/flag")).toEqual(
        new Float64Array([1, 0, NaN]),
      );
      expect(await reader.readColumn("obs/label")).toEqual(["a", "", "c"]);
    });

    it("reads columns of a CSC matrix, including empty ones", async () => {
      const reader = await openReader(annData({}));
      expect(await reader.readColumn("X[0]")).toEqual(
        new Float64Array([5, 0, 7]),
      );
      expect(await reader.readColumn("X[1]")).toEqual(new Float64Array(3));
    });

    it("reads CSC matrices with 64-bit integer data and indices", async () => {
      const reader = await openReader(
        group({
          counts: sparse(
            "csc_matrix",
            [2, 1],
            new BigInt64Array([4n]),
            new BigInt64Array([1n]),
            new BigInt64Array([0n, 1n]),
          ),
        }),
      );
      expect(await reader.readColumn("counts[0]")).toEqual(
        new Float64Array([0, 4]),
      );
    });

    it("rejects CSR matrices", async () => {
      const reader = await openReader(
        annData({
          layers: group({
            csr: sparse("csr_matrix", [3, 2], [], [], [0, 0, 0, 0]),
          }),
        }),
      );
      await expect(reader.readColumn("layers/csr[0]")).rejects.toThrow("CSR");
    });

    it("reads matrix columns by variable name", async () => {
      const reader = await openReader(annData({}));
      expect(await reader.readColumn("X[CD3]")).toEqual(
        new Float64Array([5, 0, 7]),
      );
      expect(await reader.readColumn("X[cd3]")).toEqual(
        new Float64Array([5, 0, 7]),
      );
    });

    it("reads the right column of duplicate variable names", async () => {
      const reader = await openReader(
        annData({
          var: dataFrame("gene", { gene: array(["MATR3", "MATR3"]) }),
        }),
      );
      expect(await reader.readColumn("X[1]")).toEqual(new Float64Array(3));
      await expect(reader.readColumn("X[MATR3]")).rejects.toThrow();
    });

    it("checks the column length against the number of table rows", async () => {
      const reader = await openReader(annData({}));
      await expect(
        reader.readColumn("var/gene", { numRows: 3 }),
      ).rejects.toThrow("has 2 rows, but the table has 3");
      expect(await reader.readColumn("var/gene")).toEqual(["CD3", "CD4"]);
    });

    it("rejects queries that address no column", async () => {
      const reader = await openReader(annData({}));
      await expect(reader.readColumn("obs/nope")).rejects.toThrow();
      await expect(reader.readColumn("obs")).rejects.toThrow();
      await expect(reader.readColumn("X")).rejects.toThrow();
      await expect(reader.readColumn("X[2]")).rejects.toThrow();
      await expect(reader.readColumn("obs/area[0]")).rejects.toThrow();
    });

    it("rejects an aborted signal", async () => {
      const reader = await openReader(annData({}));
      const signal = AbortSignal.abort();
      await expect(reader.readColumn("obs/area", { signal })).rejects.toBe(
        signal.reason,
      );
    });
  });

  describe("readRange", () => {
    it("returns the range of numeric columns", async () => {
      const reader = await openReader(annData({}));
      expect(await reader.readRange("obs/area")).toEqual([1.5, 3.5]);
      expect(await reader.readRange("obs/batch")).toEqual([10, 20]);
    });

    it("returns undefined for string and constant columns", async () => {
      const reader = await openReader(
        annData({ constant: array(new Float32Array([1, 1, 1])) }),
      );
      expect(await reader.readRange("obs/_index")).toBeUndefined();
      expect(await reader.readRange("constant")).toBeUndefined();
    });
  });
});
