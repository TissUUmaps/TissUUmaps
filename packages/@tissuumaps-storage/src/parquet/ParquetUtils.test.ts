import type * as hyparquet from "hyparquet";
import {
  type AsyncBuffer,
  type FileMetaData,
  type SchemaElement,
  parquetRead,
} from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ParquetUtils } from "./ParquetUtils";

vi.mock("hyparquet", async (importOriginal) => ({
  ...(await importOriginal<typeof hyparquet>()),
  parquetRead: vi.fn(),
}));

type FakeSchemaElement = Pick<SchemaElement, "name"> & Partial<SchemaElement>;

/** Row groups with the given null counts of a column, if listed */
function fakeRowGroups(column: string, nullCounts: (number | undefined)[]) {
  return nullCounts.map((nullCount) => ({
    columns: [
      {
        meta_data: {
          path_in_schema: [column],
          statistics:
            nullCount !== undefined ? { null_count: BigInt(nullCount) } : {},
        },
      },
    ],
  }));
}

function fakeMetadata(
  schema: FakeSchemaElement[],
  numRows: number | bigint = 0,
  rowGroups: ReturnType<typeof fakeRowGroups> = [],
): FileMetaData {
  // The root counts the top-level elements only: an element's children follow
  // it in the schema and are skipped (one level of nesting is enough here)
  let numChildren = 0;
  for (let i = 0; i < schema.length; i += 1 + (schema[i]!.num_children ?? 0)) {
    numChildren++;
  }
  return {
    schema: [{ name: "schema", num_children: numChildren }, ...schema],
    num_rows: BigInt(numRows),
    row_groups: rowGroups,
  } as unknown as FileMetaData;
}

describe("ParquetUtils", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("getNumRows", () => {
    it("reads the row count", () => {
      expect(ParquetUtils.getNumRows(fakeMetadata([], 355))).toBe(355);
    });

    it("rejects a row count beyond the safe integer range", () => {
      const metadata = fakeMetadata([], BigInt(Number.MAX_SAFE_INTEGER) + 1n);
      expect(() => ParquetUtils.getNumRows(metadata)).toThrow(
        "Parquet file has too many rows",
      );
    });
  });

  describe("getColumns", () => {
    it("lists the columns in schema order", () => {
      const metadata = fakeMetadata([
        { name: "geometry" },
        { name: "radius" },
        { name: "__index_level_0__" },
      ]);
      expect(ParquetUtils.getColumns(metadata)).toEqual([
        "geometry",
        "radius",
        "__index_level_0__",
      ]);
    });

    it("lists a nested column once, without its children", () => {
      const metadata = fakeMetadata([
        { name: "position", num_children: 2 },
        { name: "x" },
        { name: "y" },
        { name: "area" },
      ]);
      expect(ParquetUtils.getColumns(metadata)).toEqual(["position", "area"]);
    });

    it("lists no columns for an empty schema", () => {
      expect(ParquetUtils.getColumns(fakeMetadata([]))).toEqual([]);
    });
  });

  describe("readColumnChunks", () => {
    const metadata = fakeMetadata([{ name: "x" }], 3);
    const byteLength = 10;
    const buffer: AsyncBuffer = {
      byteLength,
      slice: (start, end = byteLength) =>
        Promise.resolve(new ArrayBuffer(end - start)),
    };

    it("reads one column and forwards its chunks with their first row", async () => {
      vi.mocked(parquetRead).mockImplementation((options) => {
        options.onChunk?.({
          columnName: "x",
          columnData: [1, 2],
          rowStart: 0,
          rowEnd: 2,
        });
        options.onChunk?.({
          columnName: "x",
          columnData: [3],
          rowStart: 2,
          rowEnd: 3,
        });
        return Promise.resolve();
      });
      const onChunk = vi.fn();
      await ParquetUtils.readColumnChunks(
        buffer,
        metadata,
        "x",
        onChunk,
        () => {},
      );
      expect(parquetRead).toHaveBeenCalledWith(
        expect.objectContaining({ metadata, compressors, columns: ["x"] }),
      );
      expect(onChunk.mock.calls).toEqual([
        [[1, 2], 0],
        [[3], 2],
      ]);
    });

    it("reports the bytes read so far against the file size", async () => {
      vi.mocked(parquetRead).mockImplementation(async (options) => {
        await options.file.slice(0, 4);
        await options.file.slice(4, 10);
      });
      const onProgress = vi.fn();
      await ParquetUtils.readColumnChunks(
        buffer,
        metadata,
        "x",
        () => {},
        onProgress,
      );
      expect(onProgress.mock.calls).toEqual([
        [4, 10],
        [10, 10],
      ]);
    });

    it("rejects when the reader fails", async () => {
      vi.mocked(parquetRead).mockRejectedValue(new Error("corrupt"));
      await expect(
        ParquetUtils.readColumnChunks(
          buffer,
          metadata,
          "x",
          () => {},
          () => {},
        ),
      ).rejects.toThrow("corrupt");
    });
  });

  describe("getTotalNumRows", () => {
    const buffer: AsyncBuffer = {
      byteLength: 0,
      slice: () => Promise.resolve(new ArrayBuffer(0)),
    };

    it("adds up the row counts of the parts", () => {
      expect(
        ParquetUtils.getTotalNumRows([
          { buffer, metadata: fakeMetadata([], 3) },
          { buffer, metadata: fakeMetadata([], 4) },
        ]),
      ).toBe(7);
    });

    it("rejects a total beyond the safe integer range", () => {
      const half = fakeMetadata([], 2 ** 52);
      expect(() =>
        ParquetUtils.getTotalNumRows([
          { buffer, metadata: half },
          { buffer, metadata: half },
          { buffer, metadata: half },
        ]),
      ).toThrow("Parquet table has too many rows");
    });
  });

  describe("getColumnElement", () => {
    it("finds a top-level column by name", () => {
      const metadata = fakeMetadata([
        { name: "position", num_children: 2 },
        { name: "x" },
        { name: "y" },
        { name: "area", type: "DOUBLE" },
      ]);
      expect(ParquetUtils.getColumnElement(metadata, "area")).toEqual({
        name: "area",
        type: "DOUBLE",
      });
      expect(ParquetUtils.getColumnElement(metadata, "x")).toBeUndefined();
    });
  });

  describe("getColumnArrayType", () => {
    it.each([
      [{ type: "INT32" }, Int32Array],
      [{ type: "INT32", converted_type: "UINT_32" }, Uint32Array],
      [
        {
          type: "INT32",
          logical_type: { type: "INTEGER", bitWidth: 32, isSigned: false },
        },
        Uint32Array,
      ],
      [{ type: "INT32", converted_type: "INT_16" }, Int32Array],
      [{ type: "INT64" }, Float64Array],
      [{ type: "DOUBLE" }, Float64Array],
      [{ type: "FLOAT" }, Float32Array],
      [
        { type: "FIXED_LEN_BYTE_ARRAY", converted_type: "DECIMAL" },
        Float64Array,
      ],
      [
        { type: "FIXED_LEN_BYTE_ARRAY", logical_type: { type: "FLOAT16" } },
        Float32Array,
      ],
      [{ type: "BYTE_ARRAY", converted_type: "UTF8" }, undefined],
      [{ type: "INT32", converted_type: "DATE" }, undefined],
      [{ type: "BOOLEAN" }, undefined],
    ] as [Partial<SchemaElement>, unknown][])(
      "reads %j as %o",
      (element, arrayType) => {
        expect(ParquetUtils.getColumnArrayType({ name: "c", ...element })).toBe(
          arrayType,
        );
      },
    );
  });

  describe("isNullable", () => {
    it("adds up the null counts of the row groups", () => {
      expect(
        ParquetUtils.isNullable(
          fakeMetadata([{ name: "c" }], 4, fakeRowGroups("c", [0, 1])),
          "c",
        ),
      ).toBe(true);
      expect(
        ParquetUtils.isNullable(
          fakeMetadata([{ name: "c" }], 4, fakeRowGroups("c", [0, 0])),
          "c",
        ),
      ).toBe(false);
    });

    it("cannot tell if a row group lacks the statistic", () => {
      expect(
        ParquetUtils.isNullable(
          fakeMetadata([{ name: "c" }], 4, fakeRowGroups("c", [0, undefined])),
          "c",
        ),
      ).toBeUndefined();
    });
  });

  describe("readParts", () => {
    function fakePart(numRows: number, byteLength: number) {
      return {
        buffer: {
          byteLength,
          slice: () => Promise.resolve(new ArrayBuffer(0)),
        },
        metadata: fakeMetadata([], numRows),
      };
    }

    it("reads the parts in order with their first row and overall progress", async () => {
      const parts = [fakePart(3, 10), fakePart(2, 30)];
      const reads: [unknown, number][] = [];
      const onProgress = vi.fn();
      await ParquetUtils.readParts(
        parts,
        onProgress,
        (part, rowOffset, onPartProgress) => {
          reads.push([part, rowOffset]);
          onPartProgress(part.buffer.byteLength / 2);
          onPartProgress(part.buffer.byteLength);
          return Promise.resolve();
        },
      );
      expect(reads).toEqual([
        [parts[0], 0],
        [parts[1], 3],
      ]);
      expect(onProgress.mock.calls).toEqual([
        [5, 40],
        [10, 40],
        [25, 40],
        [40, 40],
      ]);
    });
  });

  describe("readColumn", () => {
    const buffer: AsyncBuffer = {
      byteLength: 10,
      slice: () => Promise.resolve(new ArrayBuffer(0)),
    };

    /** Lets the Parquet reader return the given chunks for each file */
    function fakeChunks(chunksByFile: Map<FileMetaData, unknown[]>) {
      vi.mocked(parquetRead).mockImplementation((options) => {
        let rowStart = 0;
        for (const columnData of chunksByFile.get(options.metadata!)!) {
          const length = (columnData as ArrayLike<unknown>).length;
          options.onChunk?.({
            columnName: "c",
            columnData: columnData as unknown[],
            rowStart,
            rowEnd: rowStart + length,
          });
          rowStart += length;
        }
        return Promise.resolve();
      });
    }

    function fakeFile(
      element: Partial<SchemaElement>,
      numRows: number,
      nullCounts: (number | undefined)[] = [undefined],
    ) {
      return fakeMetadata(
        [{ name: "c", ...element }],
        numRows,
        fakeRowGroups("c", nullCounts),
      );
    }

    it("writes the chunks of all parts at their rows", async () => {
      const first = fakeFile({ type: "INT32", repetition_type: "REQUIRED" }, 3);
      const second = fakeFile(
        { type: "INT32", repetition_type: "REQUIRED" },
        2,
      );
      fakeChunks(
        new Map([
          [first, [new Int32Array([1, 2]), new Int32Array([3])]],
          [second, [new Int32Array([4, 5])]],
        ]),
      );
      await expect(
        ParquetUtils.readColumn(
          [
            { buffer, metadata: first },
            { buffer, metadata: second },
          ],
          "c",
          () => {},
        ),
      ).resolves.toStrictEqual(new Int32Array([1, 2, 3, 4, 5]));
    });

    it("reads integers as floats if any part may lack a value", async () => {
      const required = fakeFile(
        { type: "INT32", repetition_type: "REQUIRED" },
        2,
      );
      const withNulls = fakeFile(
        { type: "INT32", repetition_type: "OPTIONAL" },
        2,
        [1],
      );
      fakeChunks(
        new Map([
          [required, [new Int32Array([1, 2])]],
          [withNulls, [[3, null]]],
        ]),
      );
      await expect(
        ParquetUtils.readColumn(
          [
            { buffer, metadata: required },
            { buffer, metadata: withNulls },
          ],
          "c",
          () => {},
        ),
      ).resolves.toStrictEqual(new Float64Array([1, 2, 3, NaN]));
    });

    it("keeps integers next to an empty part without statistics", async () => {
      const values = fakeFile(
        { type: "INT32", repetition_type: "OPTIONAL" },
        2,
        [0],
      );
      const empty = fakeFile(
        { type: "INT32", repetition_type: "OPTIONAL" },
        0,
        [undefined],
      );
      fakeChunks(
        new Map([
          [values, [new Int32Array([1, 2])]],
          [empty, []],
        ]),
      );
      await expect(
        ParquetUtils.readColumn(
          [
            { buffer, metadata: values },
            { buffer, metadata: empty },
          ],
          "c",
          () => {},
        ),
      ).resolves.toStrictEqual(new Int32Array([1, 2]));
    });

    it("keeps integers whose statistics count no nulls", async () => {
      const metadata = fakeFile(
        { type: "INT32", repetition_type: "OPTIONAL" },
        2,
        [0],
      );
      fakeChunks(new Map([[metadata, [[1, 2]]]]));
      await expect(
        ParquetUtils.readColumn([{ buffer, metadata }], "c", () => {}),
      ).resolves.toStrictEqual(new Int32Array([1, 2]));
    });

    it("reads 64-bit integers as floats of safe integers", async () => {
      const metadata = fakeFile(
        { type: "INT64", repetition_type: "REQUIRED" },
        2,
      );
      fakeChunks(new Map([[metadata, [new BigInt64Array([1n, 2n ** 48n])]]]));
      await expect(
        ParquetUtils.readColumn([{ buffer, metadata }], "c", () => {}),
      ).resolves.toStrictEqual(new Float64Array([1, 2 ** 48]));
    });

    it("rejects 64-bit integers beyond the safe integer range", async () => {
      const metadata = fakeFile(
        { type: "INT64", repetition_type: "REQUIRED" },
        1,
      );
      fakeChunks(new Map([[metadata, [new BigInt64Array([2n ** 60n])]]]));
      await expect(
        ParquetUtils.readColumn([{ buffer, metadata }], "c", () => {}),
      ).rejects.toThrow("Value is not a safe integer");
    });

    it("reads other columns as plain arrays with null for missing values", async () => {
      const metadata = fakeFile(
        { type: "BYTE_ARRAY", converted_type: "UTF8" },
        3,
      );
      fakeChunks(new Map([[metadata, [["a", null], ["b"]]]]));
      await expect(
        ParquetUtils.readColumn([{ buffer, metadata }], "c", () => {}),
      ).resolves.toStrictEqual(["a", null, "b"]);
    });

    it("rejects a column the table does not have", async () => {
      await expect(
        ParquetUtils.readColumn(
          [{ buffer, metadata: fakeFile({ type: "INT32" }, 1) }],
          "missing",
          () => {},
        ),
      ).rejects.toThrow('Column "missing" not found in Parquet file');
    });
  });
});
