import type * as hyparquet from "hyparquet";
import { type AsyncBuffer, type FileMetaData, parquetRead } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ParquetUtils } from "./ParquetUtils";

vi.mock("hyparquet", async (importOriginal) => ({
  ...(await importOriginal<typeof hyparquet>()),
  parquetRead: vi.fn(),
}));

type FakeSchemaElement = { name: string; num_children?: number };

function fakeMetadata(
  schema: FakeSchemaElement[],
  numRows: number | bigint = 0,
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
});
