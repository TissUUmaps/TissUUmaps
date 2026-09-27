import type { FileMetaData } from "hyparquet";
import { describe, expect, it } from "vitest";

import { PandasMetadataUtils } from "./PandasMetadataUtils";

function fakePandasMetadata(
  pandas?: string,
  columnNames: string[] = [],
): FileMetaData {
  return {
    schema: [
      { name: "schema", num_children: columnNames.length },
      ...columnNames.map((name) => ({ name })),
    ],
    key_value_metadata:
      pandas !== undefined ? [{ key: "pandas", value: pandas }] : [],
  } as unknown as FileMetaData;
}

describe("PandasMetadataUtils", () => {
  describe("readIndexColumn", () => {
    it("reads the column an unnamed index was written as", () => {
      // As written by geopandas 1.0.1 for a SpatialData Xenium element
      const metadata = fakePandasMetadata(
        JSON.stringify({
          index_columns: ["__index_level_0__"],
          columns: [
            { name: "geometry", field_name: "geometry" },
            {
              name: null,
              field_name: "__index_level_0__",
              pandas_type: "int64",
            },
          ],
        }),
        ["geometry", "__index_level_0__"],
      );
      expect(PandasMetadataUtils.readIndexColumn(metadata)).toBe(
        "__index_level_0__",
      );
    });

    it("reads the column a named index was written as", () => {
      const metadata = fakePandasMetadata(
        JSON.stringify({
          index_columns: ["cell_id"],
          columns: [{ field_name: "cell_id", pandas_type: "int32" }],
        }),
        ["cell_id"],
      );
      expect(PandasMetadataUtils.readIndexColumn(metadata)).toBe("cell_id");
    });

    it("reads no column for an index missing from the file", () => {
      // As written by Dask for the Xenium transcripts
      const metadata = fakePandasMetadata(
        JSON.stringify({
          index_columns: ["__null_dask_index__"],
          columns: [
            { field_name: "x", pandas_type: "float32" },
            { field_name: "__null_dask_index__", pandas_type: "int64" },
          ],
        }),
        ["x"],
      );
      expect(PandasMetadataUtils.readIndexColumn(metadata)).toBeUndefined();
    });

    it("reads a string index column", () => {
      // As written by geopandas for a SpatialData Xenium element with string
      // cell IDs
      const metadata = fakePandasMetadata(
        JSON.stringify({
          index_columns: ["__index_level_0__"],
          columns: [
            {
              name: null,
              field_name: "__index_level_0__",
              pandas_type: "unicode",
            },
          ],
        }),
        ["__index_level_0__"],
      );
      expect(PandasMetadataUtils.readIndexColumn(metadata)).toBe(
        "__index_level_0__",
      );
    });

    it("reads an index column of any dtype", () => {
      const metadata = fakePandasMetadata(
        JSON.stringify({
          index_columns: ["cell_id"],
          columns: [{ field_name: "cell_id", pandas_type: "float64" }],
        }),
        ["cell_id"],
      );
      expect(PandasMetadataUtils.readIndexColumn(metadata)).toBe("cell_id");
    });

    it("reads no column for a multi-level index", () => {
      const metadata = fakePandasMetadata(
        JSON.stringify({
          index_columns: ["region", "cell_id"],
          columns: [
            { field_name: "region", pandas_type: "int64" },
            { field_name: "cell_id", pandas_type: "int64" },
          ],
        }),
      );
      expect(PandasMetadataUtils.readIndexColumn(metadata)).toBeUndefined();
    });

    it("reads no column for a range index", () => {
      const metadata = fakePandasMetadata(
        JSON.stringify({
          index_columns: [
            { kind: "range", name: null, start: 0, stop: 355, step: 1 },
          ],
        }),
      );
      expect(PandasMetadataUtils.readIndexColumn(metadata)).toBeUndefined();
    });

    it("reads no column from a file without pandas metadata", () => {
      expect(
        PandasMetadataUtils.readIndexColumn(fakePandasMetadata()),
      ).toBeUndefined();
    });
  });
});
