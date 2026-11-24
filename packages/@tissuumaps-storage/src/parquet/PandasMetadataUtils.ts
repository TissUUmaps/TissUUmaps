import { type FileMetaData, parquetSchema } from "hyparquet";

/**
 * The `pandas` metadata of a Parquet file written by pandas
 *
 * `index_columns` lists where the DataFrame index went: an entry is the name
 * of the column it was written as, or a description of a `RangeIndex` that was
 * not written at all.
 */
type PandasMetadata = {
  index_columns?: (string | { kind: string })[];
  columns?: {
    name?: string | null;
    field_name?: string;
    pandas_type?: string;
  }[];
};

/** Helpers for the `pandas` metadata of a Parquet file written by pandas */
export class PandasMetadataUtils {
  /**
   * Reads the column a pandas DataFrame index was written as
   *
   * Parquet has no index, so pandas either writes the index as an ordinary
   * column, named in the `pandas` metadata, or, for a `RangeIndex`, writes
   * nothing and records the range instead.
   *
   * The index column is returned whatever its dtype: whether its values can
   * serve as item IDs is only known once they are read.
   *
   * @param metadata - The file metadata
   * @returns The name of the index column, or `undefined` for files without
   * pandas metadata, for files whose index was not written or is missing
   * from the file, and for files whose index has multiple levels
   */
  static readIndexColumn(metadata: FileMetaData): string | undefined {
    const pandas = metadata.key_value_metadata?.find(
      ({ key }) => key === "pandas",
    );
    if (pandas?.value === undefined) {
      return undefined;
    }
    const { index_columns = [] } = JSON.parse(pandas.value) as PandasMetadata;
    // A multi-level index has no single column to key by
    const [indexColumn, ...moreLevels] = index_columns;
    if (typeof indexColumn !== "string" || moreLevels.length > 0) {
      return undefined;
    }
    // e.g. Dask lists __null_dask_index__ without writing it
    const isWritten = parquetSchema(metadata).children.some(
      ({ element }) => element.name === indexColumn,
    );
    return isWritten ? indexColumn : undefined;
  }

  /**
   * Reads the columns written from a pandas `category` dtype
   *
   * Parquet stores categories as their values (dictionary encoded), so only
   * the `pandas` metadata tells them apart from other columns of those
   * values, e.g. numerically coded clusters.
   *
   * @param metadata - The file metadata
   * @returns The names of the categorical columns, empty for files without
   * pandas metadata
   */
  static readCategoricalColumns(metadata: FileMetaData): string[] {
    const pandas = metadata.key_value_metadata?.find(
      ({ key }) => key === "pandas",
    );
    if (pandas?.value === undefined) {
      return [];
    }
    const { columns = [] } = JSON.parse(pandas.value) as PandasMetadata;
    // field_name is the column's name in the file, name its pandas label
    return columns
      .filter((column) => column.pandas_type === "categorical")
      .map((column) => column.field_name ?? column.name)
      .filter((name) => typeof name === "string");
  }
}
