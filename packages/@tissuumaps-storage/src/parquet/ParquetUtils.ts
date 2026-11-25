import {
  type AsyncBuffer,
  type FileMetaData,
  type SchemaElement,
  parquetRead,
  parquetSchema,
} from "hyparquet";
import { compressors } from "hyparquet-compressors";

import {
  NumberUtils,
  type TypedArray,
  type TypedArrayOrArray,
} from "@tissuumaps/core";

import type { ParquetPart } from "./types";

/**
 * Reads what any Parquet file holds, for the worker and the profiles
 *
 * A table may be split into several files, its parts, whose rows follow one
 * another in order and which share the schema of the first part. A table has
 * at least one part. Methods that read the rows of a table take its parts;
 * methods that read the metadata of a single file take that file's metadata.
 */
export class ParquetUtils {
  /**
   * @param metadata - The file metadata
   * @returns The number of rows
   * @throws Error if the row count is not a safe integer
   */
  static getNumRows(metadata: FileMetaData): number {
    const numRows = Number(metadata.num_rows);
    if (!Number.isSafeInteger(numRows)) {
      throw new Error("Parquet file has too many rows");
    }
    return numRows;
  }

  /**
   * @param parts - The parts of a table
   * @returns The number of rows of all parts together
   * @throws Error if the row count of a part or of all parts together is not a
   * safe integer
   */
  static getTotalNumRows(parts: ParquetPart[]): number {
    const numRows = parts.reduce(
      (sum, { metadata }) => sum + ParquetUtils.getNumRows(metadata),
      0,
    );
    if (!Number.isSafeInteger(numRows)) {
      throw new Error("Parquet table has too many rows");
    }
    return numRows;
  }

  /**
   * @param metadata - The file metadata
   * @returns The column names, in schema order
   */
  static getColumns(metadata: FileMetaData): string[] {
    return parquetSchema(metadata).children.map(
      (columnMetadata) => columnMetadata.element.name,
    );
  }

  /**
   * @param metadata - The file metadata
   * @param column - The column name
   * @returns The schema element of the column, or `undefined` if the file has
   * no such column
   */
  static getColumnElement(
    metadata: FileMetaData,
    column: string,
  ): SchemaElement | undefined {
    return parquetSchema(metadata).children.find(
      (columnMetadata) => columnMetadata.element.name === column,
    )?.element;
  }

  /**
   * Returns the typed array type a column is read as, or `undefined` for a
   * column read as a plain array
   *
   * 32-bit integers and floats keep their type, 64-bit integers and decimals
   * are read as 64-bit floats holding safe integers, half floats as 32-bit
   * floats; strings, dates, times, booleans, nested data and the like are read
   * as plain arrays.
   *
   * @param element - The schema element of the column
   * @returns The typed array type, or `undefined` for a plain array
   */
  static getColumnArrayType(
    element: SchemaElement,
  ):
    | Int32ArrayConstructor
    | Uint32ArrayConstructor
    | Float32ArrayConstructor
    | Float64ArrayConstructor
    | undefined {
    const { type, converted_type: converted, logical_type: logical } = element;
    const annotation = converted ?? logical?.type;
    if (annotation === "DECIMAL") {
      return Float64Array;
    }
    if (annotation === "FLOAT16") {
      return Float32Array;
    }
    if (
      annotation !== undefined &&
      annotation !== "INTEGER" &&
      !/^U?INT_(8|16|32|64)$/.test(annotation)
    ) {
      return undefined;
    }
    const unsigned =
      converted === "UINT_32" ||
      (logical?.type === "INTEGER" && !logical.isSigned);
    switch (type) {
      case "INT32":
        return unsigned ? Uint32Array : Int32Array;
      case "INT64":
      case "DOUBLE":
        return Float64Array;
      case "FLOAT":
        return Float32Array;
      default:
        return undefined;
    }
  }

  /**
   * Tells whether a column has missing values, from the null counts in its
   * row group statistics
   *
   * @param metadata - The file metadata
   * @param column - The column name
   * @returns Whether the column has missing values, or `undefined` if a row
   * group lacks the statistic
   */
  static isNullable(
    metadata: FileMetaData,
    column: string,
  ): boolean | undefined {
    let numNulls = 0;
    for (const rowGroup of metadata.row_groups) {
      const nullCount = rowGroup.columns.find(
        ({ meta_data }) => meta_data?.path_in_schema.join(".") === column,
      )?.meta_data?.statistics?.null_count;
      if (nullCount === undefined) {
        return undefined;
      }
      numNulls += Number(nullCount);
    }
    return numNulls > 0;
  }

  /**
   * Reads one column chunk by chunk
   *
   * @param buffer - The file to read from
   * @param metadata - The file metadata
   * @param column - The column to read
   * @param onChunk - Called with each chunk and the row it starts at
   * @param onProgress - Called with the bytes read so far and the file size
   */
  static readColumnChunks(
    buffer: AsyncBuffer,
    metadata: FileMetaData,
    column: string,
    onChunk: (columnData: unknown, rowStart: number) => void,
    onProgress: (progress: number, total: number) => void,
  ): Promise<void> {
    let bytesRead = 0;
    return parquetRead({
      file: {
        byteLength: buffer.byteLength,
        async slice(start, end) {
          const chunk = await buffer.slice(start, end);
          bytesRead += chunk.byteLength;
          onProgress(bytesRead, buffer.byteLength);
          return chunk;
        },
      },
      metadata,
      compressors,
      columns: [column],
      onChunk: ({ columnData, rowStart }) => onChunk(columnData, rowStart),
    });
  }

  /**
   * Reads every part of a table in turn
   *
   * @param parts - The parts of the table
   * @param onProgress - Called with the bytes read so far from all parts and
   * their total size
   * @param read - Reads one part, given the table row it starts at and a
   * callback reporting the bytes read from it
   */
  static async readParts(
    parts: ParquetPart[],
    onProgress: (progress: number, total: number) => void,
    read: (
      part: ParquetPart,
      rowOffset: number,
      onPartProgress: (progress: number) => void,
    ) => Promise<void>,
  ): Promise<void> {
    const totalBytes = parts.reduce(
      (sum, { buffer }) => sum + buffer.byteLength,
      0,
    );
    let bytesBefore = 0;
    let rowOffset = 0;
    for (const part of parts) {
      await read(part, rowOffset, (progress) =>
        onProgress(bytesBefore + progress, totalBytes),
      );
      bytesBefore += part.buffer.byteLength;
      rowOffset += ParquetUtils.getNumRows(part.metadata);
    }
  }

  /**
   * Reads a column of a table into the array the storage API holds
   *
   * The array is allocated from the column's schema (see
   * {@link ParquetUtils.getColumnArrayType}) before reading, and the chunks
   * are written into it as they arrive. A missing value is `NaN` in a float
   * array and `null` in a plain array, so an integer column is read as 64-bit
   * floats unless, in every part holding rows, its schema requires a value or
   * its statistics count no nulls.
   *
   * @param parts - The parts of the table
   * @param column - The column to read
   * @param onProgress - Called with the bytes read so far from all parts and
   * their total size
   * @returns The column values of all rows
   * @throws Error if the first part has no such column, or if the row count is
   * not a safe integer
   */
  static async readColumn(
    parts: ParquetPart[],
    column: string,
    onProgress: (progress: number, total: number) => void,
  ): Promise<TypedArrayOrArray<unknown>> {
    const element = ParquetUtils.getColumnElement(parts[0]!.metadata, column);
    if (element === undefined) {
      throw new Error(`Column '${column}' not found in the Parquet file`);
    }
    const numRows = ParquetUtils.getTotalNumRows(parts);
    const arrayType = ParquetUtils.getColumnArrayType(element);
    let result: TypedArray | unknown[];
    if (arrayType === undefined) {
      result = new Array<unknown>(numRows).fill(null);
    } else if (arrayType === Float32Array || arrayType === Float64Array) {
      result = new arrayType(numRows).fill(NaN);
    } else if (
      parts.every(
        ({ metadata }) =>
          ParquetUtils.getNumRows(metadata) === 0 ||
          ParquetUtils.getColumnElement(metadata, column)?.repetition_type ===
            "REQUIRED" ||
          ParquetUtils.isNullable(metadata, column) === false,
      )
    ) {
      result = new arrayType(numRows);
    } else {
      result = new Float64Array(numRows).fill(NaN);
    }
    await ParquetUtils.readParts(
      parts,
      onProgress,
      (part, rowOffset, onPartProgress) =>
        ParquetUtils.readColumnChunks(
          part.buffer,
          part.metadata,
          column,
          (columnData, rowStart) => {
            const start = rowOffset + rowStart;
            const chunk = columnData as
              unknown[] | TypedArray | BigInt64Array | BigUint64Array;
            if (Array.isArray(result)) {
              for (let i = 0; i < chunk.length; i++) {
                result[start + i] = chunk[i];
              }
            } else if (
              chunk instanceof BigInt64Array ||
              chunk instanceof BigUint64Array
            ) {
              for (let i = 0; i < chunk.length; i++) {
                result[start + i] = NumberUtils.parseSafeInt(chunk[i]);
              }
            } else if (ArrayBuffer.isView(chunk)) {
              result.set(chunk, start);
            } else {
              for (let i = 0; i < chunk.length; i++) {
                const v = chunk[i] as number | bigint | null;
                result[start + i] =
                  v === null
                    ? NaN
                    : typeof v === "bigint"
                      ? NumberUtils.parseSafeInt(v)
                      : v;
              }
            }
          },
          onPartProgress,
        ),
    );
    return result;
  }
}
