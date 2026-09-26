import {
  type AsyncBuffer,
  type FileMetaData,
  parquetRead,
  parquetSchema,
} from "hyparquet";
import { compressors } from "hyparquet-compressors";

/** Reads what any Parquet file holds, for the worker and the profiles */
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
   * @param metadata - The file metadata
   * @returns The column names, in schema order
   */
  static getColumns(metadata: FileMetaData): string[] {
    return parquetSchema(metadata).children.map(
      (columnMetadata) => columnMetadata.element.name,
    );
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
}
