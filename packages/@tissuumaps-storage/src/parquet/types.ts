import type { AsyncBuffer, FileMetaData } from "hyparquet";

export type ParquetSource = {
  file?: File;
  url?: string;
  headers?: { [header: string]: string };
};

/**
 * An open Parquet file and its metadata, such as one of the files a table is
 * split into
 */
export type ParquetPart = { buffer: AsyncBuffer; metadata: FileMetaData };
