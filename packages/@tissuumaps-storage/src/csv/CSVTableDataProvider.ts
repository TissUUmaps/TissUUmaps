import {
  type ParseLocalConfig,
  type ParseRemoteConfig,
  type ParseResult,
  type Parser,
  parse,
} from "papaparse";

import {
  ArrayUtils,
  AsyncUtils,
  type DataProviderLoadOptions,
  type IDArray,
  NumberUtils,
  SourceUtils,
  type TableDataProvider,
} from "@tissuumaps/core";

import { CSVTableData } from "./CSVTableData";
import {
  type CSVTableDataSource,
  type NormalizedCSVTableDataSource,
  csvTableDataSourceDefaults,
} from "./CSVTableDataSource";

type ColumnValues = string[] | Float32Array | Float64Array;

export class CSVTableDataProvider implements TableDataProvider<
  CSVTableDataSource,
  CSVTableData,
  NormalizedCSVTableDataSource
> {
  readonly name = "CSV";

  readonly schema = {
    type: "object",
    properties: {
      source: {
        type: "string",
      },
      // TODO columns
      idColumn: {
        type: "string",
      },
      nameColumn: {
        type: "string",
      },
      // TODO loadColumns
      // TODO parseConfig
    },
    required: ["source"],
  };

  readonly uischema = {
    type: "VerticalLayout",
    elements: [
      {
        type: "Control",
        scope: "#/properties/source",
        label: "Source",
      },
      // TODO columns
      {
        type: "Control",
        scope: "#/properties/idColumn",
        label: "ID Column",
      },
      {
        type: "Control",
        scope: "#/properties/nameColumn",
        label: "Name Column",
      },
      // TODO loadColumns
      // TODO parseConfig
    ],
  };

  normalize(
    dataSource: CSVTableDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedCSVTableDataSource {
    return {
      ...csvTableDataSourceDefaults,
      ...dataSource,
      source: SourceUtils.normalizeSource(
        dataSource.source,
        workspace,
        projectSource,
      ),
    };
  }

  async load(
    normalizedDataSource: NormalizedCSVTableDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<CSVTableData> {
    const { signal, onProgress, workspace = null } = options ?? {};
    signal?.throwIfAborted();

    let columnMetas:
      | {
          name: string;
          index: number;
          arrayType: Float32ArrayConstructor | Float64ArrayConstructor;
          isString: boolean;
          chunks: ColumnValues[];
        }[]
      | undefined;
    let byteLength: number | undefined;
    let parseError: unknown;

    const parseConfig: Partial<ParseLocalConfig & ParseRemoteConfig> = {
      ...normalizedDataSource.parseConfig,
      worker: true,
      header: false,
      skipEmptyLines: true,
      chunk: (results: ParseResult<string[]>, parser: Parser) => {
        if (signal?.aborted) {
          parser.abort();
          return;
        }
        let columnChunks: ColumnValues[] | undefined;
        let numChunkRows = results.data.length;
        let currentChunkRow = 0;
        for (const rowData of results.data) {
          if (columnMetas === undefined) {
            let columns = normalizedDataSource.columns;
            if (columns === undefined) {
              columns = rowData;
              numChunkRows -= 1;
            }
            columnMetas = (normalizedDataSource.loadColumns ?? columns).map(
              (column) => ({
                name: column,
                index: columns.indexOf(column),
                arrayType:
                  column === normalizedDataSource.idColumn
                    ? Float64Array
                    : Float32Array,
                isString: false,
                chunks: [],
              }),
            );
            if (columns === rowData) {
              continue;
            }
          }
          if (columnChunks === undefined) {
            columnChunks = columnMetas.map((column) =>
              column.isString
                ? new Array<string>(numChunkRows)
                : new column.arrayType(numChunkRows),
            );
          }
          for (let c = 0; c < columnMetas.length; c++) {
            const columnMeta = columnMetas[c]!;
            const columnChunk = columnChunks[c]!;
            if (columnMeta.index < 0) {
              parseError = new Error(`Column "${columnMeta.name}" not found`);
              parser.abort();
              return;
            }
            if (columnMeta.index >= rowData.length) {
              parseError = new Error(
                `Missing value for column "${columnMeta.name}"`,
              );
              parser.abort();
              return;
            }
            const value = rowData[columnMeta.index]!;
            if (Array.isArray(columnChunk)) {
              columnChunk[currentChunkRow] = value;
            } else if (value.trim() === "") {
              columnChunk[currentChunkRow] = NaN;
            } else {
              const numericValue = NumberUtils.tryParseFinite(value);
              if (numericValue !== undefined) {
                columnChunk[currentChunkRow] = numericValue;
              } else {
                columnMeta.isString = true;
                for (let i = 0; i < columnMeta.chunks.length; i++) {
                  // the column has been numeric so far
                  columnMeta.chunks[i] = Array.from(
                    columnMeta.chunks[i] as Float32Array | Float64Array,
                    (v) => (Number.isNaN(v) ? "" : String(v)),
                  );
                }
                const newColumnChunk = new Array<string>(numChunkRows);
                for (let i = 0; i < currentChunkRow; i++) {
                  const v = columnChunk[i]!;
                  newColumnChunk[i] = Number.isNaN(v) ? "" : String(v);
                }
                newColumnChunk[currentChunkRow] = value;
                columnChunks[c] = newColumnChunk;
              }
            }
          }
          currentChunkRow++;
        }
        if (columnMetas !== undefined && columnChunks !== undefined) {
          for (let c = 0; c < columnMetas.length; c++) {
            const columnMeta = columnMetas[c]!;
            const columnChunk = columnChunks[c]!;
            columnMeta.chunks.push(columnChunk);
          }
        }
        if (onProgress !== undefined && byteLength !== undefined) {
          onProgress(
            results.meta.cursor,
            Math.max(byteLength, results.meta.cursor),
          );
        }
      },
    };

    const completeParse = (
      resolve: (columnValues: Map<string, ColumnValues>) => void,
      reject: (error: unknown) => void,
    ) => {
      if (signal?.aborted) {
        reject(signal.reason);
        return;
      }
      if (parseError !== undefined) {
        reject(parseError);
        return;
      }
      const columnValues = new Map<string, ColumnValues>();
      if (columnMetas !== undefined) {
        for (const columnMeta of columnMetas) {
          let values;
          if (columnMeta.isString) {
            const chunks = columnMeta.chunks as string[][];
            values = chunks.flat();
          } else {
            const chunks = columnMeta.chunks as (Float32Array | Float64Array)[];
            const n = chunks.reduce((n, chunk) => n + chunk.length, 0);
            values = new columnMeta.arrayType(n);
            let offset = 0;
            for (const chunk of chunks) {
              values.set(chunk, offset);
              offset += chunk.length;
            }
          }
          columnValues.set(columnMeta.name, values);
          columnMeta.chunks = [];
        }
      }
      resolve(columnValues);
    };

    const source = await SourceUtils.openSourceFile(
      normalizedDataSource.source,
      workspace,
      { signal },
    );
    let columnValues: Map<string, ColumnValues>;
    if (source.url !== undefined) {
      const { url } = source;
      if (onProgress !== undefined) {
        try {
          const headResponse = await fetch(url, { method: "HEAD", signal });
          const contentLength = headResponse.headers.get("Content-Length");
          if (contentLength !== null) {
            byteLength = Number(contentLength);
          }
        } catch (error) {
          if (signal?.aborted) {
            throw error;
          }
        }
      }
      columnValues = await AsyncUtils.raceSignal(
        new Promise<Map<string, ColumnValues>>((resolve, reject) =>
          parse(url, {
            ...parseConfig,
            download: true,
            error: reject,
            complete: () => completeParse(resolve, reject),
          }),
        ),
        { signal },
      );
    } else {
      const { file } = source;
      byteLength = file.size;
      columnValues = await AsyncUtils.raceSignal(
        new Promise<Map<string, ColumnValues>>((resolve, reject) =>
          parse(file, {
            ...parseConfig,
            error: reject,
            complete: () => completeParse(resolve, reject),
          }),
        ),
        { signal },
      );
    }

    if (columnMetas === undefined || columnMetas.length === 0) {
      throw new Error("No columns found in the CSV file.");
    }

    const n = columnValues.get(columnMetas[0]!.name)?.length ?? 0;

    let ids: IDArray | undefined;
    if (normalizedDataSource.idColumn !== undefined) {
      const idColumnValues = columnValues.get(normalizedDataSource.idColumn);
      if (idColumnValues === undefined) {
        throw new Error(
          `ID column "${normalizedDataSource.idColumn}" does not exist in the table.`,
        );
      }
      if (
        Array.isArray(idColumnValues)
          ? idColumnValues.includes("")
          : idColumnValues.some((id) => Number.isNaN(id))
      ) {
        throw new Error(
          `ID column "${normalizedDataSource.idColumn}" has missing values.`,
        );
      }
      ids = ArrayUtils.toIDArray(idColumnValues);
    }

    let names: string[] | undefined;
    if (normalizedDataSource.nameColumn !== undefined) {
      const nameColumnValues = columnValues.get(
        normalizedDataSource.nameColumn,
      );
      if (nameColumnValues === undefined) {
        throw new Error(
          `Name column "${normalizedDataSource.nameColumn}" does not exist in the table.`,
        );
      }
      names = Array.from<string | number, string>(nameColumnValues, String);
    }

    return new CSVTableData(
      n,
      ids,
      names,
      columnMetas.map((columnMeta) => columnMeta.name),
      columnValues,
    );
  }
}
