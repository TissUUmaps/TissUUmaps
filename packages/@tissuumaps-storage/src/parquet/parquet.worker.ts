import type { Geometry } from "geojson";
import {
  type AsyncBuffer,
  type FileMetaData,
  type SchemaElement,
  asyncBufferFromUrl,
  parquetMetadataAsync,
  parquetRead,
  parquetSchema,
} from "hyparquet";
import { compressors } from "hyparquet-compressors";

import {
  ArrayUtils,
  type IDArray,
  NumberUtils,
  type ShapesGeometry,
  type TypedArray,
  type TypedArrayOrArray,
} from "@tissuumaps/core";

import { ShapesGeometryBuilder } from "../common/ShapesGeometryBuilder";
import { PandasMetadataUtils } from "./PandasMetadataUtils";
import {
  type CoordinateColumn,
  GeoParquetUtils,
} from "./profiles/GeoParquetUtils";
import type { ParquetSource } from "./types";

/**
 * Appends a GeoJSON geometry, as decoded from a WKB column, as one shape
 *
 * @param builder - The builder of the shapes geometry under construction
 * @param geometry - The geometry to add
 * @param id - The ID of the shape
 * @param name - The name of the shape, if any
 */
function addGeometry(
  builder: ShapesGeometryBuilder,
  geometry: Geometry,
  id: number | string,
  name?: string,
): void {
  if (geometry.type === "Polygon") {
    builder.addShape([geometry.coordinates], id, name);
    return;
  }
  if (geometry.type === "MultiPolygon") {
    builder.addShape(geometry.coordinates, id, name);
    return;
  }
  console.warn(`Unsupported geometry type: ${geometry.type}`);
}

export type ParquetRequest<TOp extends string = string> = {
  op: TOp;
};

export type ParquetResponse<TRequest extends ParquetRequest> = {
  op: TRequest["op"];
};

export type ParquetFileRequest = ParquetRequest<"file"> & {
  source: ParquetSource;
  idColumn: string | undefined;
  nameColumn: string | undefined;
};

export type ParquetFileResponse = ParquetResponse<ParquetFileRequest> & {
  numRows: number;
  columns: string[];
  coordinateColumns: CoordinateColumn[];
  ids: IDArray | undefined;
  names: string[] | undefined;
};

export type ParquetColumnRequest = ParquetRequest<"column"> & {
  source: ParquetSource;
  column: string;
};

export type ParquetColumnResponse = ParquetResponse<ParquetColumnRequest> & {
  data: TypedArrayOrArray<unknown>;
};

export type ParquetCoordinatesRequest = ParquetRequest<"coordinates"> & {
  source: ParquetSource;
  geometryColumn: string;
};

export type ParquetCoordinatesResponse =
  ParquetResponse<ParquetCoordinatesRequest> & {
    x: Float32Array;
    y: Float32Array;
  };

export type ParquetShapesRequest = ParquetRequest<"shapes"> & {
  source: ParquetSource;
  geometryColumn: string | undefined;
  idColumn: string | undefined;
  nameColumn: string | undefined;
};

export type ParquetShapesResponse = ParquetResponse<ParquetShapesRequest> & {
  geometry: ShapesGeometry;
  ids: IDArray;
  names: string[] | undefined;
};

export type ParquetRangeRequest = ParquetRequest<"range"> & {
  source: ParquetSource;
  /** The column to read, or the geometry column when an axis is given */
  column: string;
  /** Set to read the range of one axis of a point geometry column */
  axis?: "x" | "y";
};

export type ParquetRangeResponse = ParquetResponse<ParquetRangeRequest> & {
  range: [number, number] | undefined;
};

export type ParquetWorkerRequest =
  | ParquetFileRequest
  | ParquetColumnRequest
  | ParquetCoordinatesRequest
  | ParquetShapesRequest
  | ParquetRangeRequest;

export type ParquetWorkerResponse =
  | ParquetFileResponse
  | ParquetColumnResponse
  | ParquetCoordinatesResponse
  | ParquetShapesResponse
  | ParquetRangeResponse
  | { error: string };

export type ParquetWorkerResponseFor<
  TWorkerRequest extends ParquetWorkerRequest,
> = Extract<ParquetWorkerResponse, { op: TWorkerRequest["op"] }>;

export type ParquetWorkerMessage =
  ParquetWorkerResponse | { progress: number; total: number };

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<ParquetWorkerRequest>) => void) | null;
  postMessage: (
    message: ParquetWorkerMessage,
    transfer?: Transferable[],
  ) => void;
};

ctx.onmessage = (event) => {
  void (async () => {
    try {
      let result;
      switch (event.data.op) {
        case "file":
          result = await handleFileRequest(event.data, (progress, total) =>
            ctx.postMessage({ progress, total }),
          );
          break;
        case "column":
          result = await handleColumnRequest(event.data, (progress, total) =>
            ctx.postMessage({ progress, total }),
          );
          break;
        case "coordinates":
          result = await handleCoordinatesRequest(
            event.data,
            (progress, total) => ctx.postMessage({ progress, total }),
          );
          break;
        case "shapes":
          result = await handleShapesRequest(event.data, (progress, total) =>
            ctx.postMessage({ progress, total }),
          );
          break;
        case "range":
          result = await handleRangeRequest(event.data, (progress, total) =>
            ctx.postMessage({ progress, total }),
          );
          break;
        default:
          throw new Error("Unknown request");
      }
      ctx.postMessage(result.response, result.transfer);
    } catch (error) {
      ctx.postMessage({
        error: error instanceof Error ? error.message : String(error),
      });
    }
  })();
};

function openParquet(source: ParquetSource): Promise<AsyncBuffer> {
  if (source.file !== undefined) {
    return Promise.resolve({
      byteLength: source.file.size,
      slice: (start: number, end?: number) =>
        source.file!.slice(start, end).arrayBuffer(),
    });
  }
  if (source.url !== undefined) {
    return asyncBufferFromUrl({
      url: source.url,
      requestInit: { headers: source.headers },
    });
  }
  return Promise.reject(new Error("A URL or file is required to load data."));
}

function getNumRows(metadata: FileMetaData): number {
  const numRows = Number(metadata.num_rows);
  if (!Number.isSafeInteger(numRows)) {
    throw new Error("Parquet file has too many rows");
  }
  return numRows;
}

function getColumns(metadata: FileMetaData): string[] {
  return parquetSchema(metadata).children.map(
    (columnMetadata) => columnMetadata.element.name,
  );
}

function readColumnChunks(
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
 * Returns the typed array type a column is read as, or `undefined` for a
 * column read as a plain array
 *
 * 32-bit integers and floats keep their type, 64-bit integers and decimals
 * are read as 64-bit floats holding safe integers, half floats as 32-bit
 * floats; strings, dates, times, booleans, nested data and the like are read
 * as plain arrays.
 */
function getColumnArrayType(
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
 * Tells whether a column has missing values, from the null counts in its row
 * group statistics; `undefined` if a row group lacks the statistic
 */
function isNullable(
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
 * Reads a column into the array the storage API holds
 *
 * The array is allocated from the column's schema (see
 * {@link getColumnArrayType}) before reading, and the chunks are written into
 * it as they arrive. A missing value is `NaN` in a float array and `null` in a
 * plain array, so an integer column is read as 64-bit floats unless its schema
 * requires a value or its statistics count no nulls.
 */
async function readParquetColumn(
  buffer: AsyncBuffer,
  metadata: FileMetaData,
  column: string,
  onProgress: (progress: number, total: number) => void,
): Promise<TypedArrayOrArray<unknown>> {
  const element = parquetSchema(metadata).children.find(
    (columnMetadata) => columnMetadata.element.name === column,
  )?.element;
  if (element === undefined) {
    throw new Error(`Column "${column}" not found in Parquet file`);
  }
  const numRows = getNumRows(metadata);
  const arrayType = getColumnArrayType(element);
  let result: TypedArray | unknown[];
  if (arrayType === undefined) {
    result = new Array<unknown>(numRows).fill(null);
  } else if (arrayType === Float32Array || arrayType === Float64Array) {
    result = new arrayType(numRows).fill(NaN);
  } else if (
    element.repetition_type === "REQUIRED" ||
    isNullable(metadata, column) === false
  ) {
    result = new arrayType(numRows);
  } else {
    result = new Float64Array(numRows).fill(NaN);
  }
  await readColumnChunks(
    buffer,
    metadata,
    column,
    (columnData, rowStart) => {
      const chunk = columnData as
        unknown[] | TypedArray | BigInt64Array | BigUint64Array;
      if (Array.isArray(result)) {
        for (let i = 0; i < chunk.length; i++) {
          result[rowStart + i] = chunk[i];
        }
      } else if (
        chunk instanceof BigInt64Array ||
        chunk instanceof BigUint64Array
      ) {
        for (let i = 0; i < chunk.length; i++) {
          result[rowStart + i] = NumberUtils.parseSafeInt(chunk[i]);
        }
      } else if (ArrayBuffer.isView(chunk)) {
        result.set(chunk, rowStart);
      } else {
        for (let i = 0; i < chunk.length; i++) {
          const v = chunk[i] as number | bigint | null;
          result[rowStart + i] =
            v === null
              ? NaN
              : typeof v === "bigint"
                ? NumberUtils.parseSafeInt(v)
                : v;
        }
      }
    },
    onProgress,
  );
  return result;
}

function readGeometryColumn(
  buffer: AsyncBuffer,
  metadata: FileMetaData,
  column: string,
  onGeometry: (geometry: Geometry | null, row: number) => void,
  onProgress: (progress: number, total: number) => void,
): Promise<void> {
  return readColumnChunks(
    buffer,
    metadata,
    column,
    (columnData, rowStart) => {
      // WKB columns are decoded to GeoJSON geometries by the Parquet reader
      const geometries = columnData as (Geometry | null | undefined)[];
      for (let i = 0; i < geometries.length; i++) {
        onGeometry(geometries[i] ?? null, rowStart + i);
      }
    },
    onProgress,
  );
}

async function readIdsAndNames(
  buffer: AsyncBuffer,
  metadata: FileMetaData,
  idColumn: string | undefined,
  nameColumn: string | undefined,
  onProgress: (progress: number, total: number) => void,
): Promise<{ ids: IDArray | undefined; names: string[] | undefined }> {
  // Without an ID column, rows are keyed by the pandas index, if there is one
  const indexColumn =
    idColumn === undefined
      ? PandasMetadataUtils.readIndexColumn(metadata)
      : undefined;
  const keyColumn = idColumn ?? indexColumn;
  let idTotal = 0,
    nameTotal = 0,
    idProgress = 0,
    nameProgress = 0;
  let idsPromise: Promise<IDArray | undefined> | undefined;
  if (keyColumn !== undefined) {
    idsPromise = readParquetColumn(
      buffer,
      metadata,
      keyColumn,
      (progress, total) => {
        idTotal = total;
        idProgress = progress;
        onProgress(idProgress + nameProgress, idTotal + nameTotal);
      },
    ).then((idData) => {
      try {
        return ArrayUtils.toIDArray(idData);
      } catch (error) {
        throw new Error(`ID column "${keyColumn}" does not hold item IDs`, {
          cause: error,
        });
      }
    });
    if (indexColumn !== undefined) {
      // the pandas index was not asked for, so it must not fail the read
      idsPromise = idsPromise.catch((error: unknown) => {
        console.warn("Keying rows by row number instead:", error);
        return undefined;
      });
    }
  }
  const nameDataPromise =
    nameColumn !== undefined
      ? readParquetColumn(buffer, metadata, nameColumn, (progress, total) => {
          nameTotal = total;
          nameProgress = progress;
          onProgress(idProgress + nameProgress, idTotal + nameTotal);
        })
      : undefined;
  const [ids, nameData] = await Promise.all([idsPromise, nameDataPromise]);
  const names =
    nameData !== undefined ? Array.from(nameData, String) : undefined;
  // e.g. the partition-local index that Dask writes; items are looked up by ID
  if (ids !== undefined && new Set<unknown>(ids).size !== ids.length) {
    console.warn(
      `ID column "${keyColumn}" has duplicate values, keying rows by row number instead`,
    );
    return { ids: undefined, names };
  }
  return { ids, names };
}

async function handleFileRequest(
  request: ParquetFileRequest,
  onProgress: (progress: number, total: number) => void,
): Promise<{
  response: ParquetFileResponse;
  transfer?: Transferable[];
}> {
  const buffer = await openParquet(request.source);
  const metadata = await parquetMetadataAsync(buffer);
  const { ids, names } = await readIdsAndNames(
    buffer,
    metadata,
    request.idColumn,
    request.nameColumn,
    onProgress,
  );
  const geoColumns = GeoParquetUtils.readGeoColumns(metadata);
  const coordinateColumns = GeoParquetUtils.getCoordinateColumns(geoColumns);
  return {
    response: {
      op: "file",
      numRows: getNumRows(metadata),
      columns: [
        ...getColumns(metadata).filter(
          (column) => !geoColumns.some(({ name }) => name === column),
        ),
        ...coordinateColumns.map(({ column }) => column),
      ],
      coordinateColumns,
      ids,
      names,
    },
    transfer:
      ArrayBuffer.isView(ids) && ids.buffer instanceof ArrayBuffer
        ? [ids.buffer]
        : undefined,
  };
}

/**
 * Reads both coordinate axes of a point geometry column in one pass
 *
 * The WKB column is decoded once for both axes, so that a point cloud reading
 * its x and y from the same column does not decode it twice.
 *
 * @param buffer - The file to read from
 * @param metadata - The file metadata
 * @param column - The point geometry column to read
 * @param onProgress - Callback reporting the read progress
 * @returns The x and y coordinates of every row
 * @throws Error if a row holds no geometry, or one that is not a point
 */
async function readCoordinateColumns(
  buffer: AsyncBuffer,
  metadata: FileMetaData,
  column: string,
  onProgress: (progress: number, total: number) => void,
): Promise<{ x: Float32Array; y: Float32Array }> {
  const numRows = getNumRows(metadata);
  const x = new Float32Array(numRows);
  const y = new Float32Array(numRows);
  await readGeometryColumn(
    buffer,
    metadata,
    column,
    (geometry, row) => {
      if (geometry === null) {
        throw new Error(`Missing geometry in column "${column}"`);
      }
      if (geometry.type !== "Point") {
        throw new Error(
          `Column "${column}" contains a ${geometry.type} geometry`,
        );
      }
      x[row] = geometry.coordinates[0]!;
      y[row] = geometry.coordinates[1]!;
    },
    onProgress,
  );
  return { x, y };
}

async function handleCoordinatesRequest(
  request: ParquetCoordinatesRequest,
  onProgress: (progress: number, total: number) => void,
): Promise<{
  response: ParquetCoordinatesResponse;
  transfer?: Transferable[];
}> {
  const buffer = await openParquet(request.source);
  const metadata = await parquetMetadataAsync(buffer);
  const { x, y } = await readCoordinateColumns(
    buffer,
    metadata,
    request.geometryColumn,
    onProgress,
  );
  return {
    response: { op: "coordinates", x, y },
    transfer: [x.buffer, y.buffer],
  };
}

async function handleColumnRequest(
  request: ParquetColumnRequest,
  onProgress: (progress: number, total: number) => void,
): Promise<{
  response: ParquetColumnResponse;
  transfer?: Transferable[];
}> {
  const buffer = await openParquet(request.source);
  const metadata = await parquetMetadataAsync(buffer);
  const data = await readParquetColumn(
    buffer,
    metadata,
    request.column,
    onProgress,
  );
  return {
    response: { op: "column", data },
    transfer:
      ArrayBuffer.isView(data) && data.buffer instanceof ArrayBuffer
        ? [data.buffer]
        : undefined,
  };
}

async function handleShapesRequest(
  request: ParquetShapesRequest,
  onProgress: (progress: number, total: number) => void,
): Promise<{
  response: ParquetShapesResponse;
  transfer?: Transferable[];
}> {
  const buffer = await openParquet(request.source);
  const metadata = await parquetMetadataAsync(buffer);
  const geoColumns = GeoParquetUtils.readGeoColumns(metadata);
  const geoColumn =
    request.geometryColumn !== undefined
      ? geoColumns.find(({ name }) => name === request.geometryColumn)
      : GeoParquetUtils.getPrimaryColumn(geoColumns);
  if (geoColumn === undefined) {
    throw new Error(
      request.geometryColumn !== undefined
        ? `Geometry column "${request.geometryColumn}" is missing or not encoded as WKB`
        : "Parquet file has no GeoParquet geometry column",
    );
  }
  const pointColumnMessage =
    `Geometry column "${geoColumn.name}" holds points, which are read as ` +
    `the "${geoColumn.name}[x]" and "${geoColumn.name}[y]" columns of a table`;
  if (GeoParquetUtils.isPointColumn(geoColumn)) {
    throw new Error(pointColumnMessage);
  }
  // Progress only tracks the geometry, which dwarfs the ID and name columns
  const { ids: rowIds, names: rowNames } = await readIdsAndNames(
    buffer,
    metadata,
    request.idColumn,
    request.nameColumn,
    () => {},
  );
  const builder = new ShapesGeometryBuilder();
  // A column whose "geo" metadata lists no geometry types is only found to
  // hold points while decoding it
  let numPoints = 0;
  await readGeometryColumn(
    buffer,
    metadata,
    geoColumn.name,
    (rowGeometry, row) => {
      if (rowGeometry === null) {
        console.warn("Skipping row without geometry.");
        return;
      }
      if (rowGeometry.type === "Point") {
        numPoints++;
        return;
      }
      addGeometry(
        builder,
        rowGeometry,
        rowIds !== undefined ? rowIds[row]! : row,
        rowNames?.[row],
      );
    },
    onProgress,
  );
  if (builder.size === 0) {
    throw new Error(
      numPoints > 0
        ? pointColumnMessage
        : `No valid geometries found in column "${geoColumn.name}"`,
    );
  }
  if (numPoints > 0) {
    console.warn(
      `Skipped ${numPoints} points in column "${geoColumn.name}", which are ` +
        `not shapes.`,
    );
  }
  const { geometry, ids, names } = builder.build();
  return {
    response: {
      op: "shapes",
      geometry,
      ids,
      names,
    },
    transfer: [
      geometry.shapePolygonOffsets.buffer,
      geometry.polygonRingOffsets.buffer,
      geometry.ringVertexOffsets.buffer,
      geometry.coords.buffer,
      ...(ArrayBuffer.isView(ids) && ids.buffer instanceof ArrayBuffer
        ? [ids.buffer]
        : []),
    ],
  };
}

/** Parses a numeric column statistic, which is a bigint for int64 columns */
function parseStatistic(value: unknown): number | undefined {
  if (typeof value !== "number" && typeof value !== "bigint") {
    return undefined;
  }
  return NumberUtils.tryParseFinite(value, { requireSafeBigInt: true });
}

async function handleRangeRequest(
  request: ParquetRangeRequest,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _onProgress: (progress: number, total: number) => void,
): Promise<{
  response: ParquetRangeResponse;
  transfer?: Transferable[];
}> {
  const buffer = await openParquet(request.source);
  const metadata = await parquetMetadataAsync(buffer);
  if (request.axis !== undefined) {
    const { bbox } = GeoParquetUtils.readGeoColumns(metadata).find(
      ({ name }) => name === request.column,
    ) ?? { bbox: undefined };
    return {
      response: {
        op: "range",
        range:
          bbox !== undefined
            ? request.axis === "x"
              ? [bbox[0], bbox[2]]
              : [bbox[1], bbox[3]]
            : undefined,
      },
    };
  }
  let vmin = Infinity;
  let vmax = -Infinity;
  for (const rowGroup of metadata.row_groups) {
    const columnChunk = rowGroup.columns.find(
      (column) => column.meta_data?.path_in_schema.join(".") === request.column,
    );
    if (columnChunk === undefined) {
      throw new Error(`Column "${request.column}" not found in Parquet file`);
    }
    if (columnChunk.meta_data?.statistics === undefined) {
      return { response: { op: "range", range: undefined } };
    }
    const { min_value, max_value } = columnChunk.meta_data.statistics;
    const min = parseStatistic(min_value);
    const max = parseStatistic(max_value);
    if (min === undefined || max === undefined) {
      return { response: { op: "range", range: undefined } };
    }
    if (min < vmin) {
      vmin = min;
    }
    if (max > vmax) {
      vmax = max;
    }
  }
  return {
    response: {
      op: "range",
      range:
        Number.isFinite(vmin) && Number.isFinite(vmax)
          ? [vmin, vmax]
          : undefined,
    },
  };
}
