import {
  type AsyncBuffer,
  asyncBufferFromUrl,
  parquetMetadataAsync,
} from "hyparquet";

import {
  ArrayUtils,
  type IDArray,
  NumberUtils,
  type ShapesGeometry,
  TableUtils,
  type TypedArrayOrArray,
} from "@tissuumaps/core";

import { PandasMetadataUtils } from "./PandasMetadataUtils";
import { ParquetUtils } from "./ParquetUtils";
import {
  type CoordinateColumn,
  GeoParquetUtils,
} from "./profiles/GeoParquetUtils";
import type { ParquetPart, ParquetSource } from "./types";

export type ParquetRequest<TOp extends string = string> = {
  op: TOp;
};

export type ParquetResponse<TRequest extends ParquetRequest> = {
  op: TRequest["op"];
};

export type ParquetGeoRequest = ParquetRequest<"geo"> & {
  source: ParquetSource;
};

export type ParquetGeoResponse = ParquetResponse<ParquetGeoRequest> & {
  hasShapesColumn: boolean;
};

export type ParquetFileRequest = ParquetRequest<"file"> & {
  /** The files of the table, whose rows follow one another in this order */
  parts: ParquetSource[];
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
  /** See {@link ParquetFileRequest.parts} */
  parts: ParquetSource[];
  column: string;
};

export type ParquetColumnResponse = ParquetResponse<ParquetColumnRequest> & {
  data: TypedArrayOrArray<unknown>;
};

export type ParquetCoordinatesRequest = ParquetRequest<"coordinates"> & {
  /** See {@link ParquetFileRequest.parts} */
  parts: ParquetSource[];
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
  /** See {@link ParquetFileRequest.parts} */
  parts: ParquetSource[];
  /** The column to read, or the geometry column when an axis is given */
  column: string;
  /** Set to read the range of one axis of a point geometry column */
  axis?: "x" | "y";
};

export type ParquetRangeResponse = ParquetResponse<ParquetRangeRequest> & {
  range: [number, number] | undefined;
};

export type ParquetWorkerRequest =
  | ParquetGeoRequest
  | ParquetFileRequest
  | ParquetColumnRequest
  | ParquetCoordinatesRequest
  | ParquetShapesRequest
  | ParquetRangeRequest;

export type ParquetWorkerResponse =
  | ParquetGeoResponse
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
        case "geo":
          result = await handleGeoRequest(event.data);
          break;
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
  return Promise.reject(new Error("A URL or file is required to load data"));
}

async function openParquetPart(source: ParquetSource): Promise<ParquetPart> {
  const buffer = await openParquet(source);
  const metadata = await parquetMetadataAsync(buffer);
  return { buffer, metadata };
}

function openParquetParts(sources: ParquetSource[]): Promise<ParquetPart[]> {
  return Promise.all(sources.map(openParquetPart));
}

async function readIdsAndNames(
  parts: ParquetPart[],
  idColumn: string | undefined,
  nameColumn: string | undefined,
  onProgress: (progress: number, total: number) => void,
): Promise<{ ids: IDArray | undefined; names: string[] | undefined }> {
  // Without an ID column, rows are keyed by the pandas index, if there is one
  const indexColumn =
    idColumn === undefined
      ? PandasMetadataUtils.readIndexColumn(parts[0]!.metadata)
      : undefined;
  const keyColumn = idColumn ?? indexColumn;
  let idTotal = 0,
    nameTotal = 0,
    idProgress = 0,
    nameProgress = 0;
  let idsPromise: Promise<IDArray | undefined> | undefined;
  if (keyColumn !== undefined) {
    idsPromise = ParquetUtils.readColumn(
      parts,
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
        throw new Error(`ID column '${keyColumn}' does not hold item IDs`, {
          cause: error,
        });
      }
    });
    if (indexColumn !== undefined) {
      // the pandas index was not asked for, so it must not fail the read
      idsPromise = idsPromise.catch((error: unknown) => {
        console.warn("Keying rows by row number instead", error);
        return undefined;
      });
    }
  }
  const nameDataPromise =
    nameColumn !== undefined
      ? ParquetUtils.readColumn(parts, nameColumn, (progress, total) => {
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
      `ID column '${keyColumn}' has duplicate values, keying rows by row number instead`,
    );
    return { ids: undefined, names };
  }
  return { ids, names };
}

async function handleGeoRequest(request: ParquetGeoRequest): Promise<{
  response: ParquetGeoResponse;
  transfer?: Transferable[];
}> {
  const { metadata } = await openParquetPart(request.source);
  let hasShapesColumn = true;
  try {
    GeoParquetUtils.getShapesColumn(metadata, undefined);
  } catch {
    hasShapesColumn = false; // no geometry column, or one of points
  }
  return { response: { op: "geo", hasShapesColumn } };
}

async function handleFileRequest(
  request: ParquetFileRequest,
  onProgress: (progress: number, total: number) => void,
): Promise<{
  response: ParquetFileResponse;
  transfer?: Transferable[];
}> {
  const parts = await openParquetParts(request.parts);
  const { ids, names } = await readIdsAndNames(
    parts,
    request.idColumn,
    request.nameColumn,
    onProgress,
  );
  const { metadata } = parts[0]!;
  const { columns, coordinateColumns } = GeoParquetUtils.replaceGeometryColumns(
    metadata,
    ParquetUtils.getColumns(metadata),
  );
  return {
    response: {
      op: "file",
      numRows: ParquetUtils.getTotalNumRows(parts),
      columns,
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

async function handleCoordinatesRequest(
  request: ParquetCoordinatesRequest,
  onProgress: (progress: number, total: number) => void,
): Promise<{
  response: ParquetCoordinatesResponse;
  transfer?: Transferable[];
}> {
  const parts = await openParquetParts(request.parts);
  const { x, y } = await GeoParquetUtils.readCoordinateColumns(
    parts,
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
  const parts = await openParquetParts(request.parts);
  const data = await ParquetUtils.readColumn(parts, request.column, onProgress);
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
  const part = await openParquetPart(request.source);
  const { buffer, metadata } = part;
  const geoColumn = GeoParquetUtils.getShapesColumn(
    metadata,
    request.geometryColumn,
  );
  // Progress only tracks the geometry, which dwarfs the ID and name columns
  const { ids: rowIds, names: rowNames } = await readIdsAndNames(
    [part],
    request.idColumn,
    request.nameColumn,
    () => {},
  );
  const { geometry, ids, names } = await GeoParquetUtils.readShapes(
    buffer,
    metadata,
    geoColumn,
    { ids: rowIds, names: rowNames, onProgress },
  );
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

/**
 * Reads the value range of a column, or of one axis of a point geometry
 * column, from the file metadata where possible, and otherwise computes it
 * from the column values
 */
async function handleRangeRequest(
  request: ParquetRangeRequest,
  onProgress: (progress: number, total: number) => void,
): Promise<{
  response: ParquetRangeResponse;
  transfer?: Transferable[];
}> {
  const parts = await openParquetParts(request.parts);
  if (request.axis !== undefined) {
    const { axis } = request;
    // an empty part has no bounds, and needs none
    const axisRanges = parts
      .filter(({ metadata }) => ParquetUtils.getNumRows(metadata) > 0)
      .map(({ metadata }) =>
        GeoParquetUtils.getAxisRange(metadata, request.column, axis),
      );
    if (
      axisRanges.length > 0 &&
      axisRanges.every((axisRange) => axisRange !== undefined)
    ) {
      const range: [number, number] = [
        Math.min(...axisRanges.map(([min]) => min)),
        Math.max(...axisRanges.map(([, max]) => max)),
      ];
      return { response: { op: "range", range } };
    }
    const coordinates = await GeoParquetUtils.readCoordinateColumns(
      parts,
      request.column,
      onProgress,
    );
    return {
      response: {
        op: "range",
        range: await TableUtils.computeValueRange(coordinates[request.axis]),
      },
    };
  }
  let vmin = Infinity;
  let vmax = -Infinity;
  let hasStatistics = true;
  for (const rowGroup of parts.flatMap(({ metadata }) => metadata.row_groups)) {
    const columnChunk = rowGroup.columns.find(
      (column) => column.meta_data?.path_in_schema.join(".") === request.column,
    );
    if (columnChunk === undefined) {
      throw new Error(
        `Column '${request.column}' not found in the Parquet file`,
      );
    }
    const statistics = columnChunk.meta_data?.statistics;
    if (statistics?.null_count === rowGroup.num_rows) {
      continue;
    }
    const min = parseStatistic(statistics?.min_value);
    const max = parseStatistic(statistics?.max_value);
    if (min === undefined || max === undefined) {
      hasStatistics = false;
      break;
    }
    if (min < vmin) {
      vmin = min;
    }
    if (max > vmax) {
      vmax = max;
    }
  }
  if (hasStatistics) {
    return {
      response: { op: "range", range: vmin <= vmax ? [vmin, vmax] : undefined },
    };
  }
  const element = ParquetUtils.getColumnElement(
    parts[0]!.metadata,
    request.column,
  );
  // a column read as a plain array has no numeric range
  if (
    element === undefined ||
    ParquetUtils.getColumnArrayType(element) === undefined
  ) {
    return { response: { op: "range", range: undefined } };
  }
  const values = await ParquetUtils.readColumn(
    parts,
    request.column,
    onProgress,
  );
  return {
    response: {
      op: "range",
      range: await TableUtils.computeValueRange(values),
    },
  };
}
