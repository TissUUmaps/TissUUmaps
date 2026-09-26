import type { TypedArrayOrArray } from "@tissuumaps/core";

import type { HierarchicalTableColumn } from "../HierarchicalTable";

/** Opens the table of a file or URL; sent once, first */
export type HierarchicalTableOpenRequest = {
  op: "open";
  source: File | string;
};

/** The columns and row count of the opened table */
export type HierarchicalTableOpenResponse = {
  columns: HierarchicalTableColumn[];
  numRows: number;
};

/** Reads the values of a column, see {@link HierarchicalTable.readColumn} */
export type HierarchicalTableColumnRequest = {
  op: "column";
  column: string;
  numRows?: number;
};

/** The values of a column */
export type HierarchicalTableColumnResponse = {
  data: TypedArrayOrArray<unknown>;
};

/** Reads the value range of a column, see {@link HierarchicalTable.readRange} */
export type HierarchicalTableRangeRequest = {
  op: "range";
  column: string;
  numRows?: number;
};

/** The value range of a column */
export type HierarchicalTableRangeResponse = {
  range: [number, number] | undefined;
};

/** A request to the worker */
export type HierarchicalTableWorkerRequest =
  | HierarchicalTableOpenRequest
  | HierarchicalTableColumnRequest
  | HierarchicalTableRangeRequest;

/** A successful response of the worker */
export type HierarchicalTableWorkerResponse =
  | HierarchicalTableOpenResponse
  | HierarchicalTableColumnResponse
  | HierarchicalTableRangeResponse;

/** A request to the worker, answered by the message with the same id */
export type HierarchicalTableWorkerRequestMessage = {
  id: number;
} & HierarchicalTableWorkerRequest;

/** The response to a request, or the error it failed with */
export type HierarchicalTableWorkerResponseMessage = { id: number } & (
  HierarchicalTableWorkerResponse | { error: string }
);
