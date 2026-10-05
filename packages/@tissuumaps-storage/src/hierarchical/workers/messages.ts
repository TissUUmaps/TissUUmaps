import type { TypedArrayOrArray } from "@tissuumaps/core";

import type { HierarchicalTableColumn } from "../HierarchicalTable";

/** Opens the table of a file or URL; sent once, first */
export type HierarchicalTableOpenRequest = {
  op: "open";
  source: File | string;
};

/** The columns and row count of the opened table */
export type HierarchicalTableOpenResponse = {
  op: "open";
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
  op: "column";
  data: TypedArrayOrArray<unknown>;
};

/** A request to the worker */
export type HierarchicalTableWorkerRequest =
  HierarchicalTableOpenRequest | HierarchicalTableColumnRequest;

/** A successful response of the worker */
export type HierarchicalTableWorkerResponse =
  HierarchicalTableOpenResponse | HierarchicalTableColumnResponse;

/** The response to a request, by the operation the request names */
export type HierarchicalTableWorkerResponseFor<
  TRequest extends HierarchicalTableWorkerRequest,
> = Extract<HierarchicalTableWorkerResponse, { op: TRequest["op"] }>;

/** A request to the worker, answered by the message with the same id */
export type HierarchicalTableWorkerRequestMessage = {
  id: number;
} & HierarchicalTableWorkerRequest;

/** The response to a request, or the error it failed with */
export type HierarchicalTableWorkerResponseMessage = { id: number } & (
  HierarchicalTableWorkerResponse | { error: string }
);
