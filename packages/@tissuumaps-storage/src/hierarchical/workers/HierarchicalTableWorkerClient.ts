import type { TypedArrayOrArray } from "@tissuumaps/core";

import type {
  HierarchicalTable,
  HierarchicalTableColumn,
} from "../HierarchicalTable";
import type {
  HierarchicalTableColumnResponse,
  HierarchicalTableOpenResponse,
  HierarchicalTableRangeResponse,
  HierarchicalTableWorkerRequest,
  HierarchicalTableWorkerResponse,
  HierarchicalTableWorkerResponseMessage,
} from "./messages";

/**
 * A {@link HierarchicalTable} served by a Web Worker running
 * {@link serveHierarchicalTable}
 *
 * The worker stays alive between requests: each start would reload the
 * container library and the store metadata.
 */
export class HierarchicalTableWorkerClient implements HierarchicalTable {
  readonly columns: HierarchicalTableColumn[];
  readonly numRows: number;
  private readonly _channel: WorkerChannel;

  private constructor(
    channel: WorkerChannel,
    columns: HierarchicalTableColumn[],
    numRows: number,
  ) {
    this._channel = channel;
    this.columns = columns;
    this.numRows = numRows;
  }

  /**
   * Opens a hierarchical table in a worker
   *
   * @param worker - A freshly started worker running
   * {@link serveHierarchicalTable}; terminated if opening fails
   * @param source - The file or URL to open
   * @param options - Optional abort signal
   * @returns The client, which owns the worker
   * @throws Error if the worker fails to open the source
   */
  static async open(
    worker: Worker,
    source: File | string,
    options?: { signal?: AbortSignal },
  ): Promise<HierarchicalTableWorkerClient> {
    const { signal } = options ?? {};
    const channel = new WorkerChannel(worker);
    try {
      signal?.throwIfAborted();
      const { columns, numRows } = (await channel.request(
        { op: "open", source },
        { signal },
      )) as HierarchicalTableOpenResponse;
      return new HierarchicalTableWorkerClient(channel, columns, numRows);
    } catch (error) {
      channel.terminate();
      throw error;
    }
  }

  async readColumn(
    query: string,
    options?: { numRows?: number; signal?: AbortSignal },
  ): Promise<TypedArrayOrArray<unknown>> {
    const { numRows, signal } = options ?? {};
    signal?.throwIfAborted();
    const { data } = (await this._channel.request(
      { op: "column", column: query, numRows },
      { signal },
    )) as HierarchicalTableColumnResponse;
    return data;
  }

  async readRange(
    query: string,
    options?: { numRows?: number; signal?: AbortSignal },
  ): Promise<[number, number] | undefined> {
    const { numRows, signal } = options ?? {};
    signal?.throwIfAborted();
    const { range } = (await this._channel.request(
      { op: "range", column: query, numRows },
      { signal },
    )) as HierarchicalTableRangeResponse;
    return range;
  }

  /** Terminates the worker, rejecting all pending requests */
  close(): void {
    this._channel.terminate();
  }
}

type PendingRequest = {
  resolve: (response: HierarchicalTableWorkerResponse) => void;
  reject: (reason: unknown) => void;
  signal: AbortSignal | undefined;
  onAbort: () => void;
};

/**
 * Correlates the requests to a worker running {@link serveHierarchicalTable}
 * with its responses by id
 */
class WorkerChannel {
  private readonly _worker: Worker;
  private readonly _pendingRequests = new Map<number, PendingRequest>();
  private _nextRequestId = 0;
  private _terminated = false;

  /**
   * @param worker - The worker; owned by the channel from now on
   */
  constructor(worker: Worker) {
    this._worker = worker;
    this._worker.onmessage = (
      event: MessageEvent<HierarchicalTableWorkerResponseMessage>,
    ) => {
      const { id, ...response } = event.data;
      const pendingRequest = this._settle(id);
      if (pendingRequest === undefined) {
        return;
      }
      if ("error" in response) {
        pendingRequest.reject(new Error(response.error));
      } else {
        pendingRequest.resolve(response);
      }
    };
    this._worker.onerror = (event) => {
      // a worker script that fails to load fires an error without message
      this._rejectAll(new Error(event.message || "The worker failed."));
      this.terminate();
    };
    this._worker.onmessageerror = () => {
      this._rejectAll(new Error("Failed to deserialize worker response."));
      this.terminate();
    };
  }

  /**
   * Sends a request to the worker
   *
   * Aborting only rejects the promise: the worker may not be able to
   * interrupt the request, so its late response is dropped.
   *
   * @param request - The request
   * @param options - Optional abort signal
   * @returns The response of the worker
   * @throws Error if the worker has been terminated, or if the request fails
   * in the worker
   */
  request(
    request: HierarchicalTableWorkerRequest,
    options?: { signal?: AbortSignal },
  ): Promise<HierarchicalTableWorkerResponse> {
    const { signal } = options ?? {};
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    if (this._terminated) {
      return Promise.reject(new Error("Worker has been terminated"));
    }
    const id = this._nextRequestId++;
    return new Promise((resolve, reject) => {
      const onAbort = () => {
        this._pendingRequests.delete(id);
        reject(signal!.reason as Error);
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      this._pendingRequests.set(id, { resolve, reject, signal, onAbort });
      this._worker.postMessage({ ...request, id });
    });
  }

  /** Terminates the worker, rejecting all pending requests; idempotent */
  terminate(): void {
    if (this._terminated) {
      return;
    }
    this._terminated = true;
    this._rejectAll(new Error("Worker has been terminated"));
    this._worker.terminate();
  }

  private _settle(id: number): PendingRequest | undefined {
    const pendingRequest = this._pendingRequests.get(id);
    if (pendingRequest !== undefined) {
      this._pendingRequests.delete(id);
      pendingRequest.signal?.removeEventListener(
        "abort",
        pendingRequest.onAbort,
      );
    }
    return pendingRequest;
  }

  private _rejectAll(error: Error): void {
    for (const id of Array.from(this._pendingRequests.keys())) {
      this._settle(id)?.reject(error);
    }
  }
}
