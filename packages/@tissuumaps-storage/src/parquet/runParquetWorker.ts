import type { ProgressCallback } from "@tissuumaps/core";

import type {
  ParquetWorkerMessage,
  ParquetWorkerRequest,
  ParquetWorkerResponseFor,
} from "./parquet.worker";
import ParquetWorker from "./parquet.worker?worker&inline";

export function runParquetWorker<TRequest extends ParquetWorkerRequest>(
  request: TRequest,
  options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
): Promise<ParquetWorkerResponseFor<TRequest>> {
  const { signal, onProgress } = options ?? {};
  if (signal?.aborted) {
    return Promise.reject(signal.reason as Error);
  }
  const worker = new ParquetWorker();
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      worker.terminate();
      reject(signal!.reason as Error);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    worker.onmessage = (event: MessageEvent<ParquetWorkerMessage>) => {
      if ("progress" in event.data) {
        if (onProgress !== undefined) {
          onProgress(event.data.progress, event.data.total);
        }
      } else {
        worker.terminate();
        signal?.removeEventListener("abort", onAbort);
        if ("error" in event.data) {
          reject(new Error(event.data.error));
        } else {
          resolve(event.data as ParquetWorkerResponseFor<TRequest>);
        }
      }
    };
    worker.onerror = (event) => {
      worker.terminate();
      signal?.removeEventListener("abort", onAbort);
      reject(new Error(event.message));
    };
    worker.onmessageerror = () => {
      worker.terminate();
      signal?.removeEventListener("abort", onAbort);
      reject(new Error("Failed to deserialize the worker response"));
    };
    worker.postMessage(request);
  });
}
