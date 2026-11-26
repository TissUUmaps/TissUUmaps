import type { ProgressCallback } from "@tissuumaps/core";

import type {
  GeoJSONWorkerMessage,
  GeoJSONWorkerRequest,
  GeoJSONWorkerResponseFor,
} from "./geojson.worker";
import GeoJSONWorker from "./geojson.worker?worker&inline";

export function runGeoJSONWorker<TRequest extends GeoJSONWorkerRequest>(
  request: TRequest,
  options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
): Promise<GeoJSONWorkerResponseFor<TRequest>> {
  const { signal, onProgress } = options ?? {};
  if (signal?.aborted) {
    return Promise.reject(signal.reason as Error);
  }
  const worker = new GeoJSONWorker();
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      worker.terminate();
      reject(signal!.reason as Error);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    worker.onmessage = (event: MessageEvent<GeoJSONWorkerMessage>) => {
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
          resolve(event.data as GeoJSONWorkerResponseFor<TRequest>);
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
