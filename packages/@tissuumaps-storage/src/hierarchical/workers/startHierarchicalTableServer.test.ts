import { afterEach, describe, expect, it, vi } from "vitest";

import type { HierarchicalStore } from "../HierarchicalStore";
import type {
  HierarchicalTableWorkerRequestMessage,
  HierarchicalTableWorkerResponseMessage,
} from "./messages";
import { startHierarchicalTableServer } from "./startHierarchicalTableServer";

function createStore(): HierarchicalStore & { closeCalls: number } {
  return {
    get: (path) => {
      if (path === "") {
        return Promise.resolve({ kind: "group", attrs: {}, keys: ["a"] });
      }
      if (path === "a") {
        return Promise.resolve({
          kind: "array",
          shape: [3],
          dataType: "float",
          read: () => Promise.resolve(new Float64Array([1, 2, 3])),
          slice: () => Promise.reject(new Error("unused")),
        });
      }
      return Promise.resolve(null);
    },
    closeCalls: 0,
    close() {
      this.closeCalls++;
    },
  };
}

/** Serves a table from a stubbed worker scope and sends requests to it */
function serve() {
  const posted: {
    message: HierarchicalTableWorkerResponseMessage;
    transfer: Transferable[] | undefined;
  }[] = [];
  let notify: (() => void) | undefined;
  const scope = {
    onmessage: null as ((event: MessageEvent) => void) | null,
    postMessage(
      message: HierarchicalTableWorkerResponseMessage,
      transfer?: Transferable[],
    ) {
      posted.push({ message, transfer });
      notify?.();
    },
  };
  vi.stubGlobal("self", scope);
  const store = createStore();
  const stop = startHierarchicalTableServer(() => Promise.resolve(store));
  const request = async (data: HierarchicalTableWorkerRequestMessage) => {
    const response = new Promise<void>((resolve) => {
      notify = resolve;
    });
    scope.onmessage!({ data } as MessageEvent);
    await response;
    return posted.at(-1)!;
  };
  return { request, stop, scope, store };
}

describe("startHierarchicalTableServer", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("opens the store and responds with its columns", async () => {
    const { request } = serve();
    const { message } = await request({ id: 0, op: "open", source: "a.h5" });
    expect(message).toEqual({
      id: 0,
      op: "open",
      columns: [{ kind: "dataset", path: "a" }],
      numRows: 3,
    });
  });

  it("transfers the buffer of a typed column", async () => {
    const { request } = serve();
    await request({ id: 0, op: "open", source: "a.h5" });
    const { message, transfer } = await request({
      id: 1,
      op: "column",
      column: "a",
      numRows: 3,
    });
    expect(message).toEqual({
      id: 1,
      op: "column",
      data: new Float64Array([1, 2, 3]),
    });
    expect(transfer).toEqual([(message as { data: Float64Array }).data.buffer]);
  });

  it("responds with the range of a column", async () => {
    const { request } = serve();
    await request({ id: 0, op: "open", source: "a.h5" });
    const { message } = await request({ id: 1, op: "range", column: "a" });
    expect(message).toEqual({ id: 1, op: "range", range: [1, 3] });
  });

  it("responds with the error of a failed request", async () => {
    const { request } = serve();
    expect(
      (await request({ id: 0, op: "column", column: "a" })).message,
    ).toEqual({ id: 0, error: "No table has been opened" });
    await request({ id: 1, op: "open", source: "a.h5" });
    expect(
      (await request({ id: 2, op: "open", source: "a.h5" })).message,
    ).toEqual({ id: 2, error: "A table is already open in this worker" });
  });

  it("stops serving and closes the table on teardown", async () => {
    const { request, stop, scope, store } = serve();
    await request({ id: 0, op: "open", source: "a.h5" });
    stop();
    expect(scope.onmessage).toBeNull();
    expect(store.closeCalls).toBe(1);
  });
});
