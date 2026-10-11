import { describe, expect, it } from "vitest";

import type { HierarchicalTableColumn } from "../HierarchicalTable";
import { HierarchicalTableWorkerClient } from "./HierarchicalTableWorkerClient";
import type { HierarchicalTableWorkerResponseMessage } from "./messages";

function createFakeWorker() {
  return {
    messages: [] as unknown[],
    terminateCalls: 0,
    onmessage: null as ((event: MessageEvent) => void) | null,
    onerror: null as ((event: ErrorEvent) => void) | null,
    onmessageerror: null as ((event: MessageEvent) => void) | null,
    postMessage(message: unknown) {
      this.messages.push(message);
    },
    terminate() {
      this.terminateCalls++;
    },
    respond(data: HierarchicalTableWorkerResponseMessage) {
      this.onmessage!({ data } as MessageEvent);
    },
  };
}

const columns: HierarchicalTableColumn[] = [
  { kind: "dataset", path: "obs/area" },
];

async function openClient() {
  const worker = createFakeWorker();
  const client = HierarchicalTableWorkerClient.open(
    worker as unknown as Worker,
    "table.h5ad",
  );
  worker.respond({ id: 0, op: "open", columns, numRows: 3 });
  return { worker, client: await client };
}

describe("HierarchicalTableWorkerClient", () => {
  describe("open", () => {
    it("opens the source in the worker", async () => {
      const { worker, client } = await openClient();
      expect(worker.messages).toEqual([
        { op: "open", source: "table.h5ad", id: 0 },
      ]);
      expect(client.columns).toEqual(columns);
      expect(client.numRows).toBe(3);
    });

    it("terminates the worker when opening fails", async () => {
      const worker = createFakeWorker();
      const client = HierarchicalTableWorkerClient.open(
        worker as unknown as Worker,
        "table.h5ad",
      );
      worker.respond({ id: 0, error: "Not an HDF5 file" });
      await expect(client).rejects.toThrow("Not an HDF5 file");
      expect(worker.terminateCalls).toBe(1);
    });

    it("terminates the worker when the signal is already aborted", async () => {
      const worker = createFakeWorker();
      const reason = new Error("aborted");
      await expect(
        HierarchicalTableWorkerClient.open(
          worker as unknown as Worker,
          "table.h5ad",
          { signal: AbortSignal.abort(reason) },
        ),
      ).rejects.toBe(reason);
      expect(worker.messages).toEqual([]);
      expect(worker.terminateCalls).toBe(1);
    });
  });

  describe("readColumn", () => {
    it("rejects with the message of an error response", async () => {
      const { worker, client } = await openClient();
      const data = client.readColumn("obs/area");
      worker.respond({ id: 1, error: "No such column" });
      await expect(data).rejects.toThrow(new Error("No such column"));
    });

    it("requests the column with the row count and returns its data", async () => {
      const { worker, client } = await openClient();
      const data = client.readColumn("obs/area", { numRows: 3 });
      worker.respond({ id: 1, op: "column", data: [1, 2, 3] });
      await expect(data).resolves.toEqual([1, 2, 3]);
      expect(worker.messages[1]).toEqual({
        op: "column",
        column: "obs/area",
        numRows: 3,
        id: 1,
      });
    });

    it("matches responses to requests by id", async () => {
      const { worker, client } = await openClient();
      const first = client.readColumn("a");
      const second = client.readColumn("b");
      worker.respond({ id: 2, op: "column", data: [2, 3] });
      worker.respond({ id: 1, op: "column", data: [0, 1] });
      await expect(first).resolves.toEqual([0, 1]);
      await expect(second).resolves.toEqual([2, 3]);
    });

    it("ignores a response without a pending request", async () => {
      const { worker } = await openClient();
      expect(() =>
        worker.respond({ id: 7, op: "column", data: [] }),
      ).not.toThrow();
    });

    it("rejects with the abort reason and ignores the late response", async () => {
      const { worker, client } = await openClient();
      const controller = new AbortController();
      const reason = new Error("aborted");
      const data = client.readColumn("a", { signal: controller.signal });
      controller.abort(reason);
      await expect(data).rejects.toBe(reason);
      expect(() =>
        worker.respond({ id: 1, op: "column", data: [0, 1] }),
      ).not.toThrow();
    });

    it("rejects without posting when the signal is already aborted", async () => {
      const { worker, client } = await openClient();
      const reason = new Error("aborted");
      await expect(
        client.readColumn("a", { signal: AbortSignal.abort(reason) }),
      ).rejects.toBe(reason);
      expect(worker.messages).toHaveLength(1);
    });

    it("ignores an abort after the response", async () => {
      const { worker, client } = await openClient();
      const controller = new AbortController();
      const data = client.readColumn("a", { signal: controller.signal });
      worker.respond({ id: 1, op: "column", data: [0, 1] });
      controller.abort();
      await expect(data).resolves.toEqual([0, 1]);
    });

    it("rejects all pending requests when the worker fails", async () => {
      const { worker, client } = await openClient();
      const first = client.readColumn("a");
      const second = client.readColumn("b");
      worker.onerror!({ message: "Worker crashed" } as ErrorEvent);
      await expect(first).rejects.toThrow("Worker crashed");
      await expect(second).rejects.toThrow("Worker crashed");
      expect(worker.terminateCalls).toBe(1);
    });

    it("rejects all pending requests when a response cannot be read", async () => {
      const { worker, client } = await openClient();
      const data = client.readColumn("a");
      worker.onmessageerror!({} as MessageEvent);
      await expect(data).rejects.toThrow(
        "Failed to deserialize worker response.",
      );
      expect(worker.terminateCalls).toBe(1);
    });
  });

  describe("close", () => {
    it("rejects pending requests and terminates the worker once", async () => {
      const { worker, client } = await openClient();
      const data = client.readColumn("a");
      client.close();
      client.close();
      await expect(data).rejects.toThrow("Worker has been terminated");
      expect(worker.terminateCalls).toBe(1);
    });

    it("rejects later requests without posting", async () => {
      const { worker, client } = await openClient();
      client.close();
      await expect(client.readColumn("a")).rejects.toThrow(
        "Worker has been terminated",
      );
      expect(worker.messages).toHaveLength(1);
    });
  });
});
