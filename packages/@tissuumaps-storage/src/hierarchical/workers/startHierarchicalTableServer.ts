import type { HierarchicalStore } from "../HierarchicalStore";
import type { HierarchicalTable } from "../HierarchicalTable";
import { HierarchicalTableReader } from "../HierarchicalTableReader";
import type {
  HierarchicalTableWorkerRequestMessage,
  HierarchicalTableWorkerResponseMessage,
} from "./messages";

/**
 * Starts serving one hierarchical table from the calling Web Worker
 *
 * Call once from a worker entry script. One `open` request opens the store,
 * `column` requests then read from it. Responses carry the id and the
 * operation of their request, so several can be in flight at once.
 *
 * @param openStore - Opens the store of a file or URL
 * @returns A teardown callback that stops serving and closes the open table
 */
export function startHierarchicalTableServer(
  openStore: (source: File | string) => Promise<HierarchicalStore>,
): () => void {
  const ctx = self as unknown as {
    onmessage:
      | ((event: MessageEvent<HierarchicalTableWorkerRequestMessage>) => void)
      | null;
    postMessage: (
      message: HierarchicalTableWorkerResponseMessage,
      transfer?: Transferable[],
    ) => void;
  };

  let table: HierarchicalTable | undefined;

  function getTable(): HierarchicalTable {
    if (table === undefined) {
      throw new Error("No table has been opened");
    }
    return table;
  }

  ctx.onmessage = (event) => {
    void (async () => {
      const { id } = event.data;
      try {
        switch (event.data.op) {
          case "open": {
            if (table !== undefined) {
              throw new Error("A table is already open in this worker");
            }
            table = await HierarchicalTableReader.open(
              await openStore(event.data.source),
            );
            ctx.postMessage({
              id,
              op: "open",
              columns: table.columns,
              numRows: table.numRows,
            });
            break;
          }
          case "column": {
            const data = await getTable().readColumn(event.data.column, {
              numRows: event.data.numRows,
            });
            ctx.postMessage(
              { id, op: "column", data },
              ArrayBuffer.isView(data) && data.buffer instanceof ArrayBuffer
                ? [data.buffer]
                : undefined,
            );
            break;
          }
          default:
            throw new Error("Unknown request");
        }
      } catch (error) {
        ctx.postMessage({
          id,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    })();
  };

  return () => {
    ctx.onmessage = null;
    table?.close();
    table = undefined;
  };
}
