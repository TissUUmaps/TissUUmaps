import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

import { CSVTableDataProvider } from "./CSVTableDataProvider";
import type { CSVTableDataSource } from "./CSVTableDataSource";

/** The part of `FileReader` that PapaParse uses to read local files */
class FakeFileReader {
  onload: ((event: { target: { result: string } }) => void) | null = null;
  onerror: ((event: { target: { error: unknown } }) => void) | null = null;

  readAsText(blob: Blob): void {
    blob.text().then(
      (result) => this.onload?.({ target: { result } }),
      (error: unknown) => this.onerror?.({ target: { error } }),
    );
  }
}

/** A workspace directory holding one CSV file, `table.csv` */
function makeWorkspace(text: string): FileSystemDirectoryHandle {
  return {
    kind: "directory",
    getFileHandle: (name: string) =>
      name === "table.csv"
        ? Promise.resolve({
            kind: "file",
            name,
            getFile: () => Promise.resolve(new File([text], name)),
          })
        : Promise.reject(new DOMException(name, "NotFoundError")),
  } as unknown as FileSystemDirectoryHandle;
}

async function loadCSV(
  text: string,
  dataSource: Partial<CSVTableDataSource> = {},
) {
  const provider = new CSVTableDataProvider();
  const workspace = makeWorkspace(text);
  const normalizedDataSource = provider.normalize(
    { type: "csv", source: "/table.csv", ...dataSource },
    workspace,
    null,
  );
  return await provider.load(normalizedDataSource, { workspace });
}

describe("CSVTableDataProvider", () => {
  // Node has no FileReader, and no Worker either, so PapaParse parses on the
  // calling thread
  beforeAll(() => {
    vi.stubGlobal("FileReader", FakeFileReader);
  });
  afterAll(() => {
    vi.unstubAllGlobals();
  });

  describe("load", () => {
    it("reads missing value markers as NaN in numeric columns", async () => {
      const data = await loadCSV("qv,gene\n1.5,A\nNA,B\n,C\nnan,D\n2,E\n");
      expect(await data.loadValues("qv")).toEqual(
        new Float32Array([1.5, NaN, NaN, NaN, 2]),
      );
    });

    it("reads missing value markers as empty strings in other columns", async () => {
      const data = await loadCSV("gene\nNA\nA\nNA\n");
      expect(await data.loadValues("gene")).toEqual(["", "A", ""]);
    });

    it("reads a column of missing values only as numeric", async () => {
      const data = await loadCSV("qv\nNA\nNaN\n");
      expect(await data.loadValues("qv")).toEqual(new Float32Array([NaN, NaN]));
    });

    it("takes custom missing value markers", async () => {
      const data = await loadCSV("qv\n1\n-\nNA\n", { naValues: ["-"] });
      expect(await data.loadValues("qv")).toEqual(["1", "", "NA"]);
    });

    it("does not apply missing value markers to the ID column", async () => {
      const data = await loadCSV("id\nNA\nB\n", { idColumn: "id" });
      expect(data.getIds()).toEqual(["NA", "B"]);
    });
  });
});
