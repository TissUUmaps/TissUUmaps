import { afterEach, describe, expect, it, vi } from "vitest";

import { DaskUtils } from "./DaskUtils";

/** A directory handle listing the given entries, in this order */
function fakeDirectory(
  entries: { name: string; kind: "file" | "directory" }[],
): FileSystemDirectoryHandle {
  return {
    name: "points.parquet",
    async *entries() {
      for (const { name, kind } of entries) {
        await Promise.resolve();
        yield [
          name,
          kind === "file"
            ? { kind, getFile: () => Promise.resolve(new File([], name)) }
            : { kind },
        ];
      }
    },
  } as unknown as FileSystemDirectoryHandle;
}

const parquetBytes = new TextEncoder().encode("PAR1\0\0\0\0PAR1");
const htmlBytes = new TextEncoder().encode("<!doctype html><html></html>");

/**
 * Stubs `fetch` to answer with the given bodies by URL, and with a 404 for
 * any other URL
 *
 * Responses ignore the requested range, like servers without range support,
 * and record whether their body was cancelled.
 */
function stubFetch(bodies: { [url: string]: Uint8Array | number }) {
  const cancelled: string[] = [];
  const fetch = vi.fn((url: string) => {
    const body = bodies[url] ?? 404;
    if (typeof body === "number") {
      return Promise.resolve(new Response(null, { status: body }));
    }
    let offset = 0;
    const stream = new ReadableStream<Uint8Array>({
      // one byte at a time, so that the reader has to read several chunks
      pull(controller) {
        if (offset < body.length) {
          controller.enqueue(body.subarray(offset, offset + 1));
          offset++;
        } else {
          controller.close();
        }
      },
      cancel() {
        cancelled.push(url);
      },
    });
    return Promise.resolve(new Response(stream));
  });
  vi.stubGlobal("fetch", fetch);
  return { fetch, cancelled };
}

describe("DaskUtils", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("listPartFiles", () => {
    const partNames = Array.from({ length: 11 }, (_, i) => `part.${i}.parquet`);

    it("lists the partition files in partition order", async () => {
      const files = await DaskUtils.listPartFiles(
        fakeDirectory([
          { name: "_metadata", kind: "file" },
          { name: "part.11.parquet", kind: "directory" },
          { name: "part.1.parquet.crc", kind: "file" },
          { name: "part.01.parquet", kind: "file" },
          { name: "notes.parquet", kind: "file" },
          ...[...partNames]
            .reverse()
            .map((name) => ({ name, kind: "file" as const })),
        ]),
      );
      expect(files.map(({ name }) => name)).toEqual(partNames);
    });

    it("rejects a directory lacking a partition file", async () => {
      await expect(
        DaskUtils.listPartFiles(
          fakeDirectory([
            { name: "part.0.parquet", kind: "file" },
            { name: "part.2.parquet", kind: "file" },
          ]),
        ),
      ).rejects.toThrow(
        'The Dask dataset directory "points.parquet" lacks partition file part.1.parquet.',
      );
    });

    it("lists no files for a directory without partition files", async () => {
      await expect(
        DaskUtils.listPartFiles(
          fakeDirectory([{ name: "a.parquet", kind: "file" }]),
        ),
      ).resolves.toEqual([]);
    });

    it("rejects when aborted", async () => {
      const controller = new AbortController();
      controller.abort();
      await expect(
        DaskUtils.listPartFiles(fakeDirectory([]), {
          signal: controller.signal,
        }),
      ).rejects.toThrow();
    });
  });

  describe("findParquetURLs", () => {
    const url = "https://data.example/points.parquet";

    it("finds a Parquet file without looking for partitions", async () => {
      const { fetch, cancelled } = stubFetch({ [url]: parquetBytes });
      await expect(
        DaskUtils.findParquetURLs(url, { headers: { Authorization: "x" } }),
      ).resolves.toEqual([url]);
      expect(fetch).toHaveBeenCalledTimes(1);
      expect(fetch).toHaveBeenCalledWith(url, {
        headers: { Authorization: "x", Range: "bytes=0-3" },
        signal: undefined,
      });
      // the server ignored the range; the rest of the body is not read
      expect(cancelled).toEqual([url]);
    });

    it("finds the partitions of a directory until one is missing", async () => {
      stubFetch({
        [`${url}/part.0.parquet`]: parquetBytes,
        [`${url}/part.1.parquet`]: parquetBytes,
        [`${url}/part.3.parquet`]: parquetBytes,
      });
      await expect(DaskUtils.findParquetURLs(url)).resolves.toEqual([
        `${url}/part.0.parquet`,
        `${url}/part.1.parquet`,
      ]);
    });

    it("joins partition names to a directory URL with a trailing slash and a query", async () => {
      stubFetch({
        "https://data.example/points.parquet/part.0.parquet?token=1":
          parquetBytes,
      });
      await expect(
        DaskUtils.findParquetURLs(
          "https://data.example/points.parquet/?token=1",
        ),
      ).resolves.toEqual([
        "https://data.example/points.parquet/part.0.parquet?token=1",
      ]);
    });

    it("takes a response that is not a Parquet file as missing", async () => {
      stubFetch({
        [url]: htmlBytes,
        [`${url}/part.0.parquet`]: parquetBytes,
        [`${url}/part.1.parquet`]: htmlBytes,
      });
      await expect(DaskUtils.findParquetURLs(url)).resolves.toEqual([
        `${url}/part.0.parquet`,
      ]);
    });

    it("takes a file too short for the requested range as not a Parquet file", async () => {
      stubFetch({ [url]: 416, [`${url}/part.0.parquet`]: parquetBytes });
      await expect(DaskUtils.findParquetURLs(url)).resolves.toEqual([
        `${url}/part.0.parquet`,
      ]);
    });

    it("takes a response shorter than the Parquet header as missing", async () => {
      stubFetch({ [url]: new TextEncoder().encode("PA") });
      await expect(DaskUtils.findParquetURLs(url)).resolves.toEqual([]);
    });

    it.each([403, 404, 410])(
      "takes a %d response as a missing file",
      async (status) => {
        stubFetch({
          [url]: status,
          [`${url}/part.0.parquet`]: parquetBytes,
          [`${url}/part.1.parquet`]: status,
        });
        await expect(DaskUtils.findParquetURLs(url)).resolves.toEqual([
          `${url}/part.0.parquet`,
        ]);
      },
    );

    it.each([401, 429, 500, 503])(
      "rejects a %d response rather than reading part of a dataset",
      async (status) => {
        stubFetch({
          [`${url}/part.0.parquet`]: parquetBytes,
          [`${url}/part.1.parquet`]: status,
          [`${url}/part.2.parquet`]: parquetBytes,
        });
        await expect(DaskUtils.findParquetURLs(url)).rejects.toThrow(
          `Failed to check ${url}/part.1.parquet: ${status}`,
        );
      },
    );

    it("rejects when a request fails", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))),
      );
      await expect(DaskUtils.findParquetURLs(url)).rejects.toThrow(
        "Failed to fetch",
      );
    });

    it("rejects when aborted", async () => {
      stubFetch({ [url]: parquetBytes });
      const controller = new AbortController();
      controller.abort();
      await expect(
        DaskUtils.findParquetURLs(url, { signal: controller.signal }),
      ).rejects.toThrow();
    });
  });
});
