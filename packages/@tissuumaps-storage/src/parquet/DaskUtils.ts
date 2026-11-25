/**
 * Finds the files of a Dask dataset, the directory that Dask's `to_parquet`
 * writes, with one Parquet file per partition, numbered from zero
 * (`part.0.parquet`, `part.1.parquet`, …)
 */
export class DaskUtils {
  /**
   * Matches the name of a partition's file and captures its number, which Dask
   * writes without leading zeros
   */
  private static readonly _partFileNamePattern =
    /^part\.(0|[1-9]\d*)\.parquet$/;

  /** The bytes every Parquet file starts with */
  private static readonly _parquetMagic = "PAR1";

  /**
   * The HTTP statuses that mean a file is missing: besides Not Found and Gone,
   * Forbidden, which object stores answer for a missing key when listing is
   * not allowed, and web servers for a directory without an index page
   */
  private static readonly _missingFileStatuses = new Set([403, 404, 410]);

  /**
   * The HTTP status that means a file is shorter than the requested range, and
   * thus too short to be a Parquet file (e.g. an empty object marking a folder
   * in an object store)
   */
  private static readonly _rangeNotSatisfiableStatus = 416;

  /**
   * Lists the partition files of a Dask dataset directory, in partition order
   *
   * @param directory - The directory of the dataset
   * @param options - Optional abort signal
   * @returns A promise that resolves to the partition files, empty if the
   * directory holds none
   * @throws Error if a partition file is missing, i.e. the partition numbers
   * have a gap
   */
  static async listPartFiles(
    directory: FileSystemDirectoryHandle,
    options?: { signal?: AbortSignal },
  ): Promise<File[]> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const parts: { number: number; handle: FileSystemFileHandle }[] = [];
    for await (const [name, handle] of directory.entries()) {
      signal?.throwIfAborted(); // the iteration does not take a signal
      const match = DaskUtils._partFileNamePattern.exec(name);
      if (match !== null && handle.kind === "file") {
        parts.push({ number: Number(match[1]), handle });
      }
    }
    parts.sort((a, b) => a.number - b.number);
    const gap = parts.findIndex(({ number }, i) => number !== i);
    if (gap >= 0) {
      throw new Error(
        `The Dask dataset directory '${directory.name}' lacks partition file 'part.${gap}.parquet'`,
      );
    }
    const files = await Promise.all(
      parts.map(({ handle }) => handle.getFile()),
    );
    signal?.throwIfAborted(); // getFile() does not throw on abort
    return files;
  }

  /**
   * Finds the Parquet files behind a URL, which names either a Parquet file or
   * a Dask dataset directory
   *
   * A directory cannot be listed over HTTP, so the URL is first checked to be a
   * Parquet file, and otherwise its partition files are requested in order
   * until one is missing. Dask numbers the partitions without gaps, so the
   * first missing one ends the dataset. Each check requests the first bytes of
   * a file only. A file counts as missing if the server answers that it is
   * (see {@link DaskUtils._missingFileStatuses}), or with a response that does
   * not start like a Parquet file (e.g. a directory listing, a web page that a
   * server answers with for any path, or an answer that the file is too short).
   *
   * @param url - The URL of the Parquet file or Dask dataset directory
   * @param options - Optional request headers and abort signal
   * @returns A promise that resolves to the URL itself if it names a Parquet
   * file, otherwise to the URLs of the dataset's partition files, empty if
   * there are none
   * @throws TypeError if a request fails, e.g. because the server cannot be
   * reached
   * @throws Error if the server answers a check with an error other than a
   * missing file or one too short to be a Parquet file (e.g. because of missing
   * authorization, or a server error), as the dataset could not be read
   * completely
   */
  static async findParquetURLs(
    url: string,
    options?: {
      headers?: { [headerName: string]: string };
      signal?: AbortSignal;
    },
  ): Promise<string[]> {
    const { headers, signal } = options ?? {};
    signal?.throwIfAborted();
    if (await DaskUtils._isParquetURL(url, { headers, signal })) {
      return [url];
    }
    const partURLs: string[] = [];
    for (;;) {
      const partURL = DaskUtils._getPartURL(url, partURLs.length);
      if (!(await DaskUtils._isParquetURL(partURL, { headers, signal }))) {
        return partURLs;
      }
      partURLs.push(partURL);
    }
  }

  /** Appends the file name of a partition to the path of a directory URL */
  private static _getPartURL(url: string, partNumber: number): string {
    const partURL = new URL(url);
    partURL.pathname = `${partURL.pathname.replace(/\/$/, "")}/part.${partNumber}.parquet`;
    return partURL.href;
  }

  /**
   * Tells whether a URL names a Parquet file, from the first bytes of its
   * response
   *
   * Only these bytes are requested, and the response is cancelled once they
   * arrive, also from servers that ignore the requested range.
   *
   * @throws Error if the server answers with an error other than a missing
   * file or one too short to be a Parquet file
   */
  private static async _isParquetURL(
    url: string,
    options?: {
      headers?: { [headerName: string]: string };
      signal?: AbortSignal;
    },
  ): Promise<boolean> {
    const { headers, signal } = options ?? {};
    signal?.throwIfAborted();
    const magic = new TextEncoder().encode(DaskUtils._parquetMagic);
    const response = await fetch(url, {
      headers: { ...headers, Range: `bytes=0-${magic.length - 1}` },
      signal,
    });
    if (
      DaskUtils._missingFileStatuses.has(response.status) ||
      response.status === DaskUtils._rangeNotSatisfiableStatus
    ) {
      await response.body?.cancel();
      return false;
    }
    if (!response.ok) {
      await response.body?.cancel();
      throw new Error(
        `Failed to check '${url}': ${response.status} ${response.statusText}`,
      );
    }
    if (response.body === null) {
      return false;
    }
    const reader = response.body.getReader();
    const bytes = new Uint8Array(magic.length);
    let numBytes = 0;
    try {
      while (numBytes < magic.length) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        const chunk = value.subarray(0, magic.length - numBytes);
        bytes.set(chunk, numBytes);
        numBytes += chunk.length;
      }
    } finally {
      await reader.cancel();
    }
    return (
      numBytes === magic.length && bytes.every((byte, i) => byte === magic[i])
    );
  }
}
