import { SourceUtils } from "@tissuumaps/core";

/**
 * Creates an ID for a data object that is unique among the IDs of its kind
 *
 * The ID is the file name of the source with every `.` replaced by `-`, so
 * `cells.ome.zarr` gives `cells-ome-zarr`; if that is taken, it gets the first
 * free numeric suffix, starting at `-2`. Without a source, or for URLs without
 * a path, it is a random UUID, which is assumed to be unique.
 *
 * @param normalizedSource - The normalized source of the data object, if any
 * @param existingIds - The IDs of the existing data objects of the same kind
 * @returns The ID
 * @throws See `SourceUtils.getPathSegments`
 */
export function createDataObjectID(
  normalizedSource: string | undefined,
  existingIds: string[],
): string {
  const fileName =
    normalizedSource !== undefined
      ? SourceUtils.getPathSegments(normalizedSource).at(-1)
      : undefined;
  if (!fileName) {
    return crypto.randomUUID();
  }
  const id = fileName.replaceAll(".", "-");
  const ids = new Set(existingIds);
  if (!ids.has(id)) {
    return id;
  }
  let n = 2;
  while (ids.has(`${id}-${n}`)) {
    n++;
  }
  return `${id}-${n}`;
}
