import type { OMEZarrTileData, OMEZarrTileSource } from "omezarr-tilesource";
import type OpenSeadragon from "openseadragon";

import type {
  CustomTileSource,
  IntOrUintArray,
  LabelsData,
  TileSourceConfig,
} from "@tissuumaps/core";

/**
 * Loaded OME-Zarr label image data
 *
 * Provides a single tile source whose tiles hold label IDs as signed or
 * unsigned integers (see {@link OMEZarrLabelsData.getTileData}). Label IDs are
 * not enumerated up front: they are read per tile, so that arbitrarily large
 * label images can be opened without scanning them.
 */
export class OMEZarrLabelsData implements LabelsData {
  private readonly _tileSource: OMEZarrTileSource;

  /**
   * @param tileSource - The ready tile source of the label image, rendering a
   * single channel
   */
  constructor(tileSource: OMEZarrTileSource) {
    this._tileSource = tileSource;
  }

  /** Returns the tile source of the label image */
  getTileSource(): string | TileSourceConfig | CustomTileSource {
    return this._tileSource;
  }

  /**
   * Extracts the label IDs of an invalidated tile from its raw OME-Zarr data
   *
   * The tile has to belong to an `OMEZarrTileSource` rendering a single
   * channel, whose tile data (`chunks`) holds exactly one two-dimensional
   * (height x width) tile read from the Zarr array (with the channel, z-slice
   * and timepoint already selected) instead of a rendered image.
   *
   * Only integer tiles of up to 32 bits (signed or unsigned) are accepted, as
   * label IDs have to be integers and the renderer resolves them as such;
   * 64-bit integers cannot be represented without loss.
   *
   * @param event - The tile invalidation event
   * @returns The label IDs of the invalidated tile, one per raster pixel in
   * row-major order, along with the width and height of the raster in pixels
   * @throws Error if the tile data does not hold exactly one tile, or if the
   * tile is not an 8-, 16- or 32-bit integer array
   */
  async getTileData(
    event: OpenSeadragon.TileInvalidatedEvent,
  ): Promise<{ values: IntOrUintArray; width: number; height: number }> {
    const data = (await event.getData("ome-zarr")) as OMEZarrTileData;
    if (data.chunks.length !== 1) {
      throw new Error(`Expected a single tile, got ${data.chunks.length}`);
    }
    const tile = data.chunks[0]!;
    return {
      values: OMEZarrLabelsData._getLabelIds(tile.data),
      width: tile.shape[1]!,
      height: tile.shape[0]!,
    };
  }

  /**
   * Loads the label ID at a pixel of the full-resolution label image
   *
   * Loads the full-resolution tile holding the pixel, i.e. a single chunk of
   * the Zarr array, from the plane the tile source draws.
   *
   * @param x - Horizontal pixel position; pixel i spans [i, i + 1)
   * @param y - Vertical pixel position; pixel i spans [i, i + 1)
   * @param options - Optional abort signal
   * @returns The label ID at the pixel, or `undefined` if the position lies
   * outside the label image
   * @throws Error if the tile source does not load exactly one tile, or if the
   * tile is not an 8-, 16- or 32-bit integer array
   */
  async loadLabelId(
    x: number,
    y: number,
    options?: { signal?: AbortSignal },
  ): Promise<number | undefined> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const level = this._tileSource.maxLevel;
    const column = Math.floor(x);
    const row = Math.floor(y);
    if (
      column < 0 ||
      row < 0 ||
      column >= this._tileSource.getWidth(level) ||
      row >= this._tileSource.getHeight(level)
    ) {
      return undefined;
    }
    const tileWidth = this._tileSource.getTileWidth(level);
    const tileHeight = this._tileSource.getTileHeight(level);
    const tileX = Math.floor(column / tileWidth);
    const tileY = Math.floor(row / tileHeight);
    const chunks = await this._tileSource.loadChunks(
      level,
      { x: tileX, y: tileY },
      { signal },
    );
    if (chunks.length !== 1) {
      throw new Error(`Expected a single tile, got ${chunks.length}`);
    }
    const chunk = chunks[0]!;
    const labelIds = OMEZarrLabelsData._getLabelIds(chunk.data);
    return labelIds[
      (row - tileY * tileHeight) * chunk.shape[1]! +
        (column - tileX * tileWidth)
    ];
  }

  /**
   * Returns the data of a tile as label IDs
   *
   * @param data - The data of the tile
   * @returns The data itself
   * @throws Error if the data is not an 8-, 16- or 32-bit integer array
   */
  private static _getLabelIds(data: ArrayBufferView): IntOrUintArray {
    if (
      data instanceof Int8Array ||
      data instanceof Int16Array ||
      data instanceof Int32Array ||
      data instanceof Uint8Array ||
      data instanceof Uint16Array ||
      data instanceof Uint32Array
    ) {
      return data;
    }
    throw new Error(`Data type '${data.constructor.name}' is not supported`);
  }

  close(): void {}
}
