// @vitest-environment jsdom
// OpenSeadragon touches the DOM when imported
import { vec2 } from "gl-matrix";
import OpenSeadragon from "openseadragon";
import { describe, expect, it } from "vitest";

import {
  type Rect,
  type SimilarityTransform,
  identityTransform,
} from "@tissuumaps/core";

import { WebGLUtils } from "../webgl/WebGLUtils";
import { OpenSeadragonUtils } from "./OpenSeadragonUtils";

const contentSize = { x: 100, y: 80 };

/**
 * Simulates where OpenSeadragon places the four corners of a tiled image,
 * given the geometry computed by
 * {@link OpenSeadragonUtils.getTiledImageTransform}
 *
 * OpenSeadragon renders a tiled image as:
 *   1. Place at (x, y) with the given width (height = width * aspect)
 *   2. If flipped, mirror the tile content horizontally within those bounds
 *   3. Rotate around the center of the un-rotated bounds by `rotation` degrees
 *
 * This mirrors OpenSeadragon 6.1.0, where `TiledImage._getRotationPoint()`
 * returns the center of `getBoundsNoRotate()` and the drawers mirror tiles
 * whose `flipped` property is set, i.e. within the image bounds. Should an
 * OpenSeadragon upgrade change either, this model — and with it
 * {@link OpenSeadragonUtils.getTiledImageTransform} — has to be revisited.
 */
function osdCorners(
  geom: {
    flip: boolean;
    width: number;
    rotation: number;
    position: { x: number; y: number };
  },
  contentSize: { x: number; y: number },
): vec2[] {
  const aspect = contentSize.y / contentSize.x;
  const w = geom.width;
  const h = w * aspect;
  // Corners before rotation, relative to position
  let corners: vec2[] = [
    vec2.fromValues(0, 0),
    vec2.fromValues(w, 0),
    vec2.fromValues(w, h),
    vec2.fromValues(0, h),
  ];
  if (geom.flip) {
    // OpenSeadragon flips around the image center: x' = w - x
    corners = corners.map((c) => vec2.fromValues(w - c[0], c[1]));
  }
  const cx = w / 2;
  const cy = h / 2;
  const rad = (geom.rotation * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return corners.map((c) => {
    const dx = c[0] - cx;
    const dy = c[1] - cy;
    return vec2.fromValues(
      geom.position.x + cx + dx * cos - dy * sin,
      geom.position.y + cy + dx * sin + dy * cos,
    );
  });
}

/**
 * Computes where the WebGL renderers place the same four corners, using the
 * data → world matrix they upload to their shaders
 */
function webglCorners(
  objectTransform: SimilarityTransform,
  layerTransform: SimilarityTransform,
  contentSize: { x: number; y: number },
): vec2[] {
  const dataToWorldMatrix = WebGLUtils.createDataToWorldMatrix(
    objectTransform,
    layerTransform,
  );
  const corners = [
    vec2.fromValues(0, 0),
    vec2.fromValues(contentSize.x, 0),
    vec2.fromValues(contentSize.x, contentSize.y),
    vec2.fromValues(0, contentSize.y),
  ];
  return corners.map((c) => {
    const out = vec2.create();
    vec2.transformMat3(out, c, dataToWorldMatrix);
    return out;
  });
}

/**
 * Creates a stand-in for a tiled image that has the given bounds
 *
 * {@link OpenSeadragonUtils.hasBounds} only reads the bounds of a tiled image,
 * so a real one, which needs a viewer, is not required.
 */
function tiledImageWithBounds({
  x,
  y,
  width,
  height,
}: Rect): OpenSeadragon.TiledImage {
  return {
    getBounds: () => new OpenSeadragon.Rect(x, y, width, height),
  } as unknown as OpenSeadragon.TiledImage;
}

describe("OpenSeadragonUtils", () => {
  describe("createPixelTileSource", () => {
    it("declares a single level holding a single tile of the given size", () => {
      expect(
        OpenSeadragonUtils.createPixelTileSource(
          { width: 300, height: 120 },
          "pixel-url",
        ),
      ).toMatchObject({
        width: 300,
        height: 120,
        tileSize: 300,
        minLevel: 0,
        maxLevel: 0,
      });
    });

    it("sizes the tile to the larger side", () => {
      expect(
        OpenSeadragonUtils.createPixelTileSource(
          { width: 40, height: 90 },
          "pixel-url",
        ),
      ).toMatchObject({ tileSize: 90 });
    });

    it("serves the pixel URL for every tile", () => {
      // TileSourceConfig is an opaque object type, so narrow it to what the
      // pixel tile source is known to provide
      const { getTileUrl } = OpenSeadragonUtils.createPixelTileSource(
        { width: 40, height: 90 },
        "pixel-url",
      ) as { getTileUrl: (level: number, x: number, y: number) => string };
      expect(getTileUrl(0, 0, 0)).toBe("pixel-url");
      expect(getTileUrl(3, 2, 1)).toBe("pixel-url");
    });
  });

  describe("fixTileCacheCounter", () => {
    it("keeps the count of loaded cache records in sync when tiles are recolored", () => {
      OpenSeadragonUtils.fixTileCacheCounter();
      const tileCache = new OpenSeadragon.TileCache({});
      const bounds = new OpenSeadragon.Rect(0, 0, 1, 1);
      const tile = new OpenSeadragon.Tile(
        0,
        0,
        0,
        bounds,
        true,
        "",
        undefined,
        false,
        {},
        bounds,
        "",
        "tile",
      );
      // The cache internals involved are untyped; mirror what OpenSeadragon
      // does when a tile-invalidated handler sets new data for a tile
      const internalTile = tile as unknown as {
        tiledImage: unknown;
        addCache: (
          key: string,
          data: unknown,
          type: string,
          setAsMainCache: boolean,
          safely: boolean,
        ) => void;
        buildDistinctMainCacheKey: () => string;
      };
      const internalTileCache = tileCache as unknown as {
        _cachesLoaded: Record<string, unknown>;
        injectCache: (options: {
          tile: OpenSeadragon.Tile;
          cache: unknown;
          targetKey: string;
          setAsMainCache: boolean;
        }) => void;
      };
      internalTile.tiledImage = { _tileCache: tileCache };
      tile.loaded = true;
      internalTile.addCache(tile.cacheKey, {}, "test", true, false);
      for (let i = 0; i < 3; i++) {
        const workingCache = (
          new OpenSeadragon.CacheRecord() as unknown as {
            withTileReference: (tile: OpenSeadragon.Tile) => {
              addTile: (
                tile: OpenSeadragon.Tile,
                data: unknown,
                type: string,
              ) => void;
            };
          }
        ).withTileReference(tile);
        workingCache.addTile(tile, {}, "test");
        internalTileCache.injectCache({
          tile,
          cache: workingCache,
          targetKey: internalTile.buildDistinctMainCacheKey(),
          setAsMainCache: true,
        });
        expect(tileCache.numCachesLoaded()).toBe(
          Object.keys(internalTileCache._cachesLoaded).length,
        );
      }
    });
  });

  describe("getTiledImageTransform", () => {
    it("returns the content size at the origin for identity transforms", () => {
      const geom = OpenSeadragonUtils.getTiledImageTransform(
        identityTransform,
        identityTransform,
        contentSize,
      );
      expect(geom.flip).toBe(false);
      expect(geom.width).toBeCloseTo(contentSize.x);
      expect(geom.rotation).toBeCloseTo(0);
      expect(geom.position.x).toBeCloseTo(0);
      expect(geom.position.y).toBeCloseTo(0);
    });

    it("applies the object's data → layer scale to the width", () => {
      const geom = OpenSeadragonUtils.getTiledImageTransform(
        { flip: false, scale: 2, rotation: 0, translation: { x: 0, y: 0 } },
        identityTransform,
        contentSize,
      );
      expect(geom.width).toBeCloseTo(200);
      expect(geom.rotation).toBeCloseTo(0);
    });

    it("shifts the position for a flip, which OpenSeadragon applies around the image center", () => {
      const geom = OpenSeadragonUtils.getTiledImageTransform(
        { ...identityTransform, flip: true },
        identityTransform,
        contentSize,
      );
      expect(geom.flip).toBe(true);
      expect(geom.width).toBeCloseTo(100);
      // Position must shift left by the width, as the object is flipped
      // around its left edge but OpenSeadragon flips around the center
      expect(geom.position.x).toBeCloseTo(-100);
      expect(geom.position.y).toBeCloseTo(0);
    });

    it("XORs the layer flip with the object flip", () => {
      const flipped = { ...identityTransform, flip: true };
      const geom = OpenSeadragonUtils.getTiledImageTransform(
        flipped,
        flipped,
        contentSize,
      );
      expect(geom.flip).toBe(false);
    });
  });

  describe("hasBounds", () => {
    const bounds = { x: 10, y: -20, width: 300, height: 120 };

    it("returns true for a tiled image with exactly the given bounds", () => {
      expect(
        OpenSeadragonUtils.hasBounds(tiledImageWithBounds(bounds), bounds),
      ).toBe(true);
    });

    it("tolerates a rounding error in the derived height", () => {
      const tiledImage = tiledImageWithBounds({
        ...bounds,
        height: bounds.height * (1 + 1e-14),
      });
      expect(OpenSeadragonUtils.hasBounds(tiledImage, bounds)).toBe(true);
    });

    it.each([
      ["x", { ...bounds, x: bounds.x + 1 }],
      ["y", { ...bounds, y: bounds.y - 1 }],
      ["width", { ...bounds, width: bounds.width + 1 }],
      ["height", { ...bounds, height: bounds.height - 1 }],
    ])("returns false if the %s differs", (_, otherBounds) => {
      expect(
        OpenSeadragonUtils.hasBounds(tiledImageWithBounds(otherBounds), bounds),
      ).toBe(false);
    });

    it("scales the tolerance with the size of the bounds", () => {
      const offset = 1e-6;
      const large = { x: 0, y: 0, width: 1e6, height: 1e6 };
      const small = { x: 0, y: 0, width: 1, height: 1 };
      expect(
        OpenSeadragonUtils.hasBounds(
          tiledImageWithBounds({ ...large, x: offset }),
          large,
        ),
      ).toBe(true);
      expect(
        OpenSeadragonUtils.hasBounds(
          tiledImageWithBounds({ ...small, x: offset }),
          small,
        ),
      ).toBe(false);
    });
  });

  describe("OpenSeadragon/WebGL geometry parity", () => {
    it.each([
      {
        name: "scale + rotation, no flip",
        objectTransform: {
          flip: false,
          scale: 2,
          rotation: 45,
          translation: { x: 10, y: 5 },
        },
        layerTransform: identityTransform,
      },
      {
        name: "flip + rotation",
        objectTransform: {
          flip: true,
          scale: 1,
          rotation: 30,
          translation: { x: 0, y: 0 },
        },
        layerTransform: identityTransform,
      },
      {
        name: "flip + scale + rotation + translation",
        objectTransform: {
          flip: true,
          scale: 2,
          rotation: 60,
          translation: { x: -5, y: 10 },
        },
        layerTransform: identityTransform,
      },
      {
        name: "layer flip + data no-flip",
        objectTransform: {
          flip: false,
          scale: 1.5,
          rotation: 0,
          translation: { x: 3, y: 7 },
        },
        layerTransform: {
          flip: true,
          scale: 1,
          rotation: 0,
          translation: { x: 0, y: 0 },
        },
      },
      {
        name: "both flip + layer rotation",
        objectTransform: {
          flip: true,
          scale: 1,
          rotation: 0,
          translation: { x: 0, y: 0 },
        },
        layerTransform: {
          flip: true,
          scale: 1,
          rotation: 90,
          translation: { x: 0, y: 0 },
        },
      },
      {
        name: "layer scale+rotation + data translation",
        objectTransform: {
          flip: false,
          scale: 1.5,
          rotation: 0,
          translation: { x: 20, y: 10 },
        },
        layerTransform: {
          flip: false,
          scale: 2,
          rotation: 30,
          translation: { x: 5, y: 3 },
        },
      },
      {
        name: "layer scale+rotation + data scale+rotation+translation",
        objectTransform: {
          flip: false,
          scale: 2,
          rotation: 15,
          translation: { x: -5, y: 10 },
        },
        layerTransform: {
          flip: false,
          scale: 3,
          rotation: 45,
          translation: { x: 10, y: -7 },
        },
      },
      {
        name: "flip + layer scale+rotation+translation + data translation",
        objectTransform: {
          flip: true,
          scale: 1.5,
          rotation: 20,
          translation: { x: 8, y: -3 },
        },
        layerTransform: {
          flip: false,
          scale: 2,
          rotation: 60,
          translation: { x: -4, y: 12 },
        },
      },
    ] as {
      name: string;
      objectTransform: SimilarityTransform;
      layerTransform: SimilarityTransform;
    }[])(
      "OpenSeadragon corners match WebGL corners: $name",
      ({ objectTransform, layerTransform }) => {
        const geom = OpenSeadragonUtils.getTiledImageTransform(
          objectTransform,
          layerTransform,
          contentSize,
        );
        const osd = osdCorners(geom, contentSize);
        const webgl = webglCorners(
          objectTransform,
          layerTransform,
          contentSize,
        );
        for (let i = 0; i < 4; i++) {
          expect(osd[i]![0]).toBeCloseTo(webgl[i]![0], 3);
          expect(osd[i]![1]).toBeCloseTo(webgl[i]![1], 3);
        }
      },
    );
  });
});
