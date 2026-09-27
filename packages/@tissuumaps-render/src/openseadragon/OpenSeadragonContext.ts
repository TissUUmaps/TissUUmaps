import OpenSeadragon from "openseadragon";

import {
  AsyncUtils,
  type Color,
  type Dims,
  GeometryUtils,
  type OpenSeadragonViewerOptions,
  type Rect,
  type TypedArray,
} from "@tissuumaps/core";

import { OpenSeadragonUtils } from "./OpenSeadragonUtils";

/**
 * Recolors the tiles of a tiled image whose pixels carry values rather than colors
 *
 * `getTileData` extracts the raw pixel values of an invalidated tile, in row-major
 * order, one per raster pixel, along with the width and height of the raster.
 * The raster may cover the full tile size rather than the tile's source bounds,
 * as OpenSeadragon crops it when drawing. `transferValues` writes the packed RGBA
 * color of each value to the pixel at the same index, as `0xAABBGGRR`, i.e.
 * `(a << 24) | (b << 16) | (g << 8) | r` (see
 * {@link OpenSeadragonContext.updateTiledImageDataTransfer}). It is called
 * once per tile, with `pixelBuffer` as long as `values`. The buffer is reused
 * across tiles and still holds the colors of the previous one, so every pixel
 * has to be written.
 */
export type DataTransfer = {
  getTileData: (
    event: OpenSeadragon.TileInvalidatedEvent,
  ) => Promise<{ values: TypedArray; width: number; height: number }>;
  transferValues: (values: TypedArray, pixelBuffer: Uint32Array) => void;
};

/**
 * A wrapper around an OpenSeadragon viewer
 *
 * In addition to creating and destroying the viewer, this class manages viewer
 * options (including temporary overrides while the viewer is animating) and
 * mutations of the viewer's world.
 *
 * Additions are split in two: tile sources are opened concurrently, but the
 * world itself is mutated one operation at a time. World indices are shared
 * mutable state that every addition and removal invalidates, and OpenSeadragon
 * applies the `index` option only once an addition is committed - a microtask or
 * more after it was requested - so an index is valid only while nothing else
 * touches the world in between. Removals therefore go through the same queue as
 * additions, rather than being applied right away. Reordering the world
 * afterwards is not an alternative: the navigator mirrors index changes
 * positionally and on a timer, and corrupts its own world if it has not caught
 * up yet.
 *
 * Queued world mutations are ordered by request, so a caller that removes tiled
 * images before requesting additions still resolves its indices against the
 * world as it is once those removals have been applied.
 *
 * Tiled images can also be recolored: a tiled image whose pixels carry values
 * rather than colors is drawn by mapping each value to an RGBA color (see
 * {@link updateTiledImageDataTransfer}). As OpenSeadragon has no notion of a
 * per-image color mapping, it is applied to the tiles themselves, in a
 * `tile-invalidated` handler installed on the viewer. Data transfers are kept
 * per tile source, rather than per tiled image, which also covers the
 * navigator: it mirrors the world with tiled images of its own, but shares
 * their tile sources, and OpenSeadragon raises the invalidation events of its
 * tiles on this viewer. Its tiles are cached separately, though, so the
 * navigator's mirror of a tiled image is invalidated alongside the original
 * whenever the data transfer changes.
 *
 * The context also owns the world background: an opaque single-tile image at
 * world index 0 that spans the bounds registered by every source via
 * {@link setContentBounds} - the content of each renderer, and content rendered
 * outside OpenSeadragon, such as points and shapes. It thereby fits the world,
 * and with it the viewport, to all content, and it makes the drawer's canvas
 * opaque wherever content is drawn, which additive compositing relies on (see
 * `OpenSeadragonRendererBase.usesAdditiveBlending`). Its color is given on
 * construction and is expected to match the background behind the viewer,
 * which shows outside of it. Renderers insert their anchors after it, i.e. at
 * world index 1 and up. See {@link _resizeBackground} for how it is resized
 * without ever uncovering the canvas.
 */
export class OpenSeadragonContext {
  private static readonly _isLittleEndian =
    new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
  private static readonly _emptyWorldBounds: Rect = {
    x: 0,
    y: 0,
    width: 1,
    height: 1,
  };
  private static readonly _navigationKeyCodes = new Set([
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "KeyW",
    "KeyA",
    "KeyS",
    "KeyD",
    "Equal",
    "Minus",
  ]);

  readonly viewer: OpenSeadragon.Viewer;
  private readonly _tileSourceDataTransfers = new WeakMap<
    OpenSeadragon.TileSource,
    DataTransfer
  >();
  private readonly _tileDataTransfers = new WeakMap<
    OpenSeadragon.Tile,
    DataTransfer
  >();
  private readonly _backgroundPixelUrl: string;
  private readonly _contentBounds = new Map<object, Rect[]>();
  private _animationMemory?: {
    viewerOptions: Partial<OpenSeadragonViewerOptions>;
    tiledImageViewerOptions: WeakMap<
      OpenSeadragon.TiledImage,
      Partial<OpenSeadragonViewerOptions>
    >;
  };
  private _animationStartHandler?: OpenSeadragon.EventHandler<OpenSeadragon.ViewerEvent>;
  private _animationFinishHandler?: OpenSeadragon.EventHandler<OpenSeadragon.ViewerEvent>;
  private _worldMutationQueue: Promise<unknown> = Promise.resolve();
  private _pixelBuffer = new ArrayBuffer(0);
  private _destroyed: boolean = false;
  private _viewportControlledByUser: boolean = false;
  private _background: OpenSeadragon.TiledImage | undefined;
  private _backgroundUpdateQueue: Promise<unknown> = Promise.resolve();

  /**
   * Creates a new OpenSeadragonContext instance and initializes the OpenSeadragon viewer
   *
   * @param viewerElement - DOM element in which the OpenSeadragon viewer is created
   * @param backgroundColor - Color of the world background (see the class
   * documentation), which has to match the background of `viewerElement`
   * @param viewerOptions - Options for configuring the OpenSeadragon viewer (optional)
   */
  constructor(
    viewerElement: HTMLElement,
    backgroundColor: Color,
    viewerOptions?: OpenSeadragonViewerOptions,
  ) {
    this.viewer = new OpenSeadragon.Viewer({
      ...viewerOptions,
      element: viewerElement,
    });
    this._backgroundPixelUrl = OpenSeadragonUtils.createPixelUrl(
      backgroundColor.r,
      backgroundColor.g,
      backgroundColor.b,
      1,
    );
    this.viewer.addHandler("canvas-key", (event) => {
      if (["r", "R", "f"].includes(event.originalEvent.key)) {
        // disable key bindings for rotation and flipping
        event.preventDefaultAction = true;
      } else if (
        !event.preventDefaultAction &&
        OpenSeadragonContext._navigationKeyCodes.has(
          event.originalEvent.code,
        ) &&
        !event.originalEvent.ctrlKey &&
        !event.originalEvent.altKey &&
        !event.originalEvent.metaKey
      ) {
        this._viewportControlledByUser = true;
      }
    });
    // "pan"/"zoom" cannot be attributed to the user - OSD raises them for its
    // own container resizes too - so viewport control is taken from the
    // gestures that navigate it. Input events cannot be used either: they also
    // arrive for gestures that leave the viewport where it is, e.g. drawing a
    // shape on the overlay. Therefore, use canvas events instead. The navigator
    // raises its own events on this viewer, too, so it needs no separate wiring.
    for (const eventName of [
      "canvas-drag",
      "canvas-scroll",
      "canvas-pinch",
      "navigator-click",
      "navigator-drag",
      "navigator-scroll",
    ] as const) {
      this.viewer.addHandler(eventName, () => {
        this._viewportControlledByUser = true;
      });
    }
    // OSD's home button and its "0" key binding raise "home", which hands
    // viewport control back to the renderers just like resetViewport() does.
    this.viewer.addHandler("home", () => {
      this._viewportControlledByUser = false;
    });
    // OSD raises "reset-size" whenever the bounds of its world change, i.e.
    // whenever an anchor or the background is resized (see
    // OpenSeadragonRendererBase._resizeAnchor and _resizeBackground). Fitting
    // the viewport here, rather than per resize, fits it to the whole world.
    this.viewer.addHandler("reset-size", () => {
      if (!this._viewportControlledByUser) {
        this.viewer.viewport.goHome(/* immediately */ true);
      }
    });
    this.viewer.addHandler("tile-invalidated", (event) =>
      this._applyDataTransfer(event).catch((error) => {
        console.error(`Failed to transfer tile data: ${error}`);
      }),
    );
    this.viewer.addHandler("update-viewport", () => this._recolorStaleTiles());
    this._resizeBackground().catch((error) => {
      console.error("Failed to add the OpenSeadragon world background", error);
    });
  }

  /**
   * Installs OpenSeadragon `animation-start` and `animation-finish` handlers
   *
   * On animation start, the current values of all keys of the start options are
   * saved (for the viewer and for each tiled image having such a property) and
   * the start options are applied. On animation finish, the saved values are
   * restored, overridden by the finish options.
   *
   * Only keys of the start options are saved and restored; keys appearing solely
   * in the finish options are applied without ever being reverted. Likewise, only
   * tiled images present at animation start have their values restored.
   *
   * Calling this method again replaces any previously installed handlers.
   *
   * @param viewerAnimationStartOptions - Options to apply when an animation starts
   * @param viewerAnimationFinishOptions - Options to apply when an animation finishes
   */
  configureAnimationHandlers(
    viewerAnimationStartOptions: OpenSeadragonViewerOptions,
    viewerAnimationFinishOptions: OpenSeadragonViewerOptions,
  ): void {
    if (this._animationStartHandler !== undefined) {
      this.viewer.removeHandler("animation-start", this._animationStartHandler);
    }
    if (this._animationFinishHandler !== undefined) {
      this.viewer.removeHandler(
        "animation-finish",
        this._animationFinishHandler,
      );
    }
    this._animationStartHandler = () => {
      this._animationMemory = {
        viewerOptions: {},
        tiledImageViewerOptions: new WeakMap(),
      };
      for (const key of Object.keys(viewerAnimationStartOptions)) {
        // @ts-expect-error: dynamic property access
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        this._animationMemory.viewerOptions[key] = this.viewer[key];
      }
      for (let i = 0; i < this.viewer.world.getItemCount(); i++) {
        const tiledImage = this.viewer.world.getItemAt(i);
        const tiledImageViewerOptions: Partial<OpenSeadragonViewerOptions> = {};
        for (const key of Object.keys(viewerAnimationStartOptions)) {
          if (key in tiledImage) {
            // @ts-expect-error: dynamic property access
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            tiledImageViewerOptions[key] = tiledImage[key];
          }
        }
        this._animationMemory.tiledImageViewerOptions.set(
          tiledImage,
          tiledImageViewerOptions,
        );
      }
      this.setViewerOptions(viewerAnimationStartOptions);
    };
    this._animationFinishHandler = () => {
      this.setViewerOptions({
        ...this._animationMemory?.viewerOptions,
        ...viewerAnimationFinishOptions,
      });
      for (let i = 0; i < this.viewer.world.getItemCount(); i++) {
        const tiledImage = this.viewer.world.getItemAt(i);
        const tiledImageViewerOptions: Partial<OpenSeadragonViewerOptions> = {
          ...this._animationMemory?.tiledImageViewerOptions.get(tiledImage),
          ...viewerAnimationFinishOptions,
        };
        for (const [key, value] of Object.entries(tiledImageViewerOptions)) {
          // @ts-expect-error: dynamic property access
          // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
          tiledImage[key] = value;
        }
      }
      this._animationMemory = undefined;
    };
    this.viewer.addHandler("animation-start", this._animationStartHandler);
    this.viewer.addHandler("animation-finish", this._animationFinishHandler);
  }

  /**
   * Returns whether {@link destroy} has been called
   *
   * Turns `true` as soon as destruction starts, i.e. before pending world
   * mutations have settled and before the viewer is actually destroyed. Further
   * additions are rejected and further removals are ignored from that point on.
   */
  isDestroyed(): boolean {
    return this._destroyed;
  }

  /**
   * Returns the currently visible viewport bounds in world coordinates
   *
   * Reflects the current viewport, not the target of an ongoing animation.
   */
  getViewport(): Rect {
    const viewport = this.viewer.viewport.getBounds(true);
    return {
      x: viewport.x,
      y: viewport.y,
      width: viewport.width,
      height: viewport.height,
    };
  }

  /**
   * Returns the size of the viewer container element, in screen-space pixels
   */
  getContainerSize(): Dims {
    const { x, y } = this.viewer.viewport.getContainerSize();
    return { width: x, height: y };
  }

  /**
   * Applies viewer options to the OpenSeadragon viewer and all existing tiled images
   *
   * For each option key, performs a shallow merge (one level deep) of nested objects
   * on both the viewer instance and every tiled image in the world.
   *
   * @param viewerOptions - Options to apply
   */
  setViewerOptions(viewerOptions: OpenSeadragonViewerOptions): void {
    // TODO allow more than one level (deep nested shallow merge)
    for (const [key, value] of Object.entries(viewerOptions)) {
      // @ts-expect-error: dynamic property access
      if (key in this.viewer && this.viewer[key] !== value) {
        // shallow merge of nested objects (first level only)
        if (typeof value === "object" && value !== null) {
          // @ts-expect-error: dynamic property access
          // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
          this.viewer[key] = { ...this.viewer[key], ...value };
        } else {
          // @ts-expect-error: dynamic property access
          // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
          this.viewer[key] = value;
        }
      }
      for (let i = 0; i < this.viewer.world.getItemCount(); i++) {
        const tiledImage = this.viewer.world.getItemAt(i);
        // @ts-expect-error: dynamic property access
        if (key in tiledImage && tiledImage[key] !== value) {
          // shallow merge of nested objects (first level only)
          if (typeof value === "object" && value !== null) {
            // @ts-expect-error: dynamic property access
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            tiledImage[key] = { ...tiledImage[key], ...value };
          } else {
            // @ts-expect-error: dynamic property access
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            tiledImage[key] = value;
          }
        }
      }
    }
  }

  /**
   * Resolves a tile source specifier to a ready-to-use OpenSeadragon tile source
   *
   * Fetches the image information if the specifier is a URL, awaits promised
   * tile sources, and passes ready tile sources through. Doing this before an
   * addition keeps the loading concurrent, while the additions themselves stay
   * serialized (see {@link addTiledImage}).
   *
   * Rejects as soon as the operation is aborted, so that nothing waits for an
   * abandoned open; the open itself cannot be canceled, and settles unobserved.
   *
   * @param tiledImageOptions - Options containing the tile source to open
   * @param options - Optional abort signal
   * @returns A promise that resolves with the opened tile source
   */
  openTileSource(
    tiledImageOptions: Omit<
      OpenSeadragon.TileSourceSpecifier,
      "success" | "error"
    >,
    options?: { signal?: AbortSignal },
  ): Promise<OpenSeadragon.TileSource> {
    const { signal } = options ?? {};
    const tileSourcePromise = this._openTileSource(tiledImageOptions, {
      signal,
    });
    tileSourcePromise.catch(() => {}); // prevent unhandled rejections in console
    return AsyncUtils.raceSignal(tileSourcePromise, { signal });
  }

  /**
   * Adds a tiled image to the OpenSeadragon viewer
   *
   * The tile source is opened right away, i.e. concurrently with the tile sources
   * of other additions, but the tiled images are added to the world one after
   * another, in the order in which they were requested. The index is resolved in
   * between, when nothing else can shift the world anymore.
   *
   * The tile source may also be a promise of an already opened tile source,
   * which is only awaited once the addition is executed. Callers can thus derive
   * a tile source from another one that is still being opened, without giving up
   * their place in the queue and thereby their world index.
   *
   * Rejects right away once the context has been destroyed.
   *
   * Aborting before the addition is executed skips it entirely. Later than that,
   * the pending tile source load cannot be canceled anymore: the tiled image is
   * added, immediately removed again, and the returned promise rejects with the
   * signal's reason. Aborting a replacement (`tiledImageOptions.replace`) instead
   * resolves, as the replaced tiled image is already gone by then and removing
   * the new one, too, would leave a gap in the world.
   *
   * @param tiledImageOptions - Options for adding the tiled image, including the tile source
   * @param options - Optional abort signal and index resolver. The latter takes
   * precedence over `tiledImageOptions.index`, and appends the tiled image to the
   * world if it returns `undefined`.
   * @returns A promise that resolves with the added tiled image, or rejects if
   * adding it failed or the operation was aborted
   */
  addTiledImage(
    tiledImageOptions: Omit<
      OpenSeadragon.TileSourceSpecifier,
      "success" | "error"
    >,
    options?: {
      signal?: AbortSignal;
      getIndex?: () => number | undefined;
    },
  ): Promise<OpenSeadragon.TiledImage> {
    if (this._destroyed) {
      return Promise.reject(
        new Error("The OpenSeadragon context has been destroyed"),
      );
    }
    const { signal, getIndex } = options ?? {};
    const tileSourcePromise = this.openTileSource(tiledImageOptions, {
      signal,
    });
    tileSourcePromise.catch(() => {}); // prevent unhandled rejections in console
    return this._enqueueWorldMutation(async () => {
      signal?.throwIfAborted();
      const tileSource = await tileSourcePromise; // signal passed above
      return this._addTiledImage(
        { ...tiledImageOptions, tileSource },
        { signal, getIndex },
      );
    });
  }

  /**
   * Removes a tiled image from the OpenSeadragon viewer
   *
   * The removal is queued behind the world mutations requested before it, so
   * that it cannot shift the world while an addition is waiting for
   * OpenSeadragon to apply its index. Does nothing once the context has been
   * destroyed, as the viewer tears down its world itself.
   *
   * @param tiledImage - The tiled image to remove
   * @returns A promise that resolves once the tiled image has been removed.
   */
  removeTiledImage(tiledImage: OpenSeadragon.TiledImage): Promise<void> {
    if (this._destroyed) {
      return Promise.resolve();
    }
    return this._enqueueWorldMutation(() =>
      this.viewer.world.removeItem(tiledImage),
    );
  }

  /**
   * Returns the index of a tiled image in the OpenSeadragon viewer's world
   *
   * @param tiledImage - The tiled image for which to get the index
   * @returns The index of the tiled image, or -1 if it is not in the world
   */
  getTiledImageIndex(tiledImage: OpenSeadragon.TiledImage): number {
    return this.viewer.world.getIndexOfItem(tiledImage);
  }

  /**
   * Updates the data transfer of a tiled image
   *
   * The data transfer is applied per tile: the tile's pixel values are
   * extracted and each is replaced by the color it maps to (see
   * {@link DataTransfer}). Passing `undefined` leaves the tiles as they are. It
   * is applied to every tile of the tiled image, including those loaded later.
   *
   * The data transfer is remembered per tile source, until the tile source is
   * garbage-collected. Tiled images that share a tile source therefore also
   * share a data transfer, with the last one set winning - including the tiled
   * images of the navigator, which mirror those of this viewer and are
   * recolored along with them. As the navigator keeps a tile cache of its own,
   * which the invalidation of `tiledImage` does not reach, its tiled images
   * that share the tile source are invalidated explicitly, too. Those are few
   * low-resolution tiles, so this stays cheap; invalidating the whole viewer
   * would re-run every data transfer on every loaded tile instead.
   *
   * A tiled image keeps up to the whole tile cache loaded, most of it outside
   * the viewport, so only the tiles drawn in the viewport are invalidated. The
   * others are recolored once they are drawn again (see
   * {@link _recolorStaleTiles}), which keeps the cost of a change proportional
   * to the viewport rather than to the tile cache. Removing a data transfer
   * invalidates every tile of the tiled image instead, as tiles without a data
   * transfer are not caught up with.
   *
   * Data transfers are compared by identity: the tiles are only invalidated,
   * and thereby recolored from their original data, if a different data
   * transfer object is passed. Callers are expected to pass the same object for
   * as long as its outcome would not change, as invalidating the tiles re-runs
   * the data transfer on every tile in the viewport, and on every other loaded
   * tile of the tile source once it is drawn.
   *
   * @param tiledImage - The tiled image to update
   * @param dataTransfer - The data transfer to apply, or `undefined` for none
   */
  updateTiledImageDataTransfer(
    tiledImage: OpenSeadragon.TiledImage,
    dataTransfer: DataTransfer | undefined,
  ): void {
    const oldDataTransfer = this._tileSourceDataTransfers.get(
      tiledImage.source,
    );
    if (dataTransfer !== oldDataTransfer) {
      if (dataTransfer !== undefined) {
        this._tileSourceDataTransfers.set(tiledImage.source, dataTransfer);
      } else {
        this._tileSourceDataTransfers.delete(tiledImage.source);
      }
      const tiledImagesToInvalidate = [tiledImage];
      const navigator = this.viewer.navigator as OpenSeadragon.Navigator | null;
      if (navigator !== null) {
        for (let i = 0; i < navigator.world.getItemCount(); i++) {
          const navigatorTiledImage = navigator.world.getItemAt(i);
          if (navigatorTiledImage.source === tiledImage.source) {
            tiledImagesToInvalidate.push(navigatorTiledImage);
          }
        }
      }
      for (const tiledImageToInvalidate of tiledImagesToInvalidate) {
        tiledImageToInvalidate
          .requestInvalidate(
            /* restoreTiles */ true,
            /* viewportOnly */ tiledImageToInvalidate === tiledImage &&
              dataTransfer !== undefined,
          )
          .catch((error) => {
            console.error(`Failed to invalidate tiles: ${error}`);
          });
      }
    }
  }

  /**
   * Sets the bounds of the content that a source contributes to the world
   *
   * The world background (see the class documentation) is resized to the
   * bounding box of the bounds of all sources, or to
   * {@link _emptyWorldBounds} if there are none (see
   * {@link _resizeBackground}). Renderers register the bounds of their tiled
   * images under themselves; the bounds of content rendered outside
   * OpenSeadragon are registered under any other stable object. Passing no
   * bounds removes the source.
   *
   * @param source - The object under which to register the bounds
   * @param bounds - The bounds contributed by the source, in world coordinates
   * @param options - Optional abort signal
   * @returns A promise that resolves once the background covers the bounds,
   * i.e. once it has been drawn wherever they are in the viewport
   */
  setContentBounds(
    source: object,
    bounds: Rect[],
    options?: { signal?: AbortSignal },
  ): Promise<void> {
    if (bounds.length > 0) {
      this._contentBounds.set(source, bounds);
    } else {
      this._contentBounds.delete(source);
    }
    return this._resizeBackground(options);
  }

  /**
   * Hands viewport control back to the renderers
   *
   * Fits the viewport to the bounds of the world, and resumes doing so whenever
   * they change, until the user pans or zooms again. Call this when the content
   * of the viewer is replaced, e.g. on opening a project.
   */
  resetViewport(): void {
    this._viewportControlledByUser = false;
    this.viewer.viewport.goHome();
  }

  /**
   * Destroys the OpenSeadragon viewer and cleans up resources
   *
   * The context is marked as destroyed immediately (see {@link isDestroyed}), but
   * the viewer is only destroyed once no world mutation is pending anymore, as
   * their callbacks would otherwise run against a destroyed viewer. Marking the
   * context as destroyed also stops further world mutations from being enqueued,
   * so awaiting the queue's current tail is enough to drain it.
   */
  async destroy(): Promise<void> {
    this._destroyed = true;
    await this._worldMutationQueue;
    if (!this.viewer.isDestroyed()) {
      this.viewer.destroy();
    }
  }

  /**
   * Opens a tile source, see {@link openTileSource}
   *
   * @param tiledImageOptions - Options containing the tile source to open
   * @param options - Optional abort signal, which skips the open if the
   * operation is aborted before it starts
   * @returns A promise that resolves with the opened tile source
   */
  private async _openTileSource(
    tiledImageOptions: Omit<
      OpenSeadragon.TileSourceSpecifier,
      "success" | "error"
    >,
    options?: { signal?: AbortSignal },
  ): Promise<OpenSeadragon.TileSource> {
    const { signal } = options ?? {};
    // OpenSeadragon types tile sources as `string | object`, which also covers
    // promises of an already opened tile source; anything else is passed through
    const tileSource = await Promise.resolve(tiledImageOptions.tileSource);
    signal?.throwIfAborted();
    try {
      const { source: openedTileSource } =
        (await this.viewer.instantiateTileSourceClass(
          // this needs to be a shallow copy; OpenSeadragon mutates it!
          { ...tiledImageOptions, tileSource },
        )) as { source: OpenSeadragon.TileSource };
      return openedTileSource;
    } catch (error) {
      throw new Error("Failed to open tile source", { cause: error });
    }
  }

  /**
   * Resizes the world background to the registered bounds, creating it at world index 0 if it does not exist yet
   *
   * The background is kept if it already has the bounds (see
   * {@link OpenSeadragonUtils.hasBounds}). Otherwise, a new background is
   * inserted right below it, and the old one is removed only once the new one
   * has been drawn (see {@link _waitUntilDrawn}). Unlike a renderer's anchor
   * (see `OpenSeadragonRendererBase._resizeAnchor`), the background is not
   * replaced as part of the addition: OpenSeadragon removes the replaced tiled
   * image right away, whereas the new one is drawn only once its tile has
   * loaded, which would leave the canvas uncovered for a frame or more. While
   * both backgrounds exist, they sit below all anchors, so the world indices
   * that the renderers resolve relative to their anchors are unaffected.
   *
   * Resizes are run one at a time, in call order, each to the bounds
   * registered by the time it runs, so intermediate ones are skipped. A
   * failing resize does not prevent subsequent ones from running. Aborting
   * skips the resize if the new background has not been added yet, and has no
   * effect afterwards, as that would leave two backgrounds in the world. Does
   * nothing once the context has been destroyed.
   *
   * @param options - Optional abort signal
   * @returns A promise that resolves once the background has been resized
   */
  private _resizeBackground(options?: { signal?: AbortSignal }): Promise<void> {
    const { signal } = options ?? {};
    const promise = this._backgroundUpdateQueue.then(async () => {
      if (this._destroyed) {
        return;
      }
      signal?.throwIfAborted();
      const bounds =
        GeometryUtils.union(...[...this._contentBounds.values()].flat()) ??
        OpenSeadragonContext._emptyWorldBounds;
      const background = this._background;
      if (
        background !== undefined &&
        OpenSeadragonUtils.hasBounds(background, bounds)
      ) {
        return;
      }
      const newBackground = await this.addTiledImage(
        {
          index: background === undefined ? 0 : undefined,
          x: bounds.x,
          y: bounds.y,
          width: bounds.width,
          tileSource: OpenSeadragonUtils.createPixelTileSource(
            { width: bounds.width, height: bounds.height },
            this._backgroundPixelUrl,
          ),
        },
        {
          signal,
          getIndex:
            background !== undefined
              ? () => this.getTiledImageIndex(background)
              : undefined,
        },
      );
      this._background = newBackground;
      if (background !== undefined) {
        await this._waitUntilDrawn(newBackground);
        await this.removeTiledImage(background);
      }
    });
    this._backgroundUpdateQueue = promise.catch(() => {}); // prevent unhandled rejections in console
    return promise;
  }

  /**
   * Implementation of {@link addTiledImage}, bypassing the world mutation queue
   *
   * Must only be called from within an enqueued world mutation, such that the
   * index resolved by `options.getIndex` is still valid when OpenSeadragon
   * applies it.
   */
  private _addTiledImage(
    tiledImageOptions: Omit<
      OpenSeadragon.TileSourceSpecifier,
      "success" | "error"
    >,
    options?: {
      signal?: AbortSignal;
      getIndex?: () => number | undefined;
    },
  ): Promise<OpenSeadragon.TiledImage> {
    const { signal, getIndex } = options ?? {};
    return new Promise((resolve, reject) =>
      this.viewer.addTiledImage({
        ...tiledImageOptions,
        ...(getIndex !== undefined && { index: getIndex() }),
        success: (event) => {
          const { item: tiledImage } = event as unknown as {
            item: OpenSeadragon.TiledImage;
          };
          if (!signal?.aborted || tiledImageOptions.replace === true) {
            // a replacement cannot be undone, as the replaced tiled image is
            // already gone, so it is kept even if the operation was aborted
            resolve(tiledImage);
          } else {
            // the load could not be canceled, so undo it instead. This runs
            // while the enqueued world mutation is still in flight, so it must
            // not go through the queue, which would deadlock.
            this.viewer.world.removeItem(tiledImage);
            reject(signal.reason as DOMException);
          }
        },
        error: reject,
      }),
    );
  }

  /**
   * Waits until a tiled image has been drawn wherever it is in the viewport
   *
   * That is, until all tiles it needs for the current view have loaded, or
   * until it is not in the viewport at all: OpenSeadragon keeps the loading
   * state of a tiled image outside the viewport as it is, so the tiled image
   * would otherwise be waited for until it is panned into view. OpenSeadragon
   * sets the former while updating a frame, right before drawing it; the
   * latter can only change along with the viewport, as the tiled image is not
   * moved while waited for. Both are therefore checked right away, and again
   * whenever either changes. Also resolves once its tile has failed to load
   * for good, as it is never fully loaded then, and once the context has been
   * destroyed, at the latest when it destroys the viewer (see {@link destroy}).
   *
   * @param tiledImage - The tiled image to wait for
   * @returns A promise that resolves once the tiled image has been drawn
   */
  private _waitUntilDrawn(tiledImage: OpenSeadragon.TiledImage): Promise<void> {
    return new Promise((resolve) => {
      const done = () => {
        tiledImage.removeHandler("fully-loaded-change", check);
        this.viewer.removeHandler("viewport-change", check);
        this.viewer.removeHandler("before-destroy", check);
        this.viewer.removeHandler("tile-load-failed", checkFailure);
        resolve();
      };
      const check = () => {
        const drawArea = tiledImage.getDrawArea();
        if (this._destroyed || tiledImage.getFullyLoaded() || !drawArea) {
          done();
        }
      };
      const checkFailure = (event: OpenSeadragon.TileLoadFailedEvent) => {
        if (event.tiledImage === tiledImage && event.maxReached) {
          done();
        }
      };
      tiledImage.addHandler("fully-loaded-change", check);
      this.viewer.addHandler("viewport-change", check);
      this.viewer.addHandler("before-destroy", check);
      this.viewer.addHandler("tile-load-failed", checkFailure);
      check();
    });
  }

  /**
   * Appends a world mutation to the world mutation queue
   *
   * Tasks are run one at a time, in call order, and a failing task does not
   * prevent subsequent tasks from running. Every operation that adds to or
   * removes from the world has to be enqueued here, so that the world cannot
   * shift between an addition resolving its index and OpenSeadragon applying it.
   *
   * @param task - Task to run once all previously enqueued tasks have settled
   * @returns A promise that resolves with the task's result
   */
  private _enqueueWorldMutation<T>(task: () => T | Promise<T>): Promise<T> {
    const result = this._worldMutationQueue.then(task);
    this._worldMutationQueue = result.catch(() => {}); // prevent unhandled rejections in console
    return result;
  }

  /**
   * Recolors the drawn tiles whose data transfer is outdated
   *
   * A change of a data transfer only invalidates the tiles in the viewport
   * (see {@link updateTiledImageDataTransfer}), which leaves the other loaded
   * tiles of the tile source in the colors of an earlier data transfer. Every
   * tile records the data transfer it was last recolored with (see
   * {@link _applyDataTransfer}); a drawn tile that recorded a different one than
   * its tile source's is invalidated, unless it is being recolored already.
   * Such a tile is drawn in its earlier colors until it is recolored, which
   * takes a frame or two.
   *
   * Tiles loaded after a change are recolored on load, and tile sources
   * without a data transfer are skipped: their tiles were all restored when
   * the data transfer was removed.
   */
  private _recolorStaleTiles(): void {
    const staleTiles = [];
    for (let i = 0; i < this.viewer.world.getItemCount(); i++) {
      const tiledImage = this.viewer.world.getItemAt(i);
      const dataTransfer = this._tileSourceDataTransfers.get(tiledImage.source);
      if (dataTransfer === undefined) {
        continue;
      }
      for (const { tile } of tiledImage.getTilesToDraw()) {
        if (
          tile.processing === false &&
          this._tileDataTransfers.get(tile) !== dataTransfer
        ) {
          staleTiles.push(tile);
        }
      }
    }
    if (staleTiles.length > 0) {
      this.viewer.world
        .requestTileInvalidateEvent(
          staleTiles,
          OpenSeadragon.now(),
          /* restoreTiles */ true,
        )
        .catch((error) => {
          console.error(`Failed to invalidate tiles: ${error}`);
        });
    }
  }

  /**
   * Replaces the data of an invalidated tile with the colors of its values
   *
   * Does nothing unless a data transfer is set for the tile source of the tile
   * (see {@link updateTiledImageDataTransfer}).
   *
   * OpenSeadragon raises the invalidation events of all tiles of a change at
   * once, and recoloring a tile holds the thread for as long as the tile has
   * pixels, so each tile first yields to the event loop: the user interface
   * stays responsive while a change is recolored. A change that arrives
   * meanwhile invalidates the same tiles again, which marks the runs of the
   * change before it as outdated; those are abandoned, as the newer runs
   * recolor the tiles. The data transfer is read only after yielding, so that
   * a run that continues always applies the latest one. It is recorded as the
   * tile's before the tile is recolored (see {@link _recolorStaleTiles}), so
   * that a tile whose recoloring fails is not retried on every draw.
   *
   * The colors are written as packed 32-bit values through a `Uint32Array`
   * view of an `ImageData` buffer, whose bytes are R, G, B, A. The
   * `0xAABBGGRR` layout of {@link DataTransfer} lands in that order on a
   * little-endian host; on a big-endian host, the bytes of every pixel are
   * swapped afterwards. The buffer is shared by all tiles of the viewer and
   * only ever grows, as this is a hot path: changing a data transfer
   * reconverts every cached tile. Sharing is safe because nothing awaits
   * between filling and copying the buffer, and `putImageData` copies the
   * pixels.
   *
   * The pixels are copied onto a new canvas of the raster's size, which
   * becomes the tile's data. The canvas cannot be shared, as OpenSeadragon
   * keeps it in its tile cache by reference. Nor is the tile's own canvas
   * used: obtaining that would convert the tile's original data to a canvas
   * first, only for it to be overwritten. The canvas is created with
   * `willReadFrequently`, which keeps it in memory rather than on the GPU: the
   * WebGL drawer reads a pixel of every canvas it turns into a texture, to
   * check whether it is tainted, and reading from a GPU canvas waits for the
   * GPU.
   *
   * @param event - The tile invalidation event whose tile data is replaced
   * @returns A promise that resolves once the tile data has been replaced
   * @throws Error if the number of pixel values does not match the raster size
   */
  private async _applyDataTransfer(
    event: OpenSeadragon.TileInvalidatedEvent,
  ): Promise<void> {
    const tiledImage = event.tile.tiledImage;
    if (tiledImage === null) {
      return;
    }
    if (!this._tileSourceDataTransfers.has(tiledImage.source)) {
      this._tileDataTransfers.delete(event.tile);
      return;
    }
    await AsyncUtils.yield();
    if (await event.outdated()) {
      return;
    }
    const dataTransfer = this._tileSourceDataTransfers.get(tiledImage.source);
    if (dataTransfer === undefined) {
      this._tileDataTransfers.delete(event.tile);
      return;
    }
    this._tileDataTransfers.set(event.tile, dataTransfer);
    const { values, width, height } = await dataTransfer.getTileData(event);
    if (await event.outdated()) {
      return; // skip the recoloring, as a newer run replaces the data anyway
    }
    if (values.length !== width * height) {
      throw new Error("Invalid tile data size");
    }
    const byteLength = 4 * width * height;
    if (this._pixelBuffer.byteLength < byteLength) {
      this._pixelBuffer = new ArrayBuffer(byteLength);
    }
    const pixelBuffer = new Uint32Array(this._pixelBuffer, 0, width * height);
    dataTransfer.transferValues(values, pixelBuffer);
    if (!OpenSeadragonContext._isLittleEndian) {
      const view = new DataView(this._pixelBuffer);
      for (let i = 0; i < pixelBuffer.length; i++) {
        view.setUint32(4 * i, pixelBuffer[i]!, /* littleEndian */ true);
      }
    }
    const bytes = new Uint8ClampedArray(this._pixelBuffer, 0, byteLength);
    const imageData = new ImageData(bytes, width, height);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.putImageData(imageData, 0, 0);
    await event.setData(ctx, "context2d");
  }
}
