import { deepEqual } from "fast-equals";
import type OpenSeadragon from "openseadragon";

import {
  AsyncUtils,
  type CustomTileSource,
  GeometryUtils,
  type Image,
  type ImageData,
  type Labels,
  type LabelsData,
  type Layer,
  type Rect,
  type TileSourceConfig,
} from "@tissuumaps/core";

import type {
  DataTransfer,
  OpenSeadragonContext,
} from "./OpenSeadragonContext";
import { OpenSeadragonUtils } from "./OpenSeadragonUtils";

/**
 * Base class for OpenSeadragon renderers that manage tiled images for objects (images or labels)
 *
 * Renderers share a viewer, so each renderer owns an anchor: an invisible
 * single-tile image that marks where the renderer's own tiled images belong.
 * They directly follow the anchor, in the order of {@link _renderedObjects},
 * with one tiled image per channel of each object.
 *
 * Every tiled image contributes to the bounds of the world, the anchor
 * included, so the anchor spans just the renderer's own tiled images, which
 * keeps it from extending the world (see {@link updateBounds}). The same bounds
 * are registered with the context (see
 * {@link OpenSeadragonContext.setContentBounds}), whose world background fits
 * the world to the content of all renderers, and covers it from below.
 *
 * The tiled images of an object that uses additive blending are preceded by one
 * more tiled image, its backdrop (see {@link usesAdditiveBlending}).
 *
 * Tiled images are inserted behind the anchor when they are added, rather than
 * moved there afterwards, as OpenSeadragon's navigator cannot keep up with
 * reordering. {@link _cleanRenderedObjects} recreates those that are out of place.
 *
 * The layers and objects to render are set by {@link setModel}, which applies
 * the properties that tiled images take directly - transforms, visibility and
 * opacity - to the rendered objects right away. Everything else - the set and
 * order of the layers and objects, data sources and whatever the subclasses
 * resolve from an object (see {@link resolveObject}) - requires a
 * synchronization, which {@link needsSynchronization} reports and
 * {@link synchronize} performs. The properties applied directly are always read
 * from the current model, never from the reference of a rendered object (see
 * {@link _updateRenderedObject}), so a synchronization that is in flight while
 * the model changes does not write stale values.
 */
export abstract class OpenSeadragonRendererBase<
  TObject extends Image | Labels,
  TObjectData extends ImageData | LabelsData,
  TSyncContext extends {
    loadObject: (
      object: TObject,
      options?: { signal?: AbortSignal },
    ) => Promise<TObjectData>;
  },
> {
  private static readonly _emptyAnchorBounds: Rect = {
    x: 0,
    y: 0,
    width: 1,
    height: 1,
  };

  readonly context: OpenSeadragonContext;
  private _anchor: OpenSeadragon.TiledImage | undefined;
  private _model?: { layers: Layer[]; objects: TObject[] };
  private _lastSyncState?: object;
  private _renderedObjects: RenderedObject<TObject, TObjectData>[] = [];
  private _anchorTaskPromise: Promise<unknown> = Promise.resolve();
  private _destroyed: boolean = false;

  /**
   * Creates a new OpenSeadragonRendererBase instance and asynchronously adds its anchor
   *
   * The renderer must not be used before `onInitialized` has been called;
   * `onError` is called instead if adding the anchor failed or was aborted.
   *
   * @param context - The OpenSeadragon context that provides access to the viewer and other shared state
   * @param onInitialized - Called once the anchor has been added to the world
   * @param onError - Called if the anchor could not be added
   * @param options - Optional abort signal and world index at which to insert
   * the anchor, behind the context's world background (i.e. `1` or higher)
   */
  constructor(
    context: OpenSeadragonContext,
    onInitialized: () => void,
    onError: (error: Error) => void,
    options?: { signal?: AbortSignal; anchorIndex?: number },
  ) {
    this.context = context;
    this._enqueueAnchorTask(async () => {
      const { signal, anchorIndex } = options ?? {};
      signal?.throwIfAborted();
      await this._resizeAnchor(OpenSeadragonRendererBase._emptyAnchorBounds, {
        signal,
        anchorIndex,
      });
    }).then(onInitialized, onError);
  }

  /**
   * Sets the layers and objects to render
   *
   * The properties that tiled images take directly - the layer and object
   * transforms, visibility and opacity, and whatever else a subclass reads from
   * the current model when updating a tiled image (see
   * {@link getTiledImageOpacity} and {@link resolveTiledImageDataTransfer}) -
   * are applied to the rendered objects right away (see
   * {@link _updateRenderedObject}), and need no synchronization.
   * Every other change - a different set or order of layers or objects, layer
   * memberships, data sources or the configurations that subclasses resolve
   * from - requires a resynchronization, which the caller is expected to
   * trigger whenever {@link needsSynchronization} says so.
   *
   * Does not resize the anchor: a transform change moves the tiled images, so
   * the caller is expected to call {@link updateBounds} afterwards.
   *
   * The layers and objects are cloned, so that what the renderer compares
   * against later - in {@link needsSynchronization}, and in the references of
   * a synchronization - is what it was given, whatever the caller does with its
   * instances afterwards.
   *
   * @param layers - The layers to render
   * @param objects - The objects (images or labels) to render
   */
  setModel(layers: Layer[], objects: TObject[]): void {
    this._model = structuredClone({ layers, objects });
    for (const renderedObject of this._renderedObjects) {
      if (renderedObject.tiledImages !== undefined) {
        this._updateRenderedObject(renderedObject);
      }
    }
  }

  /**
   * Returns whether the rendered objects have to be synchronized with the
   * current model
   *
   * Compares the state that a synchronization depends on (see
   * {@link getSyncState}) against the one the last synchronization was based
   * on, which {@link synchronize} records before its first `await` and forgets
   * again if it fails. Everything else about the model - the properties that
   * tiled images take directly - is fully applied by {@link setModel}.
   */
  needsSynchronization(): boolean {
    return !deepEqual(this.getSyncState(), this._lastSyncState);
  }

  /**
   * Synchronizes the viewer's tiled images with the current model
   *
   * Starts loading the objects on the layers of the model (see
   * {@link setModel}), and right away removes the tiled images that are no
   * longer needed (see {@link _cleanRenderedObjects}). Each object is then
   * synchronized as soon as it has loaded, without waiting for the others: its
   * tiled images are updated, or created at its position in the world (see
   * {@link _synchronizeRenderedObject}).
   *
   * Resolves once the tiled images of all objects are in the world. The model
   * is read once, so the synchronization sees one consistent state, but the
   * properties applied to the tiled images come from the current model (see
   * {@link _updateRenderedObject}).
   *
   * Objects whose data fails to load, or whose tiled images cannot be created
   * (e.g. without tile sources), are logged and dropped. A synchronization that
   * fails or is aborted leaves the model unsynchronized (see
   * {@link needsSynchronization}).
   *
   * @param context - The inputs to synchronize with: an immutable snapshot of
   * the tables and maps that the objects resolve from and the loaders that the
   * renderer needs. It carries inputs only; a renderer that derives state from
   * an object does so in {@link resolveObject}.
   * @param options - Optional abort signal
   * @throws Error if no model has been set (see {@link setModel})
   */
  async synchronize(
    context: TSyncContext,
    options?: { signal?: AbortSignal },
  ): Promise<void> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const syncState = this.getSyncState();
    this._lastSyncState = syncState;
    try {
      const loadings = this._loadObjects(context, { signal });
      const renderedObjectsByLoading = await this._cleanRenderedObjects(
        loadings,
        { signal },
      );
      // drop the state of the other objects only once their tiled images are
      // gone, as setModel would otherwise update them without it
      this.retainObjects(loadings.map((loading) => loading.object));
      const syncResults = await Promise.allSettled(
        loadings.map((loading, index) =>
          this._synchronizeRenderedObject(
            loadings.slice(0, index),
            loading.newRefPromise,
            renderedObjectsByLoading.get(loading),
            { signal },
          ),
        ),
      );
      signal?.throwIfAborted(); // Promise.allSettled() does not throw on abort
      const firstSyncFailure = syncResults.find(
        (result) => result.status === "rejected",
      );
      if (firstSyncFailure !== undefined) {
        throw firstSyncFailure.reason;
      }
      await Promise.allSettled(
        this._renderedObjects.map(
          (renderedObject) => renderedObject.tiledImagesPromise,
        ),
      );
      signal?.throwIfAborted(); // Promise.allSettled() does not throw on abort
      await this._updateBoundsAndReveal(this._renderedObjects, { signal });
    } catch (error) {
      if (this._lastSyncState === syncState) {
        this._lastSyncState = undefined;
      }
      throw error;
    }
  }

  /**
   * Resizes the anchor to the bounding box of all tiled images, and registers it with the context
   *
   * Rendered objects whose tiled images have not been added to the world yet are
   * ignored; they update the anchor themselves upon arrival (see
   * {@link _createRenderedObject}). Resolves once the context's world
   * background covers the bounds, too (see
   * {@link OpenSeadragonContext.setContentBounds}). The bounds are registered
   * after the anchor task rather than within it, so that the anchor task
   * queue, and with it {@link destroy}, never waits for the background, which
   * waits for being drawn. Does nothing once the renderer has been destroyed,
   * as there is no anchor to resize anymore.
   *
   * @param options - Optional abort signal
   */
  async updateBounds(options?: { signal?: AbortSignal }): Promise<void> {
    if (this._destroyed) {
      return;
    }
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const bounds = await this._enqueueAnchorTask(async () => {
      signal?.throwIfAborted();
      const tiledImageBounds = [];
      for (const renderedObject of this._renderedObjects) {
        if (renderedObject.backdrop !== undefined) {
          tiledImageBounds.push(renderedObject.backdrop.getBounds());
        }
        if (renderedObject.tiledImages !== undefined) {
          for (const tiledImage of renderedObject.tiledImages) {
            tiledImageBounds.push(tiledImage.getBounds());
          }
        }
      }
      const bounds = GeometryUtils.union(...tiledImageBounds);
      await this._resizeAnchor(
        bounds ?? OpenSeadragonRendererBase._emptyAnchorBounds,
        { signal },
      );
      return bounds;
    });
    if (!this._destroyed) {
      await this.context.setContentBounds(
        this,
        bounds !== null ? [bounds] : [],
        { signal },
      );
    }
  }

  /**
   * Destroys the renderer by removing the anchor tiled image and all rendered objects from the OpenSeadragon viewer, and its bounds from the context
   *
   * Rendered objects whose tiled images have not been added to the world yet are
   * only marked for deletion, and are removed as soon as they arrive.
   *
   * The renderer is unusable afterwards: it has no anchor anymore, so
   * {@link updateBounds} does nothing, {@link _cleanRenderedObjects} throws, and
   * tiled images that still arrive are removed right away.
   *
   * The bounds are removed from the context without waiting for the world
   * background to be resized, and a failure to resize it is only logged: the
   * background waits for being drawn, which never happens while the page is
   * hidden, and the teardown of the renderer, and thereby of the context, must
   * not depend on it.
   */
  async destroy(): Promise<void> {
    this._destroyed = true;
    for (const renderedObject of this._renderedObjects) {
      await this._deleteRenderedObject(renderedObject);
    }
    this._renderedObjects = [];
    // remove the anchor once all pending anchor tasks have settled, as they
    // would otherwise re-create it
    await this._enqueueAnchorTask(async () => {
      if (this._anchor !== undefined) {
        const anchor = this._anchor;
        this._anchor = undefined;
        await this.context.removeTiledImage(anchor);
      }
    });
    this.context.setContentBounds(this, []).catch((error) => {
      console.error("Failed to remove the renderer's content bounds", error);
    });
  }

  /**
   * Returns the state of the current model that a synchronization depends on
   *
   * Unlike the WebGL renderers, whose draw order is read from the model on
   * every draw, the order of the layers and objects is part of the state: it
   * is the order of the tiled images in the world, which only a
   * synchronization can change (see {@link _cleanRenderedObjects}). See
   * {@link getLayerSyncState} and {@link getObjectSyncState} for what else is.
   *
   * @returns The state, or `undefined` if no model has been set
   */
  protected getSyncState(): object | undefined {
    if (this._model === undefined) {
      return undefined;
    }
    return {
      layers: this._model.layers.map((layer) => this.getLayerSyncState(layer)),
      objects: this._model.objects.map((object) =>
        this.getObjectSyncState(object),
      ),
    };
  }

  /**
   * Returns the state of a layer that a synchronization depends on
   *
   * The counterpart of {@link getObjectSyncState} for layers. The point size
   * factor is blanked out as well, as nothing rendered here depends on it, and
   * so is the name.
   *
   * @param layer - The layer to return the state of
   * @returns The layer without the properties that are applied by
   * {@link setModel}, and without the cosmetic ones
   */
  protected getLayerSyncState(layer: Layer): object {
    return {
      ...layer,
      name: undefined,
      transform: undefined,
      visibility: undefined,
      opacity: undefined,
      pointSizeFactor: undefined,
    };
  }

  /**
   * Returns the state of an object that a synchronization depends on
   *
   * Everything but the properties that are applied by {@link setModel}, and
   * but the cosmetic ones, which nothing rendered depends on. Those are blanked
   * out rather than dropped, so that a property added to the model later is
   * part of the state, and thereby requires a resynchronization, unless it is
   * blanked out here as well. Subclasses override this to blank out the
   * properties that they apply from the current model themselves.
   *
   * The result is only ever deep-compared against that of another object, hence
   * the opaque return type.
   *
   * @param object - The object (image or labels) to return the state of
   * @returns The object without the properties that are applied by
   * {@link setModel}, and without the cosmetic ones
   */
  protected getObjectSyncState(object: TObject): object {
    return {
      ...object,
      name: undefined,
      transform: undefined,
      visibility: undefined,
      opacity: undefined,
    };
  }

  /**
   * Retains what was resolved for the given objects, and discards the rest
   *
   * Called by {@link synchronize} with the objects about to be displayed, i.e.
   * those assigned to a rendered layer, once the rendered objects of all others
   * have been deleted and before any object has loaded. Does nothing here;
   * subclasses that keep state per object (see {@link resolveObject} and
   * {@link resolveTiledImageDataTransfer}) override this to drop the state of
   * all other objects.
   *
   * @param _objects - The objects (images or labels) about to be displayed
   */
  protected retainObjects(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _objects: TObject[],
  ): void {}

  /**
   * Resolves what a renderer derives from an object, once its data has loaded
   *
   * Called by {@link _loadObjects} for each object, concurrently, once its data
   * has loaded and before its tiled images are created or updated. An object
   * that cannot be resolved is logged, but kept: its tiled images are still
   * created or updated, with whatever the synchronous hooks return for it.
   *
   * Does nothing here. Subclasses override this to resolve what their
   * synchronous hooks return later (see {@link resolveTiledImageDataTransfer})
   * from the object, its data and the inputs of the synchronization, and to
   * keep it for as long as its outcome would not change.
   *
   * Once the signal is aborted, it has to reject rather than resolve, as
   * {@link synchronize} creates or updates the object's tiled images once it
   * resolves.
   *
   * @param _object - The object (image or labels) to resolve
   * @param _data - The loaded data of the object
   * @param _context - The inputs of the current synchronization
   * @param _options - Optional abort signal
   * @returns A promise that resolves once the object has been resolved
   */
  protected resolveObject(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _object: TObject,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _data: TObjectData,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _context: TSyncContext,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _options?: { signal?: AbortSignal },
  ): Promise<void> {
    return Promise.resolve();
  }

  /**
   * Returns the tile sources for the given object data
   *
   * @param data - The object data (image or labels) for which to retrieve the tile sources
   * @returns The tile sources, one per tiled image of the object. Defaults to
   * the single tile source of the data; renderers of multi-channel data
   * override this.
   */
  protected getTileSources(
    data: TObjectData,
  ): (string | TileSourceConfig | CustomTileSource)[] {
    return [data.getTileSource()];
  }

  /**
   * Returns whether the channels of the given object data are blended additively
   *
   * The channels of an object that blends additively are composited with
   * OpenSeadragon's "lighter" operation onto an opaque black backdrop below
   * them, so that they add up among themselves while the backdrop hides
   * whatever is below the object, thereby compositing the object as a whole over
   * it. Objects that do not blend additively have no backdrop and keep
   * OpenSeadragon's default composite operation, i.e. each of their channels is
   * composited over the one below it.
   *
   * This only composites correctly onto an opaque canvas: "lighter" adds alpha
   * as well as color, so onto the transparent canvas that OpenSeadragon clears
   * to, an object with opacity below one would accumulate alpha from each
   * channel and fade out to the wrong color, in wrong proportions, instead of
   * fading out to the background behind the viewer. The context's world
   * background (see {@link OpenSeadragonContext.setContentBounds}) therefore makes the
   * canvas opaque wherever objects are.
   *
   * @param _data - The object data (image or labels) to check
   * @returns Whether the object's channels are blended additively. Defaults to `false`.
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  protected usesAdditiveBlending(_data: TObjectData): boolean {
    return false;
  }

  /**
   * Computes the effective opacity for one of an object's tiled images
   *
   * Returns `0` when either the layer or the object is invisible; otherwise
   * multiplies layer and object opacities. The channel index is ignored here,
   * i.e. all tiled images of an object share the same opacity; subclasses
   * override this to additionally apply per-channel visibility and opacity. The
   * channel index is omitted for an object's backdrop, which carries the opacity
   * of the object itself.
   *
   * @param ref - The object reference for which to compute the opacity
   * @param _index - The index of the tiled image (e.g. channel), or `null` for the object's backdrop
   * @returns The effective opacity for the tiled image
   */
  protected getTiledImageOpacity(
    ref: ObjectRef<TObject, TObjectData>,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _index: number | null,
  ): number {
    const visibility = ref.layer.visibility && ref.object.visibility;
    const opacity = ref.layer.opacity * ref.object.opacity;
    return visibility ? opacity : 0;
  }

  /**
   * Resolves the data transfer for one of an object's tiled images
   *
   * Returns `undefined` here, i.e. the tiles are drawn as they are; subclasses
   * whose tiles carry values rather than colors override this to map the values
   * to colors (see {@link OpenSeadragonContext.updateTiledImageDataTransfer}).
   * As data transfers are compared by identity, the returned object has to stay
   * the same for as long as its outcome would not change. This hook is
   * synchronous and receives the object as it currently is in the model:
   * subclasses either resolve the data transfer here, from the current object
   * and its data, which needs no synchronization, or, if resolving is
   * asynchronous, once the object's data has loaded (see
   * {@link resolveObject}), and only return it here.
   *
   * @param _ref - The object reference for which to get the data transfer
   * @param _index - The index of the tiled image (e.g. channel), or `null` for the object's backdrop
   * @returns The data transfer to apply, or `undefined` for none. Defaults to
   * `undefined`.
   */
  protected resolveTiledImageDataTransfer(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _ref: ObjectRef<TObject, TObjectData>,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _index: number | null,
  ): DataTransfer | undefined {
    return undefined;
  }

  /**
   * Starts loading and resolving all objects assigned to the layers of the current model
   *
   * Loads each object's data with the context's `loadObject`, then resolves the
   * object (see {@link resolveObject}). Returns right away, with one loading
   * per object, in world order: by layer, then by object. A loading rejects if
   * the object's data fails to load, which is logged. An object that cannot be
   * resolved is logged, but kept.
   *
   * The model is read once, right away, so one synchronization sees one
   * consistent model state.
   *
   * @param context - The inputs of the current synchronization
   * @param options - Optional abort signal
   * @returns One loading per object, whose promise resolves to the object's
   * reference
   * @throws Error if no model has been set (see {@link setModel})
   */
  private _loadObjects(
    context: TSyncContext,
    options?: { signal?: AbortSignal },
  ): {
    layer: Layer;
    object: TObject;
    newRefPromise: Promise<ObjectRef<TObject, TObjectData>>;
  }[] {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const model = this._model;
    if (model === undefined) {
      throw new Error("Model not set");
    }
    const loadings: {
      layer: Layer;
      object: TObject;
      newRefPromise: Promise<ObjectRef<TObject, TObjectData>>;
    }[] = [];
    for (const currentLayer of model.layers) {
      for (const currentObject of model.objects.filter(
        (object) => object.layer === currentLayer.id,
      )) {
        const newRefPromise = context
          .loadObject(currentObject, { signal })
          .then(async (data) => {
            try {
              await this.resolveObject(currentObject, data, context, {
                signal,
              });
            } catch (error) {
              if (signal?.aborted) {
                throw error;
              }
              console.error(
                `Failed to resolve object with ID '${currentObject.id}'`,
                error,
              );
            }
            return { layer: currentLayer, object: currentObject, data };
          });
        newRefPromise.catch((error) => {
          if (!signal?.aborted) {
            console.error(
              `Failed to load object with ID '${currentObject.id}'`,
              error,
            );
          }
        });
        loadings.push({
          layer: currentLayer,
          object: currentObject,
          newRefPromise,
        });
      }
    }
    return loadings;
  }

  /**
   * Keeps the rendered objects that can be reused for a synchronization, and deletes the rest
   *
   * A rendered object is matched by the layer and object it references.
   * Unmatched ones are deleted first, so that their removal does not shift the
   * world indices of the objects behind them. Only the layers and objects of
   * the loadings are needed, not their data, so this runs before any of them
   * has loaded.
   *
   * A matched rendered object is reusable if it was loaded for its object's
   * data source, and if its backdrop, if any, and all of its tiled images sit
   * at the consecutive world indices expected for its position among the
   * loadings, counted from the anchor. All other matched rendered objects are
   * deleted as well, for the caller to recreate them (see
   * {@link _createRenderedObject}), which is also how the world is reordered.
   *
   * The expected indices count every matched rendered object, reusable or not,
   * by its current footprint. A matched object that is recreated, e.g. because
   * its data source changed, is reinserted at the same position, so the objects
   * behind it stay reusable. An object whose tiled images could not be added
   * has no footprint, and is thereby retried without recreating the objects
   * behind it.
   *
   * The tiled images that an earlier synchronization is still adding are waited
   * for first, so that each rendered object is either fully in the world or not
   * at all. This does not delay the additions of this synchronization, which
   * are queued behind them anyway (see
   * {@link OpenSeadragonContext.addTiledImage}).
   *
   * @param loadings - The loadings of the synchronization (see
   * {@link _loadObjects}), in world order
   * @param options - Optional abort signal
   * @returns A map of loadings to their reusable rendered objects
   * @throws Error if the renderer has no anchor, i.e. it is not initialized or
   * already destroyed
   */
  private async _cleanRenderedObjects<
    TLoading extends { layer: Layer; object: TObject },
  >(
    loadings: TLoading[],
    options?: { signal?: AbortSignal },
  ): Promise<Map<TLoading, RenderedObject<TObject, TObjectData>>> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    await Promise.allSettled(
      this._renderedObjects.map(
        (renderedObject) => renderedObject.tiledImagesPromise,
      ),
    );
    signal?.throwIfAborted(); // Promise.allSettled() does not throw on abort
    const matchedRenderedObjects = new Set<
      RenderedObject<TObject, TObjectData>
    >();
    const matchedRenderedObjectsByLoading = new Map<
      TLoading,
      RenderedObject<TObject, TObjectData>
    >();
    for (const loading of loadings) {
      const renderedObject = this._renderedObjects.find(
        (renderedObject) =>
          renderedObject.ref.layer.id === loading.layer.id &&
          renderedObject.ref.object.id === loading.object.id,
      );
      if (renderedObject !== undefined) {
        matchedRenderedObjects.add(renderedObject);
        matchedRenderedObjectsByLoading.set(loading, renderedObject);
      }
    }
    // deletions are queued right away, so the deleted objects are forgotten
    // right away, too, even if the synchronization is aborted while they run
    const unmatchedDeletions = this._renderedObjects
      .filter((renderedObject) => !matchedRenderedObjects.has(renderedObject))
      .map((renderedObject) => this._deleteRenderedObject(renderedObject));
    this._renderedObjects = [...matchedRenderedObjects];
    await Promise.all(unmatchedDeletions);
    signal?.throwIfAborted();
    if (this._anchor === undefined) {
      throw new Error("Anchor not initialized");
    }
    const anchorIndex = this.context.getTiledImageIndex(this._anchor);
    if (anchorIndex === -1) {
      throw new Error("Anchor not found");
    }
    const renderedObjectsByLoading = new Map<
      TLoading,
      RenderedObject<TObject, TObjectData>
    >();
    const survivors = new Set<RenderedObject<TObject, TObjectData>>();
    let offset = 1;
    for (const loading of loadings) {
      const renderedObject = matchedRenderedObjectsByLoading.get(loading);
      if (renderedObject !== undefined) {
        if (
          // data source configuration unchanged (checked instead of data)
          deepEqual(
            renderedObject.ref.object.dataSource,
            loading.object.dataSource,
          ) &&
          // tiled images exist, i.e. so does the backdrop if the object uses one
          renderedObject.tiledImages !== undefined &&
          // backdrop, if any, is at the expected index
          (renderedObject.backdrop === undefined ||
            this.context.getTiledImageIndex(renderedObject.backdrop) ===
              anchorIndex + offset) &&
          // tiled images are at the expected indices
          renderedObject.tiledImages.every(
            (tiledImage, c) =>
              this.context.getTiledImageIndex(tiledImage) ===
              anchorIndex +
                offset +
                (renderedObject.backdrop !== undefined ? 1 : 0) +
                c,
          )
        ) {
          renderedObjectsByLoading.set(loading, renderedObject);
          survivors.add(renderedObject);
        }
        offset +=
          (renderedObject.backdrop !== undefined ? 1 : 0) +
          (renderedObject.tiledImages?.length ?? 0);
      }
    }
    const unreusableDeletions = [...matchedRenderedObjects]
      .filter((renderedObject) => !survivors.has(renderedObject))
      .map((renderedObject) => this._deleteRenderedObject(renderedObject));
    this._renderedObjects = [...survivors];
    await Promise.all(unreusableDeletions);
    signal?.throwIfAborted();
    return renderedObjectsByLoading;
  }

  /**
   * Synchronizes the rendered object of one object, once it has loaded
   *
   * A reusable rendered object (see {@link _cleanRenderedObjects}) is updated
   * with the new reference, or deleted if the object failed to load. Otherwise,
   * a rendered object is created behind those of the preceding loadings,
   * whether these are in the world already or still being added. Preceding
   * objects that arrive later are inserted in front of it, as world mutations
   * run in request order (see {@link OpenSeadragonContext.addTiledImage}).
   *
   * @param precedingLoadings - The loadings that precede the object's in world
   * order (see {@link _loadObjects})
   * @param newRefPromise - The promise of the object's reference, from its
   * loading
   * @param renderedObject - The object's reusable rendered object, if any
   * @param options - Optional abort signal
   * @returns A promise that resolves once the object's tiled images have been
   * updated, or their creation has been requested
   */
  private async _synchronizeRenderedObject(
    precedingLoadings: { layer: Layer; object: TObject }[],
    newRefPromise: Promise<ObjectRef<TObject, TObjectData>>,
    renderedObject: RenderedObject<TObject, TObjectData> | undefined,
    options?: { signal?: AbortSignal },
  ): Promise<void> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    let newRef;
    try {
      newRef = await newRefPromise;
    } catch {
      signal?.throwIfAborted(); // an aborted loading rejects, too
      if (renderedObject !== undefined) {
        this._renderedObjects = this._renderedObjects.filter(
          (otherRenderedObject) => otherRenderedObject !== renderedObject,
        );
        await this._deleteRenderedObject(renderedObject);
      }
      return;
    }
    if (renderedObject !== undefined) {
      this._updateRenderedObject(renderedObject, newRef);
      return;
    }
    const precedingRenderedObjects = this._renderedObjects.filter(
      (otherRenderedObject) =>
        precedingLoadings.some(
          (precedingLoading) =>
            precedingLoading.layer.id === otherRenderedObject.ref.layer.id &&
            precedingLoading.object.id === otherRenderedObject.ref.object.id,
        ),
    );
    let offset = 0;
    for (const otherRenderedObject of precedingRenderedObjects) {
      if (otherRenderedObject.tiledImages !== undefined) {
        offset +=
          (otherRenderedObject.backdrop !== undefined ? 1 : 0) +
          otherRenderedObject.tiledImages.length;
      } else {
        offset +=
          (otherRenderedObject.usesBackdrop ? 1 : 0) +
          otherRenderedObject.tileSourceCount;
      }
    }
    let newRenderedObject;
    try {
      newRenderedObject = this._createRenderedObject(offset, newRef, {
        signal,
      });
    } catch (error) {
      console.error(
        `Failed to create tiled images for object with ID '${newRef.object.id}'`,
        error,
      );
      return;
    }
    this._renderedObjects = [
      ...precedingRenderedObjects,
      newRenderedObject,
      ...this._renderedObjects.slice(precedingRenderedObjects.length),
    ];
  }

  /**
   * Creates a new rendered object for the given object reference and adds one TiledImage per channel, preceded by a backdrop where required, to the world
   *
   * The TiledImages are inserted at consecutive indices, starting `offset`
   * places after the anchor, which is the world layout that
   * {@link _cleanRenderedObjects} expects. `offset` thus counts TiledImages,
   * not objects: the backdrops and TiledImages of the preceding rendered
   * objects, including those still being added. The indices are resolved when
   * each addition is executed, as the anchor may have been replaced, and world
   * indices may have shifted, in the meantime.
   *
   * The tile sources are opened here, rather than by the additions, so that the
   * backdrop can be sized like the object's content, which is only known once
   * one of them has opened. All additions are requested before this function
   * returns, in world order, so that the TiledImages of the object created next
   * end up behind them.
   *
   * Returns before the TiledImages exist: they are added asynchronously, and
   * only then assigned to the rendered object, transformed and included in the
   * anchor bounds, which is when {@link RenderedObject.tiledImagesPromise}
   * resolves.
   *
   * If the rendered object is deleted or the operation is aborted in the
   * meantime, or if any TiledImage could not be added, the added backdrop and
   * TiledImages are removed again right away: a partially added object would
   * shift the world indices of every object behind it. For the same reason, a
   * tile source that fails to open is replaced by a transparent placeholder
   * rather than skipped, as the objects created after this one are placed
   * against its full footprint. The rendered object is dropped from
   * {@link _renderedObjects} as its removals are queued, so that objects
   * created from then on are placed without it. If the context is destroyed,
   * nothing is done at all, as the viewer tears down its world itself.
   *
   * @param offset - The number of TiledImages between the anchor and the object's first new TiledImage
   * @param newRef - The object reference for which to create a rendered object
   * @param options - Optional abort signal
   * @returns The newly created rendered object, which does not have TiledImages yet
   * @throws Error if the object data provides no tile sources
   */
  private _createRenderedObject(
    offset: number,
    newRef: ObjectRef<TObject, TObjectData>,
    options?: { signal?: AbortSignal },
  ): RenderedObject<TObject, TObjectData> {
    const { signal } = options ?? {};
    const tileSources = this.getTileSources(newRef.data);
    if (tileSources.length === 0) {
      throw new Error(
        `Object with ID '${newRef.object.id}' has no tile sources`,
      );
    }
    const useBackdrop = this.usesAdditiveBlending(newRef.data);
    const tileSourcePromises = tileSources.map((tileSource) =>
      this.context.openTileSource({ tileSource }, { signal }),
    );
    const getPlaceholderTileSource = (error: unknown) => {
      if (signal?.aborted) {
        throw error; // skips the additions of the objects behind, too
      }
      return OpenSeadragonUtils.createPixelTileSource(
        { width: 1, height: 1 },
        OpenSeadragonUtils.transparentBlackPixelUrl,
      );
    };
    const backdropTileSourcePromise = useBackdrop
      ? tileSourcePromises[0]!.then(
          (firstTileSource) =>
            OpenSeadragonUtils.createPixelTileSource(
              {
                width: firstTileSource.dimensions.x,
                height: firstTileSource.dimensions.y,
              },
              OpenSeadragonUtils.opaqueBlackPixelUrl,
            ),
          getPlaceholderTileSource,
        )
      : undefined;
    const {
      promise: tiledImagesPromise,
      resolve: resolveTiledImagesPromise,
      reject: rejectTiledImagesPromise,
    } = AsyncUtils.withResolvers<OpenSeadragon.TiledImage[]>();
    tiledImagesPromise.catch(() => {}); // prevent unhandled rejections in console
    const newRenderedObject: RenderedObject<TObject, TObjectData> = {
      ref: newRef,
      tileSourceCount: tileSources.length,
      usesBackdrop: useBackdrop,
      tiledImagesPromise,
    };
    let backdropPromise: Promise<OpenSeadragon.TiledImage> | undefined;
    if (useBackdrop && backdropTileSourcePromise !== undefined) {
      backdropPromise = this.context.addTiledImage(
        {
          tileSource: backdropTileSourcePromise,
          opacity: 0, // only make visible once transformed
          // OBS: explicitly setting this would exclude the backdrop from the WebGL drawer's batched path!
          // compositeOperation: "source-over",
        },
        {
          signal,
          getIndex: () => {
            if (this._anchor !== undefined) {
              const anchorIndex = this.context.getTiledImageIndex(this._anchor);
              if (anchorIndex !== -1) {
                return anchorIndex + 1 + offset;
              }
            }
            return undefined;
          },
        },
      );
      backdropPromise.catch(() => {}); // prevent unhandled rejections in console
    }
    const tiledImagePromises = tileSourcePromises.map(
      (tileSourcePromise, index) => {
        const tiledImagePromise = this.context.addTiledImage(
          {
            tileSource: tileSourcePromise.catch(getPlaceholderTileSource),
            opacity: 0, // only make visible once transformed
            ...(useBackdrop && { compositeOperation: "lighter" }),
          },
          {
            signal,
            getIndex: () => {
              if (this._anchor !== undefined) {
                const anchorIndex = this.context.getTiledImageIndex(
                  this._anchor,
                );
                if (anchorIndex !== -1) {
                  return (
                    anchorIndex + 1 + offset + (useBackdrop ? 1 : 0) + index
                  );
                }
              }
              return undefined;
            },
          },
        );
        tiledImagePromise.catch(() => {}); // prevent unhandled rejections in console
        return tiledImagePromise;
      },
    );
    Promise.all([
      Promise.allSettled(tileSourcePromises),
      Promise.allSettled([backdropPromise, ...tiledImagePromises]),
    ])
      .then(async ([tileSourceResults, results]) => {
        const [backdropResult, ...tiledImageResults] = results;
        const backdrop =
          backdropResult?.status === "fulfilled"
            ? backdropResult.value
            : undefined;
        const tiledImages = tiledImageResults
          .filter((result) => result.status === "fulfilled")
          .map((result) => result.value);
        if (this.context.isDestroyed()) {
          return tiledImages; // the viewer tears down its world itself
        }
        const failure = [...tileSourceResults, ...results].find(
          (result) => result.status === "rejected",
        );
        if (
          signal?.aborted ||
          failure !== undefined ||
          newRenderedObject.pendingDelete ||
          this._destroyed
        ) {
          this._renderedObjects = this._renderedObjects.filter(
            (renderedObject) => renderedObject !== newRenderedObject,
          );
          const addedTiledImages =
            backdrop !== undefined ? [backdrop, ...tiledImages] : tiledImages;
          await Promise.all(
            addedTiledImages.map((tiledImage) =>
              this.context.removeTiledImage(tiledImage),
            ),
          );
          signal?.throwIfAborted();
          if (failure !== undefined) {
            console.error(
              `Failed to add tiled images for object with ID '${newRef.object.id}'`,
              failure.reason,
            );
            throw new Error("Failed to add tiled images", {
              cause: failure.reason,
            });
          }
        } else {
          newRenderedObject.backdrop = backdrop;
          newRenderedObject.tiledImages = tiledImages;
          // transform first, while still hidden, so that the context's world
          // background covers the tiled images by the time they are drawn
          newRenderedObject.pendingBackground = true;
          this._updateRenderedObject(newRenderedObject);
          await this._updateBoundsAndReveal([newRenderedObject], { signal });
        }
        return tiledImages;
      })
      .then(resolveTiledImagesPromise, rejectTiledImagesPromise);
    return newRenderedObject;
  }

  /**
   * Stores a reference on the rendered object and applies the current model to its backdrop and TiledImages
   *
   * The reference holds the layer and object as they were when it was loaded,
   * and is what the next synchronization matches against (see
   * {@link _cleanRenderedObjects}). The transform, visibility and opacity
   * applied to the TiledImages are not read from it, but from the current
   * model, as they are the properties {@link setModel} applies without a
   * synchronization: read from the reference, a call from {@link setModel}
   * would re-apply the values of the last synchronization, and a call from a
   * synchronization that was in flight while the model changed would overwrite
   * what {@link setModel} applied.
   *
   * Nothing is applied if the layer or object has left the model, the object
   * has moved to another layer, or its data source has changed: each of these
   * requires a resynchronization, which drops or recreates the rendered object.
   * The last one also protects the reference's data, which its provider may
   * release once the data source it was loaded for is gone from the model.
   *
   * The opacity is computed per TiledImage via {@link getTiledImageOpacity}, so
   * that subclasses can vary it by channel; all other properties are shared by
   * all TiledImages of an object, and by its backdrop. The backdrop is opaque
   * where the object is, so it gets {@link getTiledImageOpacity} without a
   * channel index. A rendered object that is `pendingBackground` gets
   * everything but its opacity, which stays at zero; its tiles are loaded
   * nonetheless, unless it is invisible anyway, so that they are ready by the
   * time it is revealed.
   *
   * @param renderedObject - The rendered object to update
   * @param newRef - The reference to store, addressing the same layer, object
   * and data source as the current one; defaults to the current one
   * @throws Error if the TiledImages have not been created yet
   */
  private _updateRenderedObject(
    renderedObject: RenderedObject<TObject, TObjectData>,
    newRef: ObjectRef<TObject, TObjectData> = renderedObject.ref,
  ): void {
    if (renderedObject.tiledImages === undefined) {
      throw new Error("Rendered object not loaded");
    }
    renderedObject.ref = newRef;
    const layer = this._model?.layers.find(
      (layer) => layer.id === newRef.layer.id,
    );
    const object = this._model?.objects.find(
      (object) => object.id === newRef.object.id,
    );
    if (
      layer === undefined ||
      object === undefined ||
      object.layer !== layer.id ||
      !deepEqual(object.dataSource, newRef.object.dataSource)
    ) {
      return;
    }
    const currentRef = { layer, object, data: newRef.data };
    if (renderedObject.backdrop !== undefined) {
      this._updateTiledImage(renderedObject.backdrop, currentRef, null, {
        hidden: renderedObject.pendingBackground,
      });
    }
    for (let index = 0; index < renderedObject.tiledImages.length; index++) {
      const tiledImage = renderedObject.tiledImages[index]!;
      this._updateTiledImage(tiledImage, currentRef, index, {
        hidden: renderedObject.pendingBackground,
      });
    }
  }

  /**
   * Deletes the rendered object by removing its backdrop and TiledImages from the OpenSeadragon viewer, or marking it for deletion if the TiledImages have not yet been created
   *
   * The removals are queued by the context (see
   * {@link OpenSeadragonContext.removeTiledImage}) and applied before any tiled
   * image requested after them, so a caller that deletes before it creates still
   * gets the world indices it expects.
   *
   * Deleting a rendered object does not remove it from {@link _renderedObjects}.
   * An object that is `pendingBackground` is no longer, so that it is not
   * revealed once the bounds update it waits for settles (see
   * {@link _updateBoundsAndReveal}), which would update TiledImages that are
   * no longer in the world.
   *
   * @param renderedObject - The rendered object to delete
   * @returns A promise that resolves once its backdrop and all of its TiledImages have been removed
   */
  private _deleteRenderedObject(
    renderedObject: RenderedObject<TObject, TObjectData>,
  ): Promise<void> {
    renderedObject.pendingBackground = false;
    if (renderedObject.tiledImages === undefined) {
      renderedObject.pendingDelete = true;
      return Promise.resolve();
    }
    const promises = renderedObject.tiledImages.map((tiledImage) =>
      this.context.removeTiledImage(tiledImage),
    );
    if (renderedObject.backdrop !== undefined) {
      promises.push(this.context.removeTiledImage(renderedObject.backdrop));
    }
    return Promise.all(promises).then(() => {});
  }

  /**
   * Applies the transform, opacity and data transfer of an object reference to a single TiledImage
   *
   * Only properties whose value actually changed are written, as each write
   * triggers a redraw. The data transfer is applied to the TiledImage's tiles
   * (see {@link OpenSeadragonContext.updateTiledImageDataTransfer}).
   *
   * @param tiledImage - The TiledImage to update
   * @param ref - The object reference whose transform to apply
   * @param index - The index of the tiled image (e.g. channel), or `null` for the object's backdrop
   * @param options - Whether to keep the TiledImage hidden, i.e. at opacity
   * `0`, while preloading its tiles if it would be visible otherwise
   */
  private _updateTiledImage(
    tiledImage: OpenSeadragon.TiledImage,
    ref: ObjectRef<TObject, TObjectData>,
    index: number | null,
    options?: { hidden?: boolean },
  ): void {
    const { hidden = false } = options ?? {};
    // transform --> flip, width, rotation, position
    // The bounds are taken without rotation, as OpenSeadragon rotates them
    // around the image center, which would offset the position of any rotated
    // tiled image and thus trigger a redundant write on every update.
    const bounds = tiledImage.getBoundsNoRotate();
    const transform = OpenSeadragonUtils.getTiledImageTransform(
      ref.object.transform,
      ref.layer.transform,
      tiledImage.getContentSize(),
    );
    if (tiledImage.getFlip() !== transform.flip) {
      tiledImage.setFlip(transform.flip);
    }
    if (bounds.width !== transform.width) {
      tiledImage.setWidth(transform.width, true); // implicitly updates height to maintain aspect ratio
    }
    if (tiledImage.getRotation() !== transform.rotation) {
      tiledImage.setRotation(transform.rotation, true);
    }
    if (
      bounds.x !== transform.position.x ||
      bounds.y !== transform.position.y
    ) {
      tiledImage.setPosition(transform.position, true);
    }
    // visibility & opacity --> opacity, preload
    const visibleOpacity = this.getTiledImageOpacity(ref, index);
    const opacity = hidden ? 0 : visibleOpacity;
    const preload = hidden && visibleOpacity > 0;
    if (tiledImage.getPreload() !== preload) {
      tiledImage.setPreload(preload);
    }
    const oldOpacity = tiledImage.getOpacity();
    if (opacity !== oldOpacity) {
      tiledImage.setOpacity(opacity);
      if (oldOpacity === 0 && opacity > 0) {
        // OpenSeadragon does not load tiles for invisible images,
        // so we need to trigger a reload when an image becomes visible
        tiledImage.update(/* viewportChanged */ false);
      }
    }
    // (channel/label) values --> data transfer
    const dataTransfer = this.resolveTiledImageDataTransfer(ref, index);
    this.context.updateTiledImageDataTransfer(tiledImage, dataTransfer);
  }

  /**
   * Updates the bounds (see {@link updateBounds}), and then reveals the given rendered objects that are `pendingBackground`
   *
   * The objects are revealed once the context's world background covers them,
   * and also if that failed for any other reason than an abort: an object
   * shown on an uncovered canvas, where its channels may be composited wrongly
   * (see {@link usesAdditiveBlending}), beats one that is not shown at all. An
   * aborted update leaves them hidden, for the synchronization that follows to
   * reveal them (see {@link synchronize}).
   *
   * @param renderedObjects - The rendered objects to reveal
   * @param options - Optional abort signal
   * @returns A promise that resolves once the bounds have been updated
   */
  private async _updateBoundsAndReveal(
    renderedObjects: RenderedObject<TObject, TObjectData>[],
    options?: { signal?: AbortSignal },
  ): Promise<void> {
    const { signal } = options ?? {};
    let updated = false;
    try {
      await this.updateBounds({ signal });
      updated = true;
    } finally {
      if (updated || !signal?.aborted) {
        for (const renderedObject of renderedObjects) {
          if (renderedObject.pendingBackground === true) {
            renderedObject.pendingBackground = false;
            this._updateRenderedObject(renderedObject);
          }
        }
      }
    }
  }

  /**
   * Resizes the anchor (see the class documentation), or creates it
   *
   * If the anchor already has `newBounds` (see
   * {@link OpenSeadragonUtils.hasBounds}), it is kept. Otherwise, a new anchor
   * is created at `anchorIndex` (defaulting to the index of the current
   * anchor, or appended if neither is specified) and the current anchor is
   * removed. Where possible, OpenSeadragon replaces the current anchor as part
   * of the addition, so that the new anchor takes its place without leaving a
   * gap. Replacing the anchor cannot be aborted once the new anchor has been
   * created, as that would leave the renderer without an anchor.
   *
   * The viewport is not fitted here: it follows the bounds of the world as a
   * whole, for as long as the renderers own it (see
   * {@link OpenSeadragonContext.resetViewport}). Reads and writes
   * {@link _anchor}, so it must only be called from within an anchor task (see
   * {@link _enqueueAnchorTask}).
   *
   * @param newBounds - The new bounds of the anchor, in world coordinates
   * @param options - Optional abort signal, and index at which to insert the
   * new anchor
   * @returns A promise that resolves once the anchor has the new bounds
   */
  private async _resizeAnchor(
    newBounds: Rect,
    options?: { signal?: AbortSignal; anchorIndex?: number },
  ): Promise<void> {
    const { signal, anchorIndex } = options ?? {};
    signal?.throwIfAborted();
    const anchor = this._anchor;
    if (
      anchor !== undefined &&
      OpenSeadragonUtils.hasBounds(anchor, newBounds)
    ) {
      return;
    }
    let replace = undefined;
    let getIndex = undefined;
    if (anchorIndex === undefined && anchor !== undefined) {
      replace = true;
      getIndex = () => this.context.getTiledImageIndex(anchor);
    }
    this._anchor = await this.context.addTiledImage(
      {
        index: anchorIndex,
        replace,
        x: newBounds.x,
        y: newBounds.y,
        width: newBounds.width,
        tileSource: OpenSeadragonUtils.createPixelTileSource(
          { width: newBounds.width, height: newBounds.height },
          OpenSeadragonUtils.transparentBlackPixelUrl,
        ),
        opacity: 0,
      },
      { signal, getIndex },
    );
    if (anchor !== undefined && replace !== true) {
      await this.context.removeTiledImage(anchor);
    }
  }

  /**
   * Appends a task to the anchor task queue
   *
   * Tasks are run one at a time, in call order, and a failing task does not
   * prevent subsequent tasks from running. Every task that reads or writes
   * {@link _anchor} has to be enqueued here: concurrent tasks would each replace
   * the anchor they captured, leaving the anchors created in between orphaned in
   * the world, which shifts all subsequent world indices and thereby invalidates
   * the layout expected by {@link _cleanRenderedObjects}.
   *
   * The queue is per renderer, and separate from the context's addition queue,
   * which anchor tasks enqueue onto themselves.
   *
   * @param task - Task to run once all previously enqueued tasks have settled
   * @returns A promise that resolves with the task's result
   */
  private _enqueueAnchorTask<T>(task: () => T | Promise<T>): Promise<T> {
    const result = this._anchorTaskPromise.then(task);
    this._anchorTaskPromise = result.catch(() => {}); // prevent unhandled rejections in console
    return result;
  }
}

/**
 * A reference to either an image or labels object on a specific layer
 *
 * The reference of a rendered object carries the layer and object as they were
 * when the reference was loaded; the properties applied to its tiled images
 * come from the current model instead (see
 * {@link OpenSeadragonRendererBase._updateRenderedObject}).
 */
export type ObjectRef<
  TObject extends Image | Labels,
  TObjectData extends ImageData | LabelsData,
> = {
  layer: Layer;
  object: TObject;
  data: TObjectData;
};

/**
 * Mutable state for the tiled images of a single object in the viewer
 *
 * An object occupies `tileSourceCount` consecutive world indices, one tiled
 * image per channel, in the order of its tile sources, preceded by that of its
 * `backdrop`: the opaque black tiled image that the channels of an additively
 * blended object add up on (see
 * {@link OpenSeadragonRendererBase.usesAdditiveBlending}). The count, and
 * whether there is a backdrop (`usesBackdrop`), are known as soon as the
 * rendered object is created, whereas `backdrop` and `tiledImages` are assigned
 * only once all of them have been added to the world, which is also when
 * `tiledImagesPromise` resolves, with `tiledImages` alone. The former are the
 * footprint that the object is going to have, which the synchronization that
 * creates it needs to place the objects behind it before its tiled images
 * exist; every later synchronization counts the latter, i.e. the footprint
 * that the object actually has (see
 * {@link OpenSeadragonRendererBase._cleanRenderedObjects}).
 *
 * A newly created object is `pendingBackground` from the moment its tiled
 * images are assigned until the context's world background has been resized
 * to cover them: while pending, its backdrop and tiled images are kept at
 * opacity zero, but preloaded, whoever updates them (see
 * {@link OpenSeadragonRendererBase._updateRenderedObject}).
 */
export type RenderedObject<
  TObject extends Image | Labels,
  TObjectData extends ImageData | LabelsData,
> = {
  ref: ObjectRef<TObject, TObjectData>;
  tileSourceCount: number;
  usesBackdrop: boolean;
  tiledImagesPromise: Promise<OpenSeadragon.TiledImage[]>;
  tiledImages?: OpenSeadragon.TiledImage[];
  backdrop?: OpenSeadragon.TiledImage;
  pendingDelete?: boolean;
  pendingBackground?: boolean;
};
