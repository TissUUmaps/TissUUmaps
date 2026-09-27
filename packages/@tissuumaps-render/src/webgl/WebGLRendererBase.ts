import { deepEqual } from "fast-equals";

import {
  GeometryUtils,
  type IDArray,
  type Layer,
  type Points,
  type PointsData,
  type Rect,
  type Shapes,
  type ShapesData,
  type Table,
  type TableData,
  TableUtils,
  TransformUtils,
} from "@tissuumaps/core";

import type { WebGLContext } from "./WebGLContext";
import { WebGLUtils } from "./WebGLUtils";

type ItemsInfo = { itemIds: IDArray; itemsMask: Uint8Array };

/**
 * Base class for WebGL renderers that draw the items of objects (points or shapes)
 *
 * {@link setModel} sets the layers and objects to render. The properties that
 * are applied when drawing (transforms, visibility, opacity, draw order) are
 * read from the current model on every draw (see {@link getRenderPasses}).
 * Everything else is built into GPU resources by {@link synchronize}, which
 * {@link needsSynchronization} tells when to call, through the
 * {@link prepareRenderedObject}, {@link createRenderedObject} and
 * {@link updateRenderedObject} hooks that the renderers implement.
 *
 * Objects are assigned to a layer either as a whole, by layer ID, or per item,
 * by a table column; {@link loadObjects} resolves both into one
 * {@link ObjectRef} per object and layer. The per-item assignment is cached by
 * the identity of the object and table data (see {@link _getLayerItemsInfos}),
 * so the loaders have to return immutable data that keeps its identity for as
 * long as its content is unchanged. Likewise, the renderers detect an edit to
 * a group-to-value map by comparing the maps that their objects' configurations
 * resolve from by identity (see `ConfigUtils.findGroupByMap`), so the maps
 * passed to a synchronization have to keep their identity for as long as they
 * are unchanged.
 */
export abstract class WebGLRendererBase<
  TObject extends Points | Shapes,
  TObjectData extends PointsData | ShapesData,
  TSyncContext extends {
    tables: Table[];
    loadObject: (
      object: TObject,
      options?: { signal?: AbortSignal },
    ) => Promise<TObjectData>;
    loadTable: (
      table: Table,
      options?: { signal?: AbortSignal },
    ) => Promise<TableData>;
  },
  TPreparedObject,
  TRenderedObject extends RenderedObjectBase<TObject, TObjectData>,
> {
  readonly context: WebGLContext;
  viewport: Rect = { x: 0, y: 0, width: 1, height: 1 };

  private _model?: { layers: Layer[]; objects: TObject[] };
  private _lastSyncState?: object;
  private _hasUnreportedChanges = false;
  private readonly _renderedObjects = new Map<
    string,
    Map<string, TRenderedObject>
  >();
  private readonly _layerItemsInfosCache = new WeakMap<
    TObjectData,
    WeakMap<TableData, Map<string, Map<string, ItemsInfo | null>>>
  >();

  /**
   * Creates a new WebGLRendererBase instance
   *
   * @param context - The WebGL context to use for rendering
   */
  constructor(context: WebGLContext) {
    this.context = context;
  }

  /**
   * Sets the layers and objects to render
   *
   * Layer- and object-level properties are drawn from uniforms, and the draw
   * order is the order of the model, so a change that affects nothing else
   * needs nothing done here: both are read from the new model by the next draw
   * (see {@link getRenderPasses}).
   * Every other change - a different set of layers or objects, layer
   * memberships, data sources or item-level configurations - requires a
   * resynchronization, which the caller is expected to trigger whenever
   * {@link needsSynchronization} says so.
   *
   * Does not redraw: a model that differs at all differs either in a property
   * that is applied when drawing, or in one that requires a resynchronization,
   * so the caller is expected to redraw whenever this returns `true`. A model
   * equal to the current one is ignored.
   *
   * The layers and objects are cloned, so that what the renderer compares
   * against later - in {@link needsSynchronization}, and in the references of
   * a synchronization - is what it was given, whatever the caller does with its
   * instances afterwards.
   *
   * @param layers - The layers to render
   * @param objects - The objects (points or shapes) to render
   * @returns Whether the model differs from the current one
   */
  setModel(layers: Layer[], objects: TObject[]): boolean {
    if (
      this._model !== undefined &&
      deepEqual(this._model, { layers, objects })
    ) {
      return false;
    }
    this._model = structuredClone({ layers, objects });
    return true;
  }

  /**
   * Gets the bounding box of all drawn objects, in world coordinates
   *
   * Reads the transforms from the current model (see
   * {@link getRenderPasses}), so the bounds follow a model that requires no
   * resynchronization.
   *
   * @returns The bounds, or null if nothing is drawn
   */
  getRenderedBounds(): Rect | null {
    return this.getRenderPasses().reduce<Rect | null>(
      (union, { layer, object, renderedObject }) => {
        const bounds = TransformUtils.transformBoundingBox(
          renderedObject.objectBounds,
          WebGLUtils.createDataToWorldMatrix(object.transform, layer.transform),
        );
        return union !== null ? GeometryUtils.union(union, bounds) : bounds;
      },
      null,
    );
  }

  /**
   * Returns whether the rendered objects have to be synchronized with the
   * current model and render options
   *
   * Compares the state that a synchronization depends on (see
   * {@link getSyncState}) against the one the last synchronization was based
   * on (see {@link _recordSyncState}). Everything else about the model - the
   * properties that are applied when drawing, and the order of the layers and
   * objects - is fully applied by {@link setModel}, and everything else about
   * the render options by the next draw.
   *
   * @returns Whether {@link synchronize} has to be called
   */
  needsSynchronization(): boolean {
    return !deepEqual(this.getSyncState(), this._lastSyncState);
  }

  /**
   * Synchronizes the rendered objects with the current model and render options
   *
   * Loads the objects of the model (see {@link loadObjects}), drops the
   * rendered objects that no longer match one (see
   * {@link matchOrDestroyRenderedObjects}), and then creates or updates the
   * rest in two passes. The first prepares every object (see
   * {@link prepareRenderedObject}), issuing all requests before the first
   * `await`: requests are shared with, and cancelled once abandoned by, a
   * superseded synchronization, unless they are reclaimed within the same
   * task. The second awaits the preparations in order and uploads each of them
   * in one synchronous block (see {@link createRenderedObject} and
   * {@link updateRenderedObject}), so a draw never sees a half-updated object.
   *
   * An object whose preparation fails, or finds nothing to render, is logged
   * and dropped. A synchronization that fails or is aborted leaves the model
   * unsynchronized (see {@link _discardSyncState}), and the changes it made
   * before it stopped are reported by the next one that completes.
   *
   * @param syncContext - The inputs to synchronize with: the tables and
   * group-to-value maps that the objects resolve their properties from, and
   * the loaders for object and table data. It carries inputs only; what an
   * object was resolved from is captured per object (see {@link ObjectRef})
   * @param options - Optional abort signal
   * @returns A promise that resolves to whether any rendered object changed
   * since the last completed synchronization
   * @throws Error if no model has been set (see {@link setModel})
   */
  async synchronize(
    syncContext: TSyncContext,
    options?: { signal?: AbortSignal },
  ): Promise<boolean> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const syncState = this._recordSyncState();
    try {
      const newRefs = await this.loadObjects(syncContext, { signal });
      const matches = this.matchOrDestroyRenderedObjects(newRefs);
      const preparations = matches.map(({ newRef, renderedObject }) => {
        const preparedPromise = this.prepareRenderedObject(
          newRef,
          renderedObject,
          syncContext,
          { signal },
        );
        preparedPromise.catch(() => {}); // prevent unhandled rejections in console
        return { newRef, renderedObject, preparedPromise };
      });
      for (const { newRef, renderedObject, preparedPromise } of preparations) {
        const prepared = await this._settleRenderedObjectPreparation(
          newRef,
          renderedObject,
          preparedPromise,
          { signal },
        );
        if (prepared === undefined) {
          continue;
        }
        // no awaits from here on, so that the object is uploaded atomically
        if (renderedObject === undefined) {
          this.addRenderedObject(this.createRenderedObject(newRef, prepared));
        } else if (this.updateRenderedObject(renderedObject, prepared)) {
          this._hasUnreportedChanges = true;
        }
      }
    } catch (error) {
      this._discardSyncState(syncState);
      throw error;
    }
    const hasUnreportedChanges = this._hasUnreportedChanges;
    this._hasUnreportedChanges = false;
    return hasUnreportedChanges;
  }

  /**
   * Returns the state of the current model and render options that a
   * synchronization depends on
   *
   * The layers and objects are keyed by ID, so that their order - the draw order
   * - is not part of the state (see {@link getLayerSyncState} and
   * {@link getObjectSyncState} for what is, and
   * {@link getRenderOptionsSyncState} for the render options).
   *
   * @returns The state, or `undefined` if no model has been set
   */
  protected getSyncState(): object | undefined {
    if (this._model === undefined) {
      return undefined;
    }
    return {
      layers: Object.fromEntries(
        this._model.layers.map((layer) => [
          layer.id,
          this.getLayerSyncState(layer),
        ]),
      ),
      objects: Object.fromEntries(
        this._model.objects.map((object) => [
          object.id,
          this.getObjectSyncState(object),
        ]),
      ),
      renderOptions: this.getRenderOptionsSyncState(),
    };
  }

  /**
   * Returns the state of a layer that a synchronization depends on
   *
   * The counterpart of {@link getObjectSyncState} for layers. The point size
   * factor is blanked out as well: it is a uniform of the points renderer, and
   * the shapes renderer never reads it. So is the name, which nothing rendered
   * depends on.
   *
   * @param layer - The layer to return the state of
   * @returns The layer without the properties that are applied when drawing,
   * and without the cosmetic ones
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
   * Everything but the properties that are applied when drawing, which are read
   * from the current model on every draw (see {@link getRenderPasses}) and
   * hence need no synchronization, and but the cosmetic ones, which nothing
   * rendered depends on. Those are blanked out rather than dropped, so that a
   * property added to the model later is part of the state, and thereby
   * requires a resynchronization, unless it is blanked out here as well.
   *
   * The result is only ever deep-compared against that of another object, hence
   * the opaque return type.
   *
   * @param object - The object (points or shapes) to return the state of
   * @returns The object without the properties that are applied when drawing,
   * and without the cosmetic ones
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
   * Returns the state of the render options that a synchronization depends on
   *
   * Render options that are applied when drawing need no synchronization and
   * are left out. Nothing by default; a renderer whose GPU resources are built
   * for a render option overrides this to return it.
   *
   * @returns The state, or `undefined` if no render option needs a synchronization
   */
  protected getRenderOptionsSyncState(): object | undefined {
    return undefined;
  }

  /**
   * Concurrently loads the data of all objects to be rendered on the layers of the current model
   *
   * An object assigned to a layer by layer ID is loaded for that layer only, an
   * object assigned per item by a table column is loaded for every layer, with
   * the items on each layer resolved from the table (see
   * {@link _getLayerItemsInfos}). The data of an object, and of a table, is
   * loaded once, no matter how many references it is shared by.
   *
   * The returned references are ordered by layer and then by object. Objects
   * whose data or table failed to load are logged and skipped, and so are
   * objects whose items are assigned per item without a table to resolve the
   * assignment from. Objects without items on a layer are skipped silently,
   * which also covers empty objects and objects whose table is empty - those
   * are legitimate states, not failures.
   *
   * @param syncContext - The inputs of the current synchronization: the tables
   * that the objects resolve their item layers from, and the loaders for
   * object and table data
   * @param options - Optional abort signal
   * @returns A promise that resolves to one reference per loaded object and layer
   * @throws Error if no model has been set (see {@link setModel})
   */
  protected async loadObjects(
    syncContext: TSyncContext,
    options?: { signal?: AbortSignal },
  ): Promise<ObjectRef<TObject, TObjectData>[]> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const model = this._model;
    if (model === undefined) {
      throw new Error("Model not set");
    }
    const dataPromises = new Map<string, Promise<TObjectData>>();
    const tableDataPromises = new Map<string, Promise<TableData>>();
    const tableLayersPromises = new Map<string, Promise<string[]>>();
    const layerItemsInfosPromises = new Map<
      string,
      Promise<Map<string, ItemsInfo | null>>
    >();
    const newRefPromises: Promise<ObjectRef<TObject, TObjectData>>[] = [];
    const objectIdsWithoutTable = new Set<string>();
    for (const currentLayer of model.layers) {
      for (const currentObject of model.objects) {
        if (
          typeof currentObject.layer === "string" &&
          currentObject.layer !== currentLayer.id
        ) {
          continue;
        }
        if (
          typeof currentObject.layer !== "string" &&
          currentObject.dataSource.table === undefined
        ) {
          if (!objectIdsWithoutTable.has(currentObject.id)) {
            objectIdsWithoutTable.add(currentObject.id);
            console.error(
              `Object with ID '${currentObject.id}' assigns its items to layers by column '${currentObject.layer.column}', but has no table`,
            );
          }
          continue;
        }
        let dataPromise = dataPromises.get(currentObject.id);
        if (dataPromise === undefined) {
          dataPromise = syncContext.loadObject(currentObject, { signal });
          dataPromise.catch((error) => {
            if (!signal?.aborted) {
              console.error(
                `Failed to load object with ID '${currentObject.id}'`,
                error,
              );
            }
          });
          dataPromises.set(currentObject.id, dataPromise);
        }
        let layerItemsInfosPromise;
        if (
          typeof currentObject.layer !== "string" &&
          currentObject.dataSource.table !== undefined
        ) {
          let tableDataPromise = tableDataPromises.get(
            currentObject.dataSource.table,
          );
          if (tableDataPromise === undefined) {
            const table = syncContext.tables.find(
              (table) => table.id === currentObject.dataSource.table,
            );
            if (table !== undefined) {
              tableDataPromise = syncContext.loadTable(table, { signal });
            } else {
              tableDataPromise = Promise.reject(
                new Error(
                  `Table with ID '${currentObject.dataSource.table}' not found`,
                ),
              );
            }
            tableDataPromise.catch((error) => {
              if (!signal?.aborted) {
                console.error(
                  `Failed to load table with ID '${currentObject.dataSource.table}'`,
                  error,
                );
              }
            });
            tableDataPromises.set(
              currentObject.dataSource.table,
              tableDataPromise,
            );
          }
          const tableId = currentObject.dataSource.table;
          const tableLayersColumn = currentObject.layer.column;
          const tableLayersPromiseKey = `${tableId}:${tableLayersColumn}`;
          let tableLayersPromise = tableLayersPromises.get(
            tableLayersPromiseKey,
          );
          if (tableLayersPromise === undefined) {
            tableLayersPromise = tableDataPromise.then(async (tableData) => {
              const tableLayers = await tableData.loadValues<string>(
                tableLayersColumn,
                { signal },
              );
              if (tableLayers.length !== tableData.getSize()) {
                throw new Error(
                  `Table with ID '${currentObject.dataSource.table}' has inconsistent size for column '${tableLayersColumn}'`,
                );
              }
              return tableLayers;
            });
            tableLayersPromise.catch((error) => {
              if (!signal?.aborted) {
                console.error(
                  `Failed to load layers from table with ID '${currentObject.dataSource.table}' (column '${tableLayersColumn}')`,
                  error,
                );
              }
            });
            tableLayersPromises.set(tableLayersPromiseKey, tableLayersPromise);
          }
          layerItemsInfosPromise = layerItemsInfosPromises.get(
            currentObject.id,
          );
          if (layerItemsInfosPromise === undefined) {
            layerItemsInfosPromise = Promise.all([
              dataPromise,
              tableDataPromise,
              tableLayersPromise,
            ]).then(([data, tableData, tableLayers]) => {
              const promise = this._getLayerItemsInfos(
                data,
                tableData,
                tableLayersColumn,
                tableLayers,
                model.layers,
                { signal },
              );
              promise.catch((error) => {
                if (!signal?.aborted) {
                  console.error(
                    `Failed to assign the items of object with ID '${currentObject.id}' to layers`,
                    error,
                  );
                }
              });
              return promise;
            });
            layerItemsInfosPromises.set(
              currentObject.id,
              layerItemsInfosPromise,
            );
          }
        }
        const newRefPromise = Promise.all([
          dataPromise,
          layerItemsInfosPromise,
        ]).then(([data, layerItemsInfos]) => {
          signal?.throwIfAborted();
          if (layerItemsInfos !== undefined) {
            const itemsInfo = layerItemsInfos.get(currentLayer.id);
            return {
              layerId: currentLayer.id,
              object: currentObject,
              itemIds: itemsInfo?.itemIds ?? [],
              itemsMask: itemsInfo?.itemsMask,
              data,
            };
          }
          return {
            layerId: currentLayer.id,
            object: currentObject,
            itemIds: data.getIds(),
            itemsMask: undefined,
            data,
          };
        });
        newRefPromises.push(newRefPromise);
      }
    }
    const results = await Promise.allSettled(newRefPromises);
    signal?.throwIfAborted();
    return results
      .filter((result) => result.status === "fulfilled")
      .filter((result) => result.value.itemIds.length > 0)
      .map((result) => result.value);
  }

  /**
   * Matches the rendered objects to a new set of references, and destroys the
   * ones left over
   *
   * A rendered object is matched by the layer and object it is kept under, and
   * kept if its contributed items and data source are those of the reference,
   * which it then adopts and is returned for reuse with. Every other rendered
   * object - one that no reference addresses, or one whose items or data source
   * changed - is destroyed and dropped.
   *
   * @param newRefs - The object references to match against
   * @returns One match per reference, in the order of the references, each with
   * the rendered object to reuse for it, or `undefined` if it has none yet
   */
  protected matchOrDestroyRenderedObjects(
    newRefs: ObjectRef<TObject, TObjectData>[],
  ): {
    newRef: ObjectRef<TObject, TObjectData>;
    renderedObject: TRenderedObject | undefined;
  }[] {
    const matchedRenderedObjects = new Set<TRenderedObject>();
    const matches = newRefs.map((newRef) => {
      const renderedObject = this._renderedObjects
        .get(newRef.layerId)
        ?.get(newRef.object.id);
      if (
        renderedObject !== undefined &&
        renderedObject.ref.itemIds === newRef.itemIds &&
        renderedObject.ref.itemsMask === newRef.itemsMask &&
        // check data source configuration instead of data
        deepEqual(
          renderedObject.ref.object.dataSource,
          newRef.object.dataSource,
        )
      ) {
        renderedObject.ref = newRef;
        matchedRenderedObjects.add(renderedObject);
        return { newRef, renderedObject };
      }
      return { newRef, renderedObject: undefined };
    });
    for (const renderedObjects of this._renderedObjects.values()) {
      for (const [objectId, renderedObject] of renderedObjects) {
        if (!matchedRenderedObjects.has(renderedObject)) {
          renderedObjects.delete(objectId);
          this.destroyRenderedObject(renderedObject);
          this._hasUnreportedChanges = true;
        }
      }
    }
    return matches;
  }

  /**
   * Adds a newly created rendered object under its layer and object
   *
   * @param renderedObject - The rendered object to add
   */
  protected addRenderedObject(renderedObject: TRenderedObject): void {
    const { layerId, object } = renderedObject.ref;
    let renderedObjects = this._renderedObjects.get(layerId);
    if (renderedObjects === undefined) {
      renderedObjects = new Map();
      this._renderedObjects.set(layerId, renderedObjects);
    }
    renderedObjects.set(object.id, renderedObject);
    this._hasUnreportedChanges = true;
  }

  /**
   * Drops a rendered object and releases its GPU resources
   *
   * @param renderedObject - The rendered object to remove
   */
  protected removeRenderedObject(renderedObject: TRenderedObject): void {
    this._renderedObjects
      .get(renderedObject.ref.layerId)
      ?.delete(renderedObject.ref.object.id);
    this.destroyRenderedObject(renderedObject);
    this._hasUnreportedChanges = true;
  }

  /**
   * Drops all rendered objects and releases their GPU resources
   */
  protected clearRenderedObjects(): void {
    for (const renderedObjects of this._renderedObjects.values()) {
      for (const renderedObject of renderedObjects.values()) {
        this.destroyRenderedObject(renderedObject);
      }
    }
    this._renderedObjects.clear();
    this._hasUnreportedChanges = true;
  }

  /**
   * Prepares everything that has to be uploaded for an object
   *
   * Runs in the first pass of {@link synchronize}, which calls it synchronously
   * for every object of a synchronization, one after the other: every request
   * has to be issued before the first `await`. Decides, from the object's
   * current rendered state, which of its GPU resources have to be rebuilt, and
   * resolves only those.
   *
   * @param newRef - The object to prepare
   * @param renderedObject - The object's current rendered state, if it is reused
   * @param syncContext - The inputs of the current synchronization
   * @param options - Optional abort signal
   * @returns What {@link createRenderedObject} or {@link updateRenderedObject}
   * upload, or `null` if there is nothing to render for the object, which is
   * then dropped
   */
  protected abstract prepareRenderedObject(
    newRef: ObjectRef<TObject, TObjectData>,
    renderedObject: TRenderedObject | undefined,
    syncContext: TSyncContext,
    options?: { signal?: AbortSignal },
  ): Promise<TPreparedObject | null>;

  /**
   * Creates the rendered object, with its GPU resources, of a newly prepared object
   *
   * Runs in the second pass of {@link synchronize}, synchronously.
   *
   * @param newRef - The object
   * @param prepared - Its preparation, made without a rendered object to reuse
   * @returns The rendered object, which {@link synchronize} adds
   */
  protected abstract createRenderedObject(
    newRef: ObjectRef<TObject, TObjectData>,
    prepared: TPreparedObject,
  ): TRenderedObject;

  /**
   * Updates the GPU resources of a rendered object from its preparation
   *
   * Runs in the second pass of {@link synchronize}, synchronously.
   *
   * @param renderedObject - The rendered object to update in place
   * @param prepared - Its preparation, made with it as the rendered object to reuse
   * @returns Whether any GPU resource was uploaded to, i.e. whether the object
   * draws differently now
   */
  protected abstract updateRenderedObject(
    renderedObject: TRenderedObject,
    prepared: TPreparedObject,
  ): boolean;

  /**
   * Releases the GPU resources owned by a single rendered object
   *
   * Does not remove the object from {@link _renderedObjects}, see
   * {@link removeRenderedObject}.
   *
   * @param renderedObject - The rendered object whose GPU resources to release
   */
  protected abstract destroyRenderedObject(
    renderedObject: TRenderedObject,
  ): void;

  /**
   * Returns one render pass per rendered object to draw, in draw order
   *
   * A pass carries the layer and the object of the current model, and the
   * rendered object holding the GPU resources to draw them with.
   *
   * The properties that are applied when drawing - the transforms, the
   * visibility, the opacity and the point size factors - are read from the
   * layer and the object returned here, never from the reference of a rendered
   * object: the reference holds the object as it was when it was loaded (see
   * {@link ObjectRef}), whereas these are the current ones, so that setting a
   * model that only changes them takes effect without a resynchronization.
   *
   * The model is what is iterated, the way {@link loadObjects} iterates it:
   * by layer, then by object, pairing an object with the layer it is assigned
   * to by ID, or with every layer if its items are assigned to layers by a
   * table column. That order is the draw order. Rendered objects whose layer or
   * object has been removed from the model, or whose object has moved to
   * another layer, are therefore never visited, until the resynchronization
   * that the change requires drops them. Before a model is set, there is
   * nothing to draw.
   *
   * @returns The render passes, in draw order
   */
  protected getRenderPasses(): {
    layer: Layer;
    object: TObject;
    renderedObject: TRenderedObject;
  }[] {
    if (this._model === undefined) {
      return [];
    }
    const renderPasses = [];
    for (const layer of this._model.layers) {
      const renderedObjects = this._renderedObjects.get(layer.id);
      if (renderedObjects === undefined) {
        continue;
      }
      for (const object of this._model.objects) {
        if (object.layer !== layer.id && typeof object.layer === "string") {
          continue;
        }
        const renderedObject = renderedObjects.get(object.id);
        if (renderedObject !== undefined) {
          renderPasses.push({ layer, object, renderedObject });
        }
      }
    }
    return renderPasses;
  }

  /**
   * Records the current model and render options as what the rendered objects
   * are being synchronized with
   *
   * {@link synchronize} calls this first, before its first `await`: the model
   * may be set again while it runs, and {@link needsSynchronization} has to
   * report such a change against the model the synchronization actually read,
   * not against the one it found when it finished. A synchronization that
   * fails hands the returned state back to {@link _discardSyncState}, so that
   * the model is synchronized again.
   *
   * @returns The recorded state
   */
  private _recordSyncState(): object | undefined {
    this._lastSyncState = this.getSyncState();
    return this._lastSyncState;
  }

  /**
   * Forgets a recorded synchronization state, unless another synchronization
   * has recorded its own since
   *
   * Called by a synchronization that failed, so that
   * {@link needsSynchronization} reports the model as unsynchronized again. A
   * synchronization that was aborted because a newer one started must not
   * undo what the newer one recorded, hence the identity check.
   *
   * @param syncState - The state the failed synchronization recorded, as
   * returned by {@link _recordSyncState}
   */
  private _discardSyncState(syncState: object | undefined): void {
    if (this._lastSyncState === syncState) {
      this._lastSyncState = undefined;
    }
  }

  /**
   * Returns the items of an object that belong to each of the given layers
   *
   * Layers that are missing from {@link _layerItemsInfosCache} are computed in a
   * single pass over the items of the object, looking each item's row up in the
   * table (see {@link TableUtils.forEachRow}), and are added to the cache.
   * Layers that are already cached for the same object data, table data and
   * layer column are re-used as they are, so that objects sharing their data,
   * but not their table or layer column, do not evict each other's layers.
   * Cached layers that are not among the given ones are dropped, as are their
   * item masks, which take one byte per item of the object. Item IDs and masks
   * are only allocated for layers that turn out to contain items, layers
   * without items are cached as `null`.
   *
   * @param data - The data of the object to compute the item masks for
   * @param tableData - The data of the table holding the item layers
   * @param tableLayersColumn - The name of the table column holding the layer IDs
   * @param tableLayers - The values of the table column, in table item order
   * @param layers - The layers to compute the item masks for
   * @param options - Optional abort signal
   * @returns A promise that resolves to the item IDs and mask of each layer, by
   * layer ID, or to `null` for layers without items
   */
  private async _getLayerItemsInfos(
    data: TObjectData,
    tableData: TableData,
    tableLayersColumn: string,
    tableLayers: string[],
    layers: Layer[],
    options?: { signal?: AbortSignal },
  ): Promise<Map<string, ItemsInfo | null>> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    let layerItemsInfosByTableData = this._layerItemsInfosCache.get(data);
    if (layerItemsInfosByTableData === undefined) {
      layerItemsInfosByTableData = new WeakMap();
      this._layerItemsInfosCache.set(data, layerItemsInfosByTableData);
    }
    let layerItemsInfosByColumn = layerItemsInfosByTableData.get(tableData);
    if (layerItemsInfosByColumn === undefined) {
      layerItemsInfosByColumn = new Map();
      layerItemsInfosByTableData.set(tableData, layerItemsInfosByColumn);
    }
    let layerItemsInfos = layerItemsInfosByColumn.get(tableLayersColumn);
    if (layerItemsInfos === undefined) {
      layerItemsInfos = new Map();
      layerItemsInfosByColumn.set(tableLayersColumn, layerItemsInfos);
    }
    for (const layerId of layerItemsInfos.keys()) {
      if (!layers.some((layer) => layer.id === layerId)) {
        layerItemsInfos.delete(layerId);
      }
    }
    const newLayerIds = new Set<string>();
    for (const layer of layers) {
      if (!layerItemsInfos.has(layer.id)) {
        newLayerIds.add(layer.id);
      }
    }
    if (newLayerIds.size > 0) {
      const itemIds = data.getIds();
      const newItemsMasks = new Map<string, Uint8Array>();
      await TableUtils.forEachRow(
        itemIds,
        tableData,
        (rowIndex, i) => {
          if (rowIndex === undefined) {
            return;
          }
          const layerId = tableLayers[rowIndex]!;
          if (newLayerIds.has(layerId)) {
            let newItemsMask = newItemsMasks.get(layerId);
            if (newItemsMask === undefined) {
              newItemsMask = new Uint8Array(itemIds.length);
              newItemsMasks.set(layerId, newItemsMask);
            }
            newItemsMask[i] = 1;
          }
        },
        { signal },
      );
      for (const newLayerId of newLayerIds) {
        const newItemsMask = newItemsMasks.get(newLayerId);
        layerItemsInfos.set(
          newLayerId,
          newItemsMask !== undefined
            ? {
                itemIds: itemIds.filter((_, i) => newItemsMask[i] === 1),
                itemsMask: newItemsMask,
              }
            : null,
        );
      }
    }
    return layerItemsInfos;
  }

  /**
   * Awaits the preparation of an object, dropping the object if it fails or
   * finds nothing to render
   *
   * @param newRef - The prepared object
   * @param renderedObject - The object's current rendered state, if any, which
   * is removed if the object is dropped
   * @param preparedPromise - The preparation, see {@link prepareRenderedObject}
   * @param options - Optional abort signal
   * @returns The preparation, or `undefined` if the object was dropped (which
   * is logged)
   */
  private async _settleRenderedObjectPreparation(
    newRef: ObjectRef<TObject, TObjectData>,
    renderedObject: TRenderedObject | undefined,
    preparedPromise: Promise<TPreparedObject | null>,
    options?: { signal?: AbortSignal },
  ): Promise<TPreparedObject | undefined> {
    const { signal } = options ?? {};
    let prepared;
    try {
      prepared = await preparedPromise;
    } catch (error) {
      signal?.throwIfAborted();
      console.error(
        `Failed to prepare object with ID '${newRef.object.id}'`,
        error,
      );
      if (renderedObject !== undefined) {
        this.removeRenderedObject(renderedObject);
      }
      return undefined;
    }
    signal?.throwIfAborted();
    if (prepared === null) {
      console.warn(
        `Object with ID '${newRef.object.id}' has nothing to render, skipping`,
      );
      if (renderedObject !== undefined) {
        this.removeRenderedObject(renderedObject);
      }
      return undefined;
    }
    return prepared;
  }

  /**
   * Creates the loader for the table that an object resolves its properties
   * from
   *
   * @param ref - The object reference
   * @param syncContext - The inputs of the current synchronization: the tables to
   * look the object's table up in, and the loader for table data
   * @returns The loader, or `undefined` if the object has no table, or its
   * table was not found (which is logged)
   */
  protected static createObjectTableLoader(
    ref: ObjectRef<Points | Shapes, PointsData | ShapesData>,
    syncContext: {
      tables: Table[];
      loadTable: (
        table: Table,
        options?: { signal?: AbortSignal },
      ) => Promise<TableData>;
    },
  ): ((options?: { signal?: AbortSignal }) => Promise<TableData>) | undefined {
    if (ref.object.dataSource.table === undefined) {
      return undefined;
    }
    const table = syncContext.tables.find(
      (table) => table.id === ref.object.dataSource.table,
    );
    if (table === undefined) {
      console.warn(`Table with ID '${ref.object.dataSource.table}' not found`);
      return undefined;
    }
    return (options?: { signal?: AbortSignal }) =>
      syncContext.loadTable(table, options);
  }

  /**
   * Computes the factor that the alpha of every item of an object is
   * multiplied with when drawing
   *
   * @param layer - The layer the object is drawn on, as of the current model
   * @param object - The object being drawn, as of the current model
   * @returns The product of the layer and object opacities, or `0` if the
   * layer or the object is invisible
   */
  protected static computeOpacityFactor(
    layer: Layer,
    object: Points | Shapes,
  ): number {
    if (layer.visibility === false || object.visibility === false) {
      return 0;
    }
    return layer.opacity * object.opacity;
  }
}

/**
 * A reference to a points or shapes object on a specific layer
 *
 * Identifies the layer, and carries the object as it was when the reference was
 * loaded: everything that is resolved into a GPU resource is read from it, so
 * that one synchronization sees one consistent model state, however often the
 * model is set while it runs. The properties that are applied when drawing are
 * not read from it - those come from the current model, which may have moved on
 * since (see {@link WebGLRendererBase.getRenderPasses}).
 *
 * The item IDs and the item mask are cached and shared between references and
 * across calls to {@link WebGLRendererBase.loadObjects}, and must not be
 * modified.
 */
export type ObjectRef<
  TObject extends Points | Shapes,
  TObjectData extends PointsData | ShapesData,
> = {
  layerId: string;
  object: TObject;
  itemIds: IDArray;
  itemsMask: Uint8Array | undefined;
  data: TObjectData;
};

/**
 * The rendered state of one {@link ObjectRef}
 *
 * Extended by the renderers with the GPU resources they own, and with a
 * snapshot of the object properties those were built from, which their change
 * detection compares against. Rendered objects are mutated in place across
 * synchronizations:
 * {@link WebGLRendererBase.matchOrDestroyRenderedObjects} replaces the
 * reference of a matched object, and the renderers replace the bounds, the
 * snapshot and the GPU resources of an object as they are rebuilt.
 */
export type RenderedObjectBase<
  TObject extends Points | Shapes,
  TObjectData extends PointsData | ShapesData,
> = {
  /** The reference the object was last matched to, holding what its GPU resources were resolved from */
  ref: ObjectRef<TObject, TObjectData>;
  /** The bounds of the object's items, in data coordinates */
  objectBounds: Rect;
};
