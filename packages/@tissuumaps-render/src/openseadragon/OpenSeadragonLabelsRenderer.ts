import { deepEqual } from "fast-equals";

import {
  AsyncUtils,
  type Color,
  ColorUtils,
  ConfigUtils,
  type GroupValueMap,
  type Labels,
  type LabelsData,
  type Table,
  type TableData,
  defaultLabelColor,
  defaultLabelOpacity,
  defaultLabelVisibility,
  getActiveConfigSource,
} from "@tissuumaps/core";

import { ColorResolver } from "../resolvers/ColorResolver";
import { OpacityResolver } from "../resolvers/OpacityResolver";
import { VisibilityResolver } from "../resolvers/VisibilityResolver";
import type { DataTransfer } from "./OpenSeadragonContext";
import {
  type ObjectRef,
  OpenSeadragonRendererBase,
} from "./OpenSeadragonRendererBase";

export type OpenSeadragonLabelsSyncContext = {
  tables: Table[];
  colorMaps: GroupValueMap<Color>[];
  visibilityMaps: GroupValueMap<boolean>[];
  opacityMaps: GroupValueMap<number>[];
  loadObject: (
    labels: Labels,
    options?: { signal?: AbortSignal },
  ) => Promise<LabelsData>;
  loadTable: (
    table: Table,
    options?: { signal?: AbortSignal },
  ) => Promise<TableData>;
};

/**
 * Renderer for the tiled images of {@link Labels} data objects
 *
 * The tiles of a labels object carry label IDs rather than colors, and are
 * recolored by a data transfer (see
 * {@link OpenSeadragonContext.updateTiledImageDataTransfer}) that looks each
 * ID up in a color lookup table, which folds the visibility and opacity into
 * the alpha channel. Layer and object opacity are not part of it; OpenSeadragon
 * applies them when drawing the tiled image.
 *
 * A label image does not enumerate its labels (see {@link LabelsData}), so the
 * lookup table is resolved per object for the labels that the referenced table
 * lists, in the same way as the WebGL renderers resolve the appearance of their
 * items, and every other label is resolved as it is first drawn (see
 * {@link _resolveDataTransfer}).
 *
 * As data transfers are compared by identity, each object's data transfer is
 * kept (see {@link resolveObject}) until its data, one of the configurations
 * it was resolved from, or one of the group-to-value maps those configurations
 * reference changes, so that tiles are only recolored when needed. Maps are
 * never mutated - an edit replaces the map object - so they are compared by
 * identity, which is why the maps passed to a synchronization have to keep
 * their identity for as long as they are unchanged.
 */
export class OpenSeadragonLabelsRenderer extends OpenSeadragonRendererBase<
  Labels,
  LabelsData,
  OpenSeadragonLabelsSyncContext
> {
  private readonly _renderedLabels = new Map<
    string,
    {
      data: LabelsData;
      state: Pick<Labels, "labelColor" | "labelVisibility" | "labelOpacity"> & {
        labelColorMap: GroupValueMap<Color> | undefined;
        labelVisibilityMap: GroupValueMap<boolean> | undefined;
        labelOpacityMap: GroupValueMap<number> | undefined;
      };
      dataTransfer: DataTransfer;
    }
  >();

  /**
   * Drops the data transfers of all labels objects other than the given ones
   *
   * @param labels - The labels objects about to be displayed
   */
  protected override retainObjects(labels: Labels[]): void {
    for (const labelsId of this._renderedLabels.keys()) {
      if (!labels.some((currentLabels) => currentLabels.id === labelsId)) {
        this._renderedLabels.delete(labelsId);
      }
    }
  }

  /**
   * Resolves the data transfer of a labels object, unless it is up to date
   *
   * An object's data transfer is kept as long as its data, its label color,
   * visibility and opacity configurations, and the group-to-value maps those
   * configurations resolve from (see {@link ConfigUtils.findGroupByMap}) are
   * unchanged, and is resolved anew otherwise (see
   * {@link _resolveDataTransfer}). Configurations are compared by value, maps
   * by identity. If resolving from the referenced table fails, e.g. because the
   * table failed to load, the failure is logged and the object's labels are
   * resolved without table data instead, until its data, one of its
   * configurations or one of its maps changes.
   *
   * @param labels - The labels object to resolve
   * @param data - The loaded data of the labels object
   * @param context - The inputs of the current synchronization
   * @param options - Optional abort signal
   * @returns A promise that resolves once the data transfer has been resolved
   */
  protected override async resolveObject(
    labels: Labels,
    data: LabelsData,
    context: OpenSeadragonLabelsSyncContext,
    options?: { signal?: AbortSignal },
  ): Promise<void> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const renderedLabels = this._renderedLabels.get(labels.id);
    const state = {
      labelColor: structuredClone(labels.labelColor),
      labelVisibility: structuredClone(labels.labelVisibility),
      labelOpacity: structuredClone(labels.labelOpacity),
      labelColorMap: ConfigUtils.findGroupByMap(
        labels.labelColor,
        context.colorMaps,
      ),
      labelVisibilityMap: ConfigUtils.findGroupByMap(
        labels.labelVisibility,
        context.visibilityMaps,
      ),
      labelOpacityMap: ConfigUtils.findGroupByMap(
        labels.labelOpacity,
        context.opacityMaps,
      ),
    };
    if (
      renderedLabels === undefined ||
      renderedLabels.data !== data ||
      !deepEqual(renderedLabels.state.labelColor, state.labelColor) ||
      renderedLabels.state.labelColorMap !== state.labelColorMap ||
      !deepEqual(renderedLabels.state.labelVisibility, state.labelVisibility) ||
      renderedLabels.state.labelVisibilityMap !== state.labelVisibilityMap ||
      !deepEqual(renderedLabels.state.labelOpacity, state.labelOpacity) ||
      renderedLabels.state.labelOpacityMap !== state.labelOpacityMap
    ) {
      const dataTransfer =
        await OpenSeadragonLabelsRenderer._resolveDataTransfer(
          labels,
          data,
          context,
          options,
        );
      this._renderedLabels.set(labels.id, { data, state, dataTransfer });
    }
  }

  /**
   * Returns the data transfer resolved for the given labels object
   *
   * A labels object has a single tiled image, whose tiles carry the label IDs,
   * and no backdrop, so the index is not looked at.
   *
   * @param ref - The labels reference for which to get the data transfer
   * @returns The data transfer resolved by {@link resolveObject}, or
   * `undefined` if the object has not been resolved
   */
  protected override resolveTiledImageDataTransfer(
    ref: ObjectRef<Labels, LabelsData>,
  ): DataTransfer | undefined {
    const renderedLabels = this._renderedLabels.get(ref.object.id);
    if (renderedLabels !== undefined) {
      return renderedLabels.dataTransfer;
    }
    return undefined;
  }

  /**
   * Resolves the data transfer of a labels object
   *
   * If a label color, visibility or opacity configuration reads from the
   * referenced table, the table is loaded and the appearance of every label it
   * lists is resolved up front, in the same way as the WebGL renderers resolve
   * the appearance of their items, into a lookup table from label ID to packed
   * pixel. Every other label - those the table does not list, and all labels
   * if no configuration needs the table - is resolved as it is first drawn
   * (see {@link _createDataTransfer}).
   *
   * If the table cannot be loaded or a configuration cannot be resolved from
   * it, the failure is logged and the lookup table is left empty, so that all
   * labels are resolved without table data.
   *
   * @param labels - The labels object to resolve
   * @param data - The loaded data of the labels object
   * @param context - The inputs of the current synchronization
   * @param options - Optional abort signal
   * @returns The resolved data transfer
   */
  private static async _resolveDataTransfer(
    labels: Labels,
    data: LabelsData,
    context: OpenSeadragonLabelsSyncContext,
    options?: { signal?: AbortSignal },
  ): Promise<DataTransfer> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const labelPixelValues = new Map<number, number>();
    const requiresTable = [
      labels.labelColor,
      labels.labelVisibility,
      labels.labelOpacity,
    ].some((config) => {
      const activeConfigSource = getActiveConfigSource(config);
      return activeConfigSource === "from" || activeConfigSource === "groupBy";
    });
    if (requiresTable) {
      const table = context.tables.find(
        (table) => table.id === labels.dataSource.table,
      );
      if (labels.dataSource.table === undefined) {
        console.warn(
          `Labels ${labels.id} has no table, but a configuration requires one`,
        );
      } else if (table === undefined) {
        console.warn(`Table with ID ${labels.dataSource.table} not found`);
      } else {
        try {
          const tableData = await context.loadTable(table, { signal });
          const labelIds = tableData.getIds();
          if (Array.isArray(labelIds)) {
            throw new Error("String IDs cannot address label values");
          }
          const loadTable = () => Promise.resolve(tableData);
          const [
            packedLabelColors,
            packedLabelVisibilities,
            packedLabelOpacities,
          ] = await Promise.all([
            ColorResolver.resolveColors(
              labelIds,
              labels.labelColor,
              context.colorMaps,
              defaultLabelColor,
              { signal, loadTable },
            ),
            VisibilityResolver.resolveVisibilities(
              labelIds,
              labels.labelVisibility,
              context.visibilityMaps,
              defaultLabelVisibility,
              { signal, loadTable },
            ),
            OpacityResolver.resolveOpacities(
              labelIds,
              labels.labelOpacity,
              context.opacityMaps,
              defaultLabelOpacity,
              { signal, loadTable },
            ),
          ]);
          await AsyncUtils.forEach(
            labelIds,
            (labelId, i) => {
              labelPixelValues.set(
                labelId,
                ColorUtils.withAlpha(
                  packedLabelColors[i]!,
                  packedLabelVisibilities[i]!,
                  packedLabelOpacities[i]!,
                ),
              );
            },
            { signal },
          );
        } catch (error) {
          signal?.throwIfAborted();
          console.warn(
            `Failed to resolve labels ${labels.id} from table ${labels.dataSource.table}, resolving without table`,
            error,
          );
          labelPixelValues.clear();
        }
      }
    }
    return OpenSeadragonLabelsRenderer._createDataTransfer(
      labels,
      data,
      labelPixelValues,
    );
  }

  /**
   * Creates the data transfer of a labels object from a lookup table
   *
   * Label value `0` is background and maps to a fully transparent pixel. Labels
   * that the lookup table does not list are resolved without table data as
   * they are first drawn (see e.g. {@link ColorResolver.resolveColorWithoutTable}):
   * constant values and random colors resolve exactly, whereas configurations
   * that need the table fall back to the default label color, visibility and
   * opacity. The result is memoized in the lookup table, so that each label is
   * resolved once.
   *
   * @param labels - The labels object whose configurations unlisted labels are resolved from
   * @param data - The loaded data of the labels object
   * @param labelPixelValues - The lookup table from label ID to packed pixel,
   * extended as labels are drawn
   * @returns The data transfer
   */
  private static _createDataTransfer(
    labels: Labels,
    data: LabelsData,
    labelPixelValues: Map<number, number>,
  ): DataTransfer {
    return {
      getTileData: (event) => data.getTileData(event),
      transferValues: (values, pixelBuffer) => {
        for (let i = 0; i < values.length; i++) {
          const labelId = values[i]!;
          if (labelId === 0) {
            pixelBuffer[i] = 0;
          } else {
            let labelPixelValue = labelPixelValues.get(labelId);
            if (labelPixelValue === undefined) {
              labelPixelValue = ColorUtils.withAlpha(
                ColorResolver.resolveColorWithoutTable(
                  labelId,
                  labels.labelColor,
                  defaultLabelColor,
                ),
                VisibilityResolver.resolveVisibilityWithoutTable(
                  labelId,
                  labels.labelVisibility,
                  defaultLabelVisibility,
                ),
                OpacityResolver.resolveOpacityWithoutTable(
                  labelId,
                  labels.labelOpacity,
                  defaultLabelOpacity,
                ),
              );
              labelPixelValues.set(labelId, labelPixelValue);
            }
            pixelBuffer[i] = labelPixelValue;
          }
        }
      },
    };
  }
}
