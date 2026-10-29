import {
  type AppStore,
  type Color,
  type ColorConfig,
  type DataStoreApi,
  type GroupValueMap,
  type Image,
  ImageChannelViewMode,
  type ImageData,
  ImageUtils,
  type Layer,
  type ProjectStore,
  type Shapes,
  type SizeConfig,
  type TableColumnRef,
  type VisibilityConfig,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
  isRandomConfig,
} from "@tissuumaps/core";

import { loadedImageData, shownChannels } from "./action-checks";
import { colorName, groupPalettes, numberPalettes } from "./colors";
import { personaRules } from "./instructions";
import {
  annotatingTable,
  colorConfigOf,
  columnRefOf,
  groupByColumnOf,
  list,
  positionColumnsOf,
  round,
  sameItemsAs,
  visibilityConfigOf,
} from "./objects";
import type { Persona } from "./persona";
import { clusterColumn, objectTables } from "./tables";
import type {
  ItemsKind,
  ItemsObject,
  TableInfo,
  TableKnowledge,
} from "./types";

/**
 * Names a table column, e.g. 'column "cell_type" of table tables/table'
 *
 * @param ref - The column
 * @returns The text
 */
function describeColumnRef(ref: TableColumnRef): string {
  return `column "${ref.column}"${ref.table === undefined ? "" : ` of table ${ref.table}`}`;
}

/**
 * Describes a color map with its first colors, e.g. 'color map "Chat colors –
 * cell_labels": 3 red, others light grey'
 *
 * @param map - The color map
 * @returns The text
 */
function describeColorMap(map: GroupValueMap<Color>): string {
  const groups = Object.entries(map.values);
  const colors = groups
    .slice(0, 10)
    .map(([group, color]) => `${group} ${colorName(color)}`);
  if (groups.length > 10) {
    colors.push(`... (${groups.length} groups)`);
  }
  if (map.default !== undefined) {
    colors.push(`others ${colorName(map.default)}`);
  }
  return `color map "${map.name}": ${colors.join(", ") || "no colors yet"}`;
}

/**
 * Describes a color setting in a few words, e.g. "fixed red", 'by groups of
 * column "cell_type", palette distinct' or 'from the numbers in column "area",
 * palette viridis, range 0 to 500'
 *
 * @param config - The color setting
 * @param colorMaps - The project's color maps
 * @returns The text
 */
function describeColor(
  config: ColorConfig,
  colorMaps: GroupValueMap<Color>[],
): string {
  const source = getActiveConfigSource(config);
  if (source === "constant" && isConstantConfig<Color>(config)) {
    return `fixed ${colorName(config.constant.value)}`;
  }
  if (
    source === "from" &&
    isFromConfig<{ range?: [number, number]; palette: string }>(config)
  ) {
    // without a range, the lowest value gets the first color and the highest the last
    const range =
      config.from.range === undefined
        ? "range from the lowest to the highest value"
        : `range ${config.from.range[0]} to ${config.from.range[1]}`;
    return `from the numbers in ${describeColumnRef(config.from)}, palette ${config.from.palette}, ${range}`;
  }
  if (
    source === "groupBy" &&
    isGroupByConfig<false, { palette?: string }>(config)
  ) {
    const { map, palette } = config.groupBy;
    // what TissUUmaps does without a map or palette
    let how = "no palette or color map, so all in one color";
    if (map !== undefined) {
      const colorMap = colorMaps.find((m) => m.id === map);
      how = describeColorMap(colorMap ?? { id: map, name: map, values: {} });
    } else if (palette !== undefined) {
      how = `palette ${palette}`;
    }
    return `by groups of ${describeColumnRef(config.groupBy)}, ${how}`;
  }
  if (source === "random" && isRandomConfig<{ palette: string }>(config)) {
    return `random colors, palette ${config.random.palette}`;
  }
  return "default colors";
}

/**
 * Describes which items of labels or points are shown
 *
 * @param config - The visibility setting
 * @param visibilityMaps - The project's visibility maps
 * @returns The text, e.g. 'only the groups 1, 3 of column "leiden" are shown',
 * or `undefined` when all items are shown
 */
function describeFilter(
  config: VisibilityConfig,
  visibilityMaps: GroupValueMap<boolean>[],
): string | undefined {
  const source = getActiveConfigSource(config);
  if (source === "constant" && isConstantConfig<boolean>(config)) {
    return config.constant.value ? undefined : "all items hidden";
  }
  if (source === "from" && isFromConfig(config)) {
    return `only the items where ${describeColumnRef(config.from)} is not 0 are shown`;
  }
  if (!(source === "groupBy" && isGroupByConfig<true>(config))) {
    return undefined;
  }
  const map = visibilityMaps.find((m) => m.id === config.groupBy.map);
  if (map === undefined) {
    return `by groups of ${describeColumnRef(config.groupBy)}`;
  }
  const groups = Object.entries(map.values);
  const shown = groups.filter(([, visible]) => visible).map(([group]) => group);
  const hidden = groups
    .filter(([, visible]) => !visible)
    .map(([group]) => group);
  return map.default === false
    ? `only the groups ${list(shown)} of ${describeColumnRef(config.groupBy)} are shown (show_all_groups removes this filter)`
    : `the groups ${list(hidden)} of ${describeColumnRef(config.groupBy)} are hidden (show_all_groups removes this filter)`;
}

/**
 * Describes the base point size in a few words
 *
 * @param pointSize - The size setting
 * @returns The text
 */
function describePointSize(pointSize: SizeConfig): string {
  const source = getActiveConfigSource(pointSize);
  if (source === "constant" && isConstantConfig<number>(pointSize)) {
    return `${pointSize.constant.value}`;
  }
  const ref = columnRefOf(pointSize);
  if (ref === undefined) {
    return "default";
  }
  return source === "from"
    ? `from table column "${ref.column}"`
    : `for each group of table column "${ref.column}"`;
}

/**
 * Describes layers, images or shapes, one line each, e.g.
 * '- id: img1, name: "DAPI", visible: yes, opacity: 1'
 *
 * @param title - The section title
 * @param objects - The objects
 * @param colorMaps - The project's color maps
 * @param channelsOf - For images, the line about their channels, if any
 * @returns The text
 */
function describeObjects(
  title: string,
  objects: readonly (Layer | Image | Shapes)[],
  colorMaps: GroupValueMap<Color>[],
  channelsOf?: (image: Image) => string | undefined,
): string {
  if (objects.length === 0) {
    return `## ${title}\nnone`;
  }
  const lines = objects.map((object) => {
    let line = `- id: ${object.id}, name: "${object.name}"`;
    line += `, visible: ${object.visibility ? "yes" : "no"}`;
    line += `, opacity: ${round(object.opacity)}`;
    if ("shapeFillColor" in object) {
      line += `, fill color: ${describeColor(object.shapeFillColor, colorMaps)}`;
      line += `, outline color: ${describeColor(object.shapeStrokeColor, colorMaps)}`;
    }
    const table = annotatingTable(object);
    if (table !== undefined) {
      line += `, annotated by table: ${table}`;
    }
    const channels =
      channelsOf !== undefined && "dataSource" in object
        ? channelsOf(object as Image)
        : undefined;
    return channels === undefined ? line : `${line}\n${channels}`;
  });
  return `## ${title}\n${lines.join("\n")}`;
}

/**
 * Describes the channels of an image as the Images panel shows them, with
 * their indices, e.g. "  - 4 channels, composite (the visible channels
 * blended): 0 DAPI (visible, white), 1 ATP1A1 (hidden, red), ..."
 *
 * @param image - The image
 * @param data - The data of the image, if it is loaded
 * @returns The line, or `undefined` for an image without channels
 */
function describeChannels(
  image: Image,
  data: ImageData | undefined,
): string | undefined {
  const channels = shownChannels(image, data);
  if (channels.length === 0) {
    return undefined;
  }
  const mode = image.channelViewMode ?? ImageChannelViewMode.composite;
  const active = ImageUtils.getActiveChannel(image, channels.length);
  const how =
    mode === ImageChannelViewMode.composite
      ? "composite (the visible channels blended)"
      : `${mode} (only channel ${active}, ${channels[active]?.name ?? ""}, is shown)`;
  const shown = channels.slice(0, 30).map((channel, c) => {
    const settings = [channel.visibility ? "visible" : "hidden"];
    settings.push(colorName(channel.color));
    if (channel.opacity !== 1) {
      settings.push(`opacity ${round(channel.opacity)}`);
    }
    return `${c} ${channel.name} (${settings.join(", ")})`;
  });
  if (channels.length > 30) {
    shown.push("...");
  }
  return `  - ${channels.length} channel${channels.length === 1 ? "" : "s"}, ${how}: ${shown.join(", ")}`;
}

/**
 * The lines about the useful columns of one table of labels or points
 *
 * @param object - The labels or points
 * @param tableId - The table
 * @param info - What was found out about the table, if anything
 * @returns The lines
 */
function describeTableColumns(
  object: ItemsObject,
  tableId: string,
  info: TableInfo | undefined,
): string[] {
  const own = tableId === annotatingTable(object);
  const which = own
    ? `table ${tableId}`
    : `another table, ${tableId}, with rows for the same items`;
  if (info === undefined) {
    return [`  - columns: the table ${tableId} is not loaded yet`];
  }
  if (info.status === "reading") {
    return [`  - columns of the table ${tableId}: still being read`];
  }
  if (info.status === "error") {
    return [
      `  - columns: the table ${tableId} could not be read (${info.error})`,
    ];
  }
  const position = own ? positionColumnsOf(object) : [];
  const useful = info.useful.filter(
    (column) => !position.includes(column.name),
  );
  const other = info.other + info.useful.length - useful.length;
  const lines = [
    `  - columns you can use (${which}, ${info.rows} rows):${useful.length === 0 ? " none" : ""}`,
    ...useful.map((column) => `    - ${column.name} (${column.text})`),
  ];
  if (other > 0) {
    const why = info.tooBig ? ", not read because the table is very big" : "";
    lines.push(`  - (+ ${other} other column${other === 1 ? "" : "s"}${why})`);
  }
  return lines;
}

/**
 * The cluster columns labels or points can use, e.g. [obs/leiden, obs/kmeans]
 *
 * @param object - The labels or points
 * @param tables - What the panel knows about the tables
 * @returns The column names
 */
function clusterColumns(object: ItemsObject, tables: TableKnowledge): string[] {
  return objectTables(object, tables).flatMap((tableId) => {
    const info = tables.info.get(tableId);
    if (info?.status !== "ready") {
      return [];
    }
    return info.useful
      .filter(
        (column) =>
          column.groups !== undefined && clusterColumn.test(column.name),
      )
      .map((column) => column.name);
  });
}

/**
 * Describes labels or points, each with its colors, its filter and the columns
 * it can use, e.g.
 *
 * ```
 * - **cell_labels** (labels, hidden, opacity 1, id: labels/cell_labels@global)
 *   - color: by groups of column "leiden", palette distinct
 *   - columns you can use (table tables/table, 1000 rows):
 *     - leiden (16 groups: 0, 1, 2, ..., 15)
 *   - (+ 4 other columns)
 * ```
 *
 * @param title - The section title
 * @param kind - The section
 * @param project - The project
 * @param tables - What the panel knows about the tables
 * @returns The text
 */
function describeItemsObjects(
  title: string,
  kind: ItemsKind,
  project: ProjectStore,
  tables: TableKnowledge,
): string {
  const objects: readonly ItemsObject[] = project[kind];
  if (objects.length === 0) {
    return `## ${title}\nnone`;
  }
  const blocks = objects.map((object) => {
    let line = `- **${object.name}** (${kind}, ${object.visibility ? "visible" : "hidden"}, opacity ${round(object.opacity)}`;
    if ("pointSize" in object) {
      line += `, point size ${describePointSize(object.pointSize)}`;
      line += `, point size factor ${round(object.pointSizeFactor)}`;
    }
    line += `, id: ${object.id})`;
    const lines = [
      line,
      `  - color: ${describeColor(colorConfigOf(object), project.colorMaps)}`,
    ];
    const filter = describeFilter(
      visibilityConfigOf(object),
      project.visibilityMaps,
    );
    if (filter !== undefined) {
      lines.push(`  - filter: ${filter}`);
    }
    const others = sameItemsAs(project, kind, object);
    if (others.length > 0) {
      const names = others.map(
        (other) => `${other.name} (${other.visibility ? "visible" : "hidden"})`,
      );
      lines.push(`  - same items as: ${list(names)}, drawn in the same place`);
    }
    const tableIds = objectTables(object, tables);
    if (tableIds.length === 0) {
      lines.push("  - columns: none (no table is linked)");
    }
    for (const tableId of tableIds) {
      lines.push(
        ...describeTableColumns(object, tableId, tables.info.get(tableId)),
      );
    }
    // several cluster columns: the summary says which one "clusters" means
    const clusters = clusterColumns(object, tables);
    if (clusters.length > 1) {
      const colorColumn = groupByColumnOf(object)?.column;
      lines.push(
        colorColumn !== undefined && clusters.includes(colorColumn)
          ? `  - "clusters" means ${colorColumn} here: the colors show it now`
          : `  - cluster columns: ${list(clusters)}. If the user just says "clusters", do not call a tool: ask which one.`,
      );
    }
    return lines.join("\n");
  });
  return `## ${title}\n${blocks.join("\n")}`;
}

/**
 * The app state as JSON, for the strong models: the project store and the
 * parts of the app store that only change the view, without the actions
 *
 * The OpenSeadragon options are left out: they are long and say nothing about
 * the project.
 *
 * @param project - The project store
 * @param app - The app store
 * @returns The JSON text
 */
export function stateJson(project: ProjectStore, app: AppStore): string {
  const view = {
    interactionMode: app.interactionMode,
    imageChannelPreview: app.imageChannelPreview,
    highlightedItemGroup: app.highlightedItemGroup,
    activePanelId: app.activePanelId,
    expandedImageIds: app.expandedImageIds,
    expandedLabelsIds: app.expandedLabelsIds,
    expandedPointsIds: app.expandedPointsIds,
    expandedShapesIds: app.expandedShapesIds,
    expandedTableIds: app.expandedTableIds,
  };
  return JSON.stringify(
    { projectStore: project, appStore: view },
    (key, value: unknown) =>
      typeof value === "function" || key === "osOptions" ? undefined : value,
  );
}

/**
 * Builds what changes between the requests to the model: the personality and
 * a summary of the open project. The instructions, which never change, are
 * sent apart from it (see `providers.ts`).
 *
 * @param project - The project
 * @param dataStore - The data store, for the channels of the images
 * @param tables - What the panel knows about the tables
 * @param persona - The personality of the assistant
 * @returns The text
 */
export function buildContext(
  project: ProjectStore,
  dataStore: DataStoreApi,
  tables: TableKnowledge,
  persona: Persona,
): string {
  return [
    "# Your personality",
    persona.tone,
    personaRules,
    "",
    `# Project "${project.name}"`,
    describeObjects("Layers", project.layers, project.colorMaps),
    describeObjects("Images", project.images, project.colorMaps, (image) =>
      describeChannels(image, loadedImageData(dataStore, image)),
    ),
    describeItemsObjects("Labels", "labels", project, tables),
    describeItemsObjects("Points", "points", project, tables),
    describeObjects("Shapes", project.shapes, project.colorMaps),
    "## Color palettes in TissUUmaps",
    `- for groups: ${groupPalettes.join(", ")}`,
    `- for numbers: ${numberPalettes.join(", ")}`,
  ].join("\n");
}

/**
 * Example requests for the open project, shown above the text box; each one
 * names an object of the project and only columns and groups of that
 * object's tables
 *
 * @param project - The project
 * @param tables - What the panel knows about the tables
 * @returns Up to four requests
 */
export function exampleRequests(
  project: ProjectStore,
  tables: TableKnowledge,
): string[] {
  let clusters: string | undefined;
  let filter: string | undefined;
  let gene: string | undefined;
  const objects: ItemsObject[] = [...project.labels, ...project.points];
  for (const object of objects) {
    const groupBy = groupByColumnOf(object);
    for (const tableId of objectTables(object, tables)) {
      const info = tables.info.get(tableId);
      if (info?.status !== "ready") {
        continue;
      }
      for (const column of info.useful) {
        const short = column.name.replace(/^.*\//, "");
        if (
          clusters === undefined &&
          column.groups !== undefined &&
          clusterColumn.test(column.name)
        ) {
          clusters = `Color the ${object.name} by ${short}`;
        }
        // only with gene names: "Color ... by 0 expression" would make no sense
        const [firstGene] = column.genes ?? [];
        if (gene === undefined && firstGene !== undefined) {
          gene = `Color the ${object.name} by ${firstGene} expression`;
        }
        // many groups that give the colors now, e.g. a color for each gene of
        // the transcripts: only the groups of exactly that column
        const colorsFromHere =
          groupBy?.column === column.name &&
          (groupBy.table ?? annotatingTable(object)) === tableId;
        const [first, second] = column.groups ?? [];
        if (
          filter === undefined &&
          column.many === true &&
          colorsFromHere &&
          second !== undefined
        ) {
          filter = `Show only ${first} and ${second} in ${object.name}`;
        }
      }
    }
  }
  const [image] = project.images;
  return [
    clusters,
    filter,
    gene,
    "What is shown now?",
    image === undefined ? undefined : `Make the ${image.name} 50% transparent`,
    project.points.length > 0 ? "Hide all points" : undefined,
  ]
    .filter((example) => example !== undefined)
    .slice(0, 4);
}
