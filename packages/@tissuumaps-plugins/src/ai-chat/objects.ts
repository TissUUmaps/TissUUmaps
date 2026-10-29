import {
  type ColorConfig,
  type Config,
  type ProjectStore,
  type TableColumnRef,
  type VisibilityConfig,
  getActiveConfigSource,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

import type {
  Change,
  ItemsKind,
  ItemsObject,
  ItemsTarget,
  ListKind,
  MapKind,
  ObjectKind,
  ProjectObject,
  Target,
  ToolArguments,
} from "./types";

/** The sections of the project, in summary order */
export const objectKinds: ObjectKind[] = [
  "images",
  "labels",
  "points",
  "shapes",
];

/** The sections whose items can be colored and filtered */
export const itemsKinds: ItemsKind[] = ["labels", "points"];

/**
 * Whether two values are the same, also for settings like `{ groupBy: ... }`,
 * which can be equal without being the same object
 *
 * @param a - The first value
 * @param b - The second value
 * @returns Whether their JSON is the same, whatever the order of the keys
 */
export function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(sortedKeys(a)) === JSON.stringify(sortedKeys(b));
}

/** A value with the keys of its objects in order, to compare it as JSON */
function sortedKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortedKeys);
  }
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, sortedKeys(value[key])]),
    );
  }
  return value;
}

/**
 * Writes a list of names for a tool result, e.g. "he_image, transcripts"
 *
 * @param names - The names
 * @returns The names separated by commas, or "none"
 */
export function list(names: string[]): string {
  return names.join(", ") || "none";
}

/**
 * Rounds a number to 3 decimals, to show it in a readable way
 *
 * @param number - The number
 * @returns The rounded number
 */
export function round(number: number): number {
  return Math.round(number * 1000) / 1000;
}

/**
 * Accepts "image" for "images", as small models often write
 *
 * @param text - The section name given by the model
 * @returns The section, or `undefined` if there is none with that name
 */
function toObjectKind(text: unknown): ObjectKind | undefined {
  return objectKinds.find(
    (kind) => kind === text || kind === `${String(text)}s`,
  );
}

/**
 * Finds an object or map of the project by its ID
 *
 * @param project - The project
 * @param kind - The section or map list the ID is in
 * @param id - The ID
 * @returns The object or map, or `undefined` if there is none
 */
export function findById(
  project: ProjectStore,
  kind: ListKind,
  id: string,
): { id: string; name: string } | undefined {
  const objects: readonly { id: string; name: string }[] = project[kind];
  return objects.find((object) => object.id === id);
}

/**
 * Finds the section of an ID, which can be in more than one; its prefix (e.g.
 * "shapes/") does not always match the section, so all sections are searched
 *
 * @param project - The project
 * @param id - The ID
 * @param type - The section the model named, if any
 * @returns The section, or an error message for the model
 */
function findKind(
  project: ProjectStore,
  id: string,
  type: unknown,
): { kind: ObjectKind } | { error: string } {
  const kinds = objectKinds.filter((kind) => findById(project, kind, id));
  const [onlyKind] = kinds;
  if (onlyKind === undefined) {
    if (project.tables.some((table) => table.id === id)) {
      return {
        error: `Error: "${id}" is a table. Tables cannot be shown, hidden or changed.`,
      };
    }
    return {
      error: `Error: there is no object with id "${id}". Use an id from the project summary.`,
    };
  }
  const kind = toObjectKind(type);
  if (kind === undefined) {
    if (kinds.length > 1) {
      return {
        error: `Error: the id "${id}" is in more than one section: ${kinds.join(", ")}. Call the tool again with type set to one of them.`,
      };
    }
    return { kind: onlyKind };
  }
  if (!kinds.includes(kind)) {
    return {
      error: `Error: the id "${id}" is not in ${kind}, it is in: ${kinds.join(", ")}.`,
    };
  }
  return { kind };
}

/**
 * Updates an object or map with its project store action
 *
 * @param project - The project
 * @param kind - The section or map list
 * @param id - The ID of the object or map
 * @param updates - The properties to change
 */
export function applyUpdate(
  project: ProjectStore,
  kind: ListKind,
  id: string,
  updates: Record<string, unknown>,
): void {
  switch (kind) {
    case "images":
      project.updateImage(id, updates);
      break;
    case "labels":
      project.updateLabels(id, updates);
      break;
    case "points":
      project.updatePoints(id, updates);
      break;
    case "shapes":
      project.updateShapes(id, updates);
      break;
    case "layers":
      project.updateLayer(id, updates);
      break;
    case "markerMaps":
      project.updateMarkerMap(id, updates);
      break;
    case "sizeMaps":
      project.updateSizeMap(id, updates);
      break;
    case "colorMaps":
      project.updateColorMap(id, updates);
      break;
    case "visibilityMaps":
      project.updateVisibilityMap(id, updates);
      break;
    case "opacityMaps":
      project.updateOpacityMap(id, updates);
      break;
  }
}

/** Whether a list is one of the lists of maps */
export function isMapKind(kind: Change["kind"]): kind is MapKind {
  return kind.endsWith("Maps");
}

/**
 * Changes one property of an object or map, and remembers the value from
 * before, so that "Undo" can put it back; all tools change the project only
 * through here
 *
 * @param project - The project
 * @param changes - The changes of the current request, which this one is added to
 * @param kind - The section or map list
 * @param id - The ID of the object or map
 * @param property - The property to change
 * @param value - The new value
 * @throws Error if there is no such object or map
 */
export function change(
  project: ProjectStore,
  changes: Change[],
  kind: ListKind,
  id: string,
  property: string,
  value: unknown,
): void {
  const object = findById(project, kind, id);
  if (object === undefined) {
    throw new Error(`There is no ${kind} with id "${id}"`);
  }
  changes.push({
    kind,
    id,
    name: object.name,
    property,
    before: (object as Record<string, unknown>)[property],
    after: value,
  });
  applyUpdate(project, kind, id, { [property]: value });
}

/**
 * Shows a hidden object whose opacity, size, colors or filter changed, since
 * the change could not be seen otherwise; "Undo" hides it again
 *
 * @param project - The project
 * @param changes - The changes of the current request
 * @param kind - The section of the object
 * @param object - The object
 * @returns Whether the object was hidden and is now shown
 */
export function showIfHidden(
  project: ProjectStore,
  changes: Change[],
  kind: ObjectKind,
  object: ProjectObject,
): boolean {
  if (object.visibility) {
    return false;
  }
  change(project, changes, kind, object.id, "visibility", true);
  return true;
}

/**
 * Finds the objects a tool should change: those in the "ids" argument, or all
 * the objects of the "category" argument; ids win over a category, since they
 * are more precise. If one ID is wrong, nothing is changed.
 *
 * @param project - The project
 * @param args - The arguments of the tool call
 * @param kinds - The sections the tool works on
 * @returns The objects, or an error message for the model
 */
export function findTargets(
  project: ProjectStore,
  args: ToolArguments,
  kinds: readonly ObjectKind[] = objectKinds,
): { targets: Target[] } | { error: string } {
  let ids = args["ids"];
  // small models send one id as text, or "id" instead of "ids"
  if (typeof ids === "string") {
    ids = [ids];
  }
  if (ids === undefined && typeof args["id"] === "string") {
    ids = [args["id"]];
  }
  const idList = Array.isArray(ids) ? ids.map(String) : [];
  if (args["category"] !== undefined && idList.length === 0) {
    const category = toObjectKind(args["category"]);
    if (category === undefined || !kinds.includes(category)) {
      return { error: `Error: category must be ${kinds.join(" or ")}.` };
    }
    const objects: readonly ProjectObject[] = project[category];
    if (objects.length === 0) {
      return { error: `Error: there are no ${category} in this project.` };
    }
    // the objects that really are in this section, whatever their IDs look
    // like (e.g. "shapes/cell_circles@global" is in points)
    return {
      targets: objects.map((object) => ({ kind: category, object })),
    };
  }
  if (idList.length === 0) {
    return { error: "Error: give ids (a list of ids) or a category." };
  }
  const targets: Target[] = [];
  for (const id of idList) {
    const found = findKind(
      project,
      id,
      kinds.length === 1 ? kinds[0] : args["type"],
    );
    if ("error" in found) {
      return { error: `${found.error} Nothing was changed.` };
    }
    if (!kinds.includes(found.kind)) {
      return {
        error: `Error: "${id}" is in ${found.kind}, but this tool only works on ${kinds.join(" and ")}. Nothing was changed.`,
      };
    }
    const object = findById(project, found.kind, id) as ProjectObject;
    if (!targets.some((t) => t.kind === found.kind && t.object.id === id)) {
      targets.push({ kind: found.kind, object });
    }
  }
  return { targets };
}

/**
 * Like {@link findTargets}, for the color and filter tools, which work on
 * labels and points only
 *
 * @param project - The project
 * @param args - The arguments of the tool call
 * @returns The objects, or an error message for the model
 */
export function findItemsTargets(
  project: ProjectStore,
  args: ToolArguments,
): { targets: ItemsTarget[] } | { error: string } {
  const found = findTargets(project, args, itemsKinds);
  return "error" in found ? found : { targets: found.targets as ItemsTarget[] };
}

/**
 * The color setting of labels or points
 *
 * @param object - The labels or points
 * @returns The color setting
 */
export function colorConfigOf(object: ItemsObject): ColorConfig {
  return "labelColor" in object ? object.labelColor : object.pointColor;
}

/**
 * The visibility setting of labels or points, which holds a group filter
 *
 * @param object - The labels or points
 * @returns The visibility setting
 */
export function visibilityConfigOf(object: ItemsObject): VisibilityConfig {
  return "labelVisibility" in object
    ? object.labelVisibility
    : object.pointVisibility;
}

/**
 * The name of the color setting of a section
 *
 * @param kind - The section
 * @returns The property name
 */
export function colorSetting(kind: ItemsKind): "labelColor" | "pointColor" {
  return kind === "labels" ? "labelColor" : "pointColor";
}

/**
 * The name of the visibility setting of a section
 *
 * @param kind - The section
 * @returns The property name
 */
export function visibilitySetting(
  kind: ItemsKind,
): "labelVisibility" | "pointVisibility" {
  return kind === "labels" ? "labelVisibility" : "pointVisibility";
}

/**
 * The table column a setting takes its values from, if any
 *
 * @param config - The setting
 * @returns The column, or `undefined` for a constant or random setting
 */
export function columnRefOf(
  config: Config<string>,
): TableColumnRef | undefined {
  const source = getActiveConfigSource(config);
  if (source === "from" && isFromConfig(config)) {
    return config.from;
  }
  if (source === "groupBy" && isGroupByConfig<false>(config)) {
    return config.groupBy;
  }
  return undefined;
}

/**
 * The column whose groups give the colors of labels or points now, if any
 *
 * @param object - The labels or points
 * @returns The column, or `undefined` if the colors do not come from groups
 */
export function groupByColumnOf(
  object: ItemsObject,
): TableColumnRef | undefined {
  const config = colorConfigOf(object);
  return getActiveConfigSource(config) === "groupBy" &&
    isGroupByConfig<false>(config)
    ? config.groupBy
    : undefined;
}

/**
 * The settings of labels or points that can take their values from a column
 *
 * @param object - The labels or points
 * @returns The settings
 */
export function columnSettingsOf(object: ItemsObject): Config<string>[] {
  return "labelColor" in object
    ? [object.labelColor, object.labelVisibility, object.labelOpacity]
    : [
        object.pointColor,
        object.pointVisibility,
        object.pointOpacity,
        object.pointSize,
        object.pointMarker,
      ];
}

/**
 * The table whose rows annotate the items of an object (e.g. one row per
 * cell), if any
 *
 * @param object - The object
 * @returns The table ID, or `undefined` if none is linked
 */
export function annotatingTable(object: object): string | undefined {
  if (!("dataSource" in object)) {
    return undefined;
  }
  const dataSource = object.dataSource;
  if (
    typeof dataSource !== "object" ||
    dataSource === null ||
    !("table" in dataSource)
  ) {
    return undefined;
  }
  return typeof dataSource.table === "string" ? dataSource.table : undefined;
}

/**
 * The x and y columns of points loaded from a table, which only tell where
 * each point is drawn
 *
 * @param object - The labels or points
 * @returns The column names, empty for other objects
 */
export function positionColumnsOf(object: ItemsObject): string[] {
  const dataSource: object & { type: string } = object.dataSource;
  if (dataSource.type !== "table") {
    return [];
  }
  const x =
    "x" in dataSource && typeof dataSource.x === "string" ? dataSource.x : "x";
  const y =
    "y" in dataSource && typeof dataSource.y === "string" ? dataSource.y : "y";
  return [x, y];
}

/**
 * Other labels or points that show the same items in the same place: the same
 * file (e.g. the same labels added twice, like "cells" and "cells-epcam"), or
 * items annotated by the same table
 *
 * @param project - The project
 * @param kind - The section of the object
 * @param object - The labels or points
 * @returns The other objects
 */
export function sameItemsAs(
  project: ProjectStore,
  kind: ItemsKind,
  object: ItemsObject,
): ItemsObject[] {
  const objects: readonly ItemsObject[] = project[kind];
  return objects.filter(
    (other) => other.id !== object.id && showSameItems(object, other),
  );
}

/**
 * Whether two labels or points show the same items in the same place
 *
 * @param a - The first object
 * @param b - The second object
 * @returns Whether they show the same items
 */
function showSameItems(a: ItemsObject, b: ItemsObject): boolean {
  if (!same(a.layer, b.layer)) {
    return false;
  }
  const { table: tableA, ...fileA } = a.dataSource as {
    table?: string;
    source?: string;
  };
  const { table: tableB, ...fileB } = b.dataSource as {
    table?: string;
    source?: string;
  };
  if (fileA.source !== undefined && same(fileA, fileB)) {
    return true;
  }
  return tableA !== undefined && tableA === tableB;
}

/** Whether a value is one of some texts */
export function isOneOf(value: unknown, texts: string[]): boolean {
  return typeof value === "string" && texts.includes(value);
}

/** Whether a value is a plain object, e.g. the changes of an update */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
