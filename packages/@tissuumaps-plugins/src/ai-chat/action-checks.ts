import {
  type Color,
  type DataStoreApi,
  type Image,
  type ImageData,
  ImageUtils,
  type Layer,
  type ProjectStore,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
  isRandomConfig,
} from "@tissuumaps/core";

import {
  type ConfigRule,
  type ValueKind,
  channelViewModes,
  configSettings,
  coordinateSpaces,
  mapLists,
  markerValues,
} from "./actions";
import { groupPalettes, isRgb, numberPalettes } from "./colors";
import { annotatingTable, findById, isOneOf, isRecord, list } from "./objects";
import { storeProperties } from "./store-types.generated";
import { listColumns, matrixColumns } from "./tables";
import type { MapKind, ObjectKind, ProjectObject, ToolContext } from "./types";

/**
 * Checks a value of a setting or a map
 *
 * @param kind - The kind of value it should be
 * @param value - The value
 * @param what - Where the value is, for the message
 * @returns What is wrong, or "" if nothing is
 */
export function checkValue(
  kind: ValueKind,
  value: unknown,
  what: string,
): string {
  switch (kind) {
    case "color":
      return isRgb(value)
        ? ""
        : `${what} must be a color {"r":0-255,"g":0-255,"b":0-255}`;
    case "boolean":
      return typeof value === "boolean" ? "" : `${what} must be true or false`;
    case "opacity":
      return typeof value === "number" && value >= 0 && value <= 1
        ? ""
        : `${what} must be a number from 0 to 1`;
    case "size":
      return typeof value === "number" && value > 0
        ? ""
        : `${what} must be a number above 0`;
    case "marker":
      return typeof value === "number" && markerValues.includes(value)
        ? ""
        : `${what} must be a marker number from 0 to 14 (e.g. 2 Square, 6 Disc)`;
  }
}

/**
 * Checks that a column of a setting exists, in the object's own table or in
 * the table the setting names
 *
 * @param context - What the tool works with
 * @param object - The object
 * @param ref - The column, as {table?, column}
 * @param what - Where the column is, for the message
 * @param options - Optional abort signal
 * @returns What is wrong, or "" if nothing is
 */
async function checkColumn(
  context: ToolContext,
  object: ProjectObject,
  ref: unknown,
  what: string,
  options?: { signal?: AbortSignal },
): Promise<string> {
  if (
    !isRecord(ref) ||
    typeof ref.column !== "string" ||
    (ref.table !== undefined && typeof ref.table !== "string")
  ) {
    return `${what} needs "column", and "table" if it is in another table`;
  }
  const column = ref.column;
  const tableId =
    typeof ref.table === "string" ? ref.table : annotatingTable(object);
  if (tableId === undefined) {
    return `${object.name} has no table of its own: give "table" in ${what}`;
  }
  const tableRef = context.dataStore.getState().tableDataRefs.get(tableId);
  if (tableRef === undefined) {
    return `the table "${tableId}" is not loaded`;
  }
  const data = await tableRef.promise;
  for (const { name, matrix } of await listColumns(data, options)) {
    if (name === column) {
      return "";
    }
    if (
      matrix &&
      column.startsWith(`${name}[`) &&
      (await matrixColumns(data, name, options)).includes(column)
    ) {
      return "";
    }
  }
  return `the table "${tableId}" has no column "${column}": give its exact name, e.g. "obs/leiden"`;
}

/**
 * Checks a setting that is a configuration, e.g. labelColor
 *
 * @param setting - The name of the setting
 * @param rule - What it can hold
 * @param config - The new configuration
 * @param object - The object
 * @param context - What the tool works with
 * @param options - Optional abort signal
 * @returns What is wrong, or "" if nothing is
 */
async function checkConfig(
  setting: string,
  rule: ConfigRule,
  config: unknown,
  object: ProjectObject,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  if (!isRecord(config)) {
    return `${setting} must be an object, e.g. {"constant":{"value":...}}`;
  }
  const sources = [
    "constant",
    "from",
    "groupBy",
    ...(rule.color === true ? ["random"] : []),
  ];
  const parts = Object.keys(config).filter((key) => key !== "source");
  const unknown = parts.filter((key) => !sources.includes(key));
  if (unknown.length > 0 || parts.length === 0) {
    return `${setting} takes one of ${list(sources)}${unknown.length > 0 ? `, not ${list(unknown)}` : ""}`;
  }
  if (config.source !== undefined && !isOneOf(config.source, parts)) {
    return `${setting}.source must name one of its parts: ${list(parts)}`;
  }
  const project = context.projectStore.getState();
  const problems: string[] = [];
  const checkUnit = (part: Record<string, unknown>, where: string) => {
    if (part.unit !== undefined && !isOneOf(part.unit, coordinateSpaces)) {
      problems.push(`${where}.unit must be one of ${list(coordinateSpaces)}`);
    }
  };
  if (isConstantConfig(config)) {
    const constant: unknown = config.constant;
    if (isRecord(constant)) {
      const problem = checkValue(
        rule.value,
        constant.value,
        `${setting}.constant.value`,
      );
      if (problem !== "") {
        problems.push(problem);
      }
      checkUnit(constant, `${setting}.constant`);
    } else {
      problems.push(`${setting}.constant must be {"value":...}`);
    }
  }
  if (isFromConfig(config)) {
    const from: unknown = config.from;
    const problem = await checkColumn(
      context,
      object,
      from,
      `${setting}.from`,
      options,
    );
    if (problem !== "") {
      problems.push(problem);
    }
    if (isRecord(from)) {
      if (rule.color === true && !isOneOf(from.palette, numberPalettes)) {
        problems.push(
          `${setting}.from.palette must be one of ${list(numberPalettes)}`,
        );
      }
      const range = from.range;
      if (
        range !== undefined &&
        !(
          Array.isArray(range) &&
          range.length === 2 &&
          typeof range[0] === "number" &&
          typeof range[1] === "number" &&
          range[0] < range[1]
        )
      ) {
        problems.push(`${setting}.from.range must be [lowest, highest]`);
      }
      checkUnit(from, `${setting}.from`);
    }
  }
  if (isGroupByConfig(config)) {
    const groupBy: unknown = config.groupBy;
    const problem = await checkColumn(
      context,
      object,
      groupBy,
      `${setting}.groupBy`,
      options,
    );
    if (problem !== "") {
      problems.push(problem);
    }
    if (isRecord(groupBy)) {
      if (groupBy.map === undefined) {
        if (rule.mapRequired) {
          problems.push(
            `${setting}.groupBy needs "map": the id of a map of ${rule.maps}`,
          );
        }
      } else if (
        typeof groupBy.map !== "string" ||
        findById(project, rule.maps, groupBy.map) === undefined
      ) {
        problems.push(
          `${setting}.groupBy.map must be the id of a map of ${rule.maps}`,
        );
      }
      if (
        groupBy.palette !== undefined &&
        !(rule.color === true && isOneOf(groupBy.palette, groupPalettes))
      ) {
        problems.push(
          `${setting}.groupBy.palette must be one of ${list(groupPalettes)}`,
        );
      }
      checkUnit(groupBy, `${setting}.groupBy`);
    }
  }
  if (isRandomConfig(config)) {
    const random: unknown = config.random;
    if (!isRecord(random) || !isOneOf(random.palette, groupPalettes)) {
      problems.push(
        `${setting}.random needs "palette", one of ${list(groupPalettes)}`,
      );
    }
  }
  return problems.join("; ");
}

/**
 * The data of an image, if it is loaded
 *
 * @param dataStore - The data store
 * @param image - The image
 * @returns The data, or `undefined` while it is not loaded
 */
export function loadedImageData(
  dataStore: DataStoreApi,
  image: Image,
): ImageData | undefined {
  const ref = dataStore.getState().imageDataRefs.get(image.id);
  return ref?.status === "loaded" ? ref.data : undefined;
}

/** A channel of an image, as the Images panel shows it */
type ShownChannel = {
  name: string;
  visibility: boolean;
  opacity: number;
  color: Color;
  contrastLimits?: [number, number];
};

/**
 * The channels of an image as the Images panel shows them: as many as its
 * data has, even when the image's channels list is shorter, each with the
 * settings of the image, or else those of its data
 *
 * @param image - The image
 * @param data - The data of the image, if it is loaded
 * @returns The channels; without the data, those the image lists
 */
export function shownChannels(
  image: Image,
  data: ImageData | undefined,
): ShownChannel[] {
  const sizeC = data?.getSizeC();
  if (data === undefined || sizeC === undefined) {
    return (image.channels ?? []).map((channel, c) => ({
      name: channel.name ?? `Channel ${c}`,
      visibility: channel.visibility ?? true,
      opacity: channel.opacity ?? 1,
      color: channel.color ?? ImageUtils.getDefaultChannelColor(c),
      contrastLimits: channel.contrastLimits,
    }));
  }
  return Array.from({ length: sizeC }, (_, c) => ({
    name:
      image.channels?.[c]?.name ?? data.getChannelName?.(c) ?? `Channel ${c}`,
    visibility: ImageUtils.getChannelVisibility(image, data, c),
    opacity: ImageUtils.getChannelOpacity(image, data, c),
    color: ImageUtils.getChannelColor(image, data, c),
    contrastLimits: ImageUtils.getChannelContrastLimits(image, data, c),
  }));
}

/**
 * Checks the channels of an image, and fills each entry in with the channel
 * as it is now
 *
 * Like the Images panel, it counts the channels of the image's data, and
 * lengthens the image's channels list when it is shorter, with {} for the
 * channels it does not list yet.
 *
 * @param image - The image
 * @param value - The channels the model gave, one entry per channel
 * @param data - The data of the image, if it is loaded
 * @returns What is wrong, or the channels as they will be
 */
function checkChannels(
  image: Image,
  value: unknown,
  data: ImageData | undefined,
): { problem: string } | { channels: unknown[] } {
  const current = image.channels ?? [];
  if (!Array.isArray(value)) {
    return {
      problem:
        "channels must be a list with one entry per channel, {} for the channels that stay",
    };
  }
  const shown = shownChannels(image, data);
  if (shown.length > 0 && value.length !== shown.length) {
    const names = shown.map(({ name }, c) => `${c} ${name}`);
    return {
      problem: `${image.name} has ${shown.length} channels (${names.join(", ")}): give ${shown.length} entries, {} for the channels that stay`,
    };
  }
  const known = storeProperties.ImageChannel ?? [];
  const problems: string[] = [];
  const length = Math.max(value.length, current.length);
  const channels = Array.from({ length }, (_, index) => {
    const entry: unknown = value[index] ?? {};
    const what = `channels[${index}]`;
    if (!isRecord(entry)) {
      problems.push(`${what} must be an object`);
      return entry;
    }
    for (const [key, setting] of Object.entries(entry)) {
      let problem = "";
      if (!known.includes(key)) {
        problem = `a channel has no "${key}"; it has ${list(known)}`;
      } else if (key === "name" && typeof setting !== "string") {
        problem = `${what}.name must be a text`;
      } else if (key === "visibility") {
        problem = checkValue("boolean", setting, `${what}.visibility`);
      } else if (key === "opacity") {
        problem = checkValue("opacity", setting, `${what}.opacity`);
      } else if (key === "color") {
        problem = checkValue("color", setting, `${what}.color`);
      } else if (
        key === "contrastLimits" &&
        !(
          Array.isArray(setting) &&
          setting.length === 2 &&
          typeof setting[0] === "number" &&
          typeof setting[1] === "number" &&
          setting[0] < setting[1]
        )
      ) {
        problem = `${what}.contrastLimits must be [lowest, highest]`;
      }
      if (problem !== "") {
        problems.push(problem);
      }
    }
    return { ...current[index], ...entry };
  });
  return problems.length > 0 ? { problem: problems.join("; ") } : { channels };
}

/**
 * Checks the changes of an update action, and fills in what the model may
 * give in part (the transform and the channels)
 *
 * @param type - The type of the entry, e.g. "Image"
 * @param object - The entry
 * @param updates - The changes the model gave
 * @param context - What the tool works with
 * @param options - Optional abort signal
 * @returns What is wrong, and the changes as they will be made
 */
export async function checkUpdates(
  type: string,
  object: ProjectObject | Layer,
  updates: Record<string, unknown>,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<{ problems: string[]; values: Record<string, unknown> }> {
  const project = context.projectStore.getState();
  const known = storeProperties[type] ?? [];
  const problems: string[] = [];
  const values: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(updates)) {
    values[key] = value;
    if (key === "id" || key === "dataSource") {
      problems.push(`${key} cannot be changed`);
      continue;
    }
    if (!known.includes(key)) {
      const changeable = known.filter(
        (name) => name !== "id" && name !== "dataSource",
      );
      problems.push(`${type} has no "${key}"; it has ${list(changeable)}`);
      continue;
    }
    const rule = configSettings[key];
    let problem = "";
    if (rule !== undefined && "dataSource" in object) {
      problem = await checkConfig(key, rule, value, object, context, options);
    } else if (key === "name") {
      problem =
        typeof value === "string" && value.trim() !== ""
          ? ""
          : "name must be a text that is not empty";
    } else if (key === "visibility") {
      problem = checkValue("boolean", value, key);
    } else if (key === "opacity") {
      problem = checkValue("opacity", value, key);
    } else if (key === "pointSizeFactor") {
      problem = checkValue("size", value, key);
    } else if (key === "layer") {
      problem =
        typeof value === "string" &&
        findById(project, "layers", value) !== undefined
          ? ""
          : `layer must be the id of a layer: ${list(project.layers.map(({ id }) => id))}`;
    } else if (key === "transform") {
      const transform = object.transform;
      if (isRecord(value)) {
        const merged = {
          ...transform,
          ...value,
          translation: {
            ...transform.translation,
            ...(isRecord(value.translation) ? value.translation : {}),
          },
        };
        values[key] = merged;
        const valid =
          typeof merged.flip === "boolean" &&
          typeof merged.scale === "number" &&
          merged.scale > 0 &&
          typeof merged.rotation === "number" &&
          Number.isFinite(merged.rotation) &&
          Number.isFinite(merged.translation.x) &&
          Number.isFinite(merged.translation.y);
        problem = valid
          ? ""
          : "transform takes flip (true or false), scale (above 0), rotation (in degrees) and translation {x, y}";
      } else {
        problem = 'transform must be an object, e.g. {"rotation": 90}';
      }
    } else if (key === "channelViewMode") {
      problem = isOneOf(value, channelViewModes)
        ? ""
        : `channelViewMode must be one of ${list(channelViewModes)}`;
    } else if (key === "activeChannel") {
      const image = object as Image;
      const count = shownChannels(
        image,
        loadedImageData(context.dataStore, image),
      ).length;
      problem =
        typeof value === "number" &&
        Number.isInteger(value) &&
        value >= 0 &&
        (count === 0 || value < count)
          ? ""
          : `activeChannel must be the number of a channel, from 0${count === 0 ? "" : ` to ${count - 1}`}`;
    } else if (key === "channels") {
      const image = object as Image;
      const checked = checkChannels(
        image,
        value,
        loadedImageData(context.dataStore, image),
      );
      if ("problem" in checked) {
        problem = checked.problem;
      } else {
        values[key] = checked.channels;
      }
    } else {
      problem = `${key} cannot be changed with call_action`;
    }
    if (problem !== "") {
      problems.push(problem);
    }
  }
  return { problems, values };
}

/**
 * Checks the values of a map
 *
 * @param kind - The list of the map
 * @param map - The map, or the changes of a map
 * @returns What is wrong, or "" if nothing is
 */
export function checkMap(kind: MapKind, map: Record<string, unknown>): string {
  const { value } = mapLists[kind];
  const problems: string[] = [];
  for (const [key, entry] of Object.entries(map)) {
    if (key === "name") {
      if (typeof entry !== "string" || entry.trim() === "") {
        problems.push("name must be a text that is not empty");
      }
    } else if (key === "values") {
      if (isRecord(entry)) {
        for (const [group, setting] of Object.entries(entry)) {
          const problem = checkValue(value, setting, `values["${group}"]`);
          if (problem !== "") {
            problems.push(problem);
          }
        }
      } else {
        problems.push(
          'values must be an object of group → value, e.g. {"3": ...}',
        );
      }
    } else if (key === "default") {
      const problem =
        entry === undefined ? "" : checkValue(value, entry, "default");
      if (problem !== "") {
        problems.push(problem);
      }
    } else if (key !== "id") {
      problems.push(
        `a map has no "${key}"; it has id, name, values and default`,
      );
    }
  }
  return problems.join("; ");
}

/**
 * Checks a group to highlight: its object, column and group
 *
 * @param project - The project
 * @param context - What the tool works with
 * @param value - The group, as setHighlightedItemGroup takes it
 * @param options - Optional abort signal
 * @returns What is wrong, or "" if nothing is
 */
export async function checkHighlight(
  project: ProjectStore,
  context: ToolContext,
  value: unknown,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const shape =
    'give {"annotatedObject": {"labelsId": ...} (or pointsId, shapesId), "groupBy": {"column": ...}, "group": "..."}, or null for none';
  if (
    !isRecord(value) ||
    !isRecord(value.annotatedObject) ||
    typeof value.group !== "string"
  ) {
    return shape;
  }
  const kinds: Record<string, ObjectKind> = {
    labelsId: "labels",
    pointsId: "points",
    shapesId: "shapes",
  };
  const [key, id] = Object.entries(value.annotatedObject)[0] ?? [];
  const kind = key === undefined ? undefined : kinds[key];
  const object =
    kind === undefined || typeof id !== "string"
      ? undefined
      : (findById(project, kind, id) as ProjectObject | undefined);
  if (object === undefined) {
    return `there is no object ${JSON.stringify(value.annotatedObject)}: ${shape}`;
  }
  return checkColumn(context, object, value.groupBy, "groupBy", options);
}
