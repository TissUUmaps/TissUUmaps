import {
  type AppStore,
  type Color,
  type GroupValueMap,
  type Image,
  ImageChannelViewMode,
  type Layer,
  type Marker,
  type ProjectStore,
} from "@tissuumaps/core";

import {
  checkHighlight,
  checkMap,
  checkUpdates,
  checkValue,
  loadedImageData,
  shownChannels,
} from "./action-checks";
import {
  allowedActionNames,
  mapLists,
  moveActions,
  projectActions,
  refusedActions,
  updateActions,
  viewActions,
} from "./actions";
import { colorName, isRgb } from "./colors";
import {
  change,
  findById,
  isRecord,
  list,
  objectKinds,
  round,
  same,
} from "./objects";
import type {
  ListKind,
  MapKind,
  ObjectKind,
  ProjectObject,
  ToolArguments,
  ToolContext,
} from "./types";

/** The tool that calls an action, only offered to OpenAI and Anthropic */
export const callActionDefinition = {
  type: "function",
  function: {
    name: "call_action",
    description:
      "Call an action of the app state (see its types) for a change that the other tools cannot make, " +
      "e.g. the color of an image channel, the marker of points, a transform, the order of objects. " +
      `Allowed actions: ${allowedActionNames.join(", ")}.`,
    parameters: {
      type: "object",
      required: ["store", "action", "args"],
      properties: {
        store: {
          type: "string",
          enum: ["projectStore", "appStore"],
          description:
            "The store of the action: appStore for setHighlightedItemGroup and setExpanded*.",
        },
        action: {
          type: "string",
          description: 'The name of the action, e.g. "updateImage".',
        },
        args: {
          type: "array",
          items: {},
          description:
            'The arguments of the action, in order, e.g. ["images/he_image@global", {"transform": {"rotation": 90}}]. ' +
            "For transform and channels, give only what changes; channels is a list with one entry per channel, " +
            '{} for the channels that stay. Colors are {"r":0-255,"g":0-255,"b":0-255}. Use ids from the app state.',
        },
      },
    },
  },
};

/** Moves an entry of a list to another index, with its project store action */
export function applyMove(
  project: ProjectStore,
  kind: ObjectKind | "layers",
  id: string,
  index: number,
): void {
  switch (kind) {
    case "layers":
      project.moveLayer(id, index);
      break;
    case "images":
      project.moveImage(id, index);
      break;
    case "labels":
      project.moveLabels(id, index);
      break;
    case "points":
      project.movePoints(id, index);
      break;
    case "shapes":
      project.moveShapes(id, index);
      break;
  }
}

/** Adds a map, with its project store action */
function applyAddMap(
  project: ProjectStore,
  kind: MapKind,
  map: GroupValueMap<unknown>,
): void {
  switch (kind) {
    case "markerMaps":
      project.addMarkerMap(map as GroupValueMap<Marker>);
      break;
    case "sizeMaps":
      project.addSizeMap(map as GroupValueMap<number>);
      break;
    case "colorMaps":
      project.addColorMap(map as GroupValueMap<Color>);
      break;
    case "visibilityMaps":
      project.addVisibilityMap(map as GroupValueMap<boolean>);
      break;
    case "opacityMaps":
      project.addOpacityMap(map as GroupValueMap<number>);
      break;
  }
}

/** Deletes a map, with its project store action, for "Undo" */
export function deleteMap(
  project: ProjectStore,
  kind: MapKind,
  id: string,
): void {
  switch (kind) {
    case "markerMaps":
      project.deleteMarkerMap(id);
      break;
    case "sizeMaps":
      project.deleteSizeMap(id);
      break;
    case "colorMaps":
      project.deleteColorMap(id);
      break;
    case "visibilityMaps":
      project.deleteVisibilityMap(id);
      break;
    case "opacityMaps":
      project.deleteOpacityMap(id);
      break;
  }
}

/** Sets a setting of the project itself, with its project store action */
export function setProjectSetting(
  project: ProjectStore,
  property: string,
  value: unknown,
): void {
  if (property === "name") {
    project.setName(value as string);
  } else if (property === "viewerBackgroundColor") {
    project.setViewerBackgroundColor(value as Color);
  }
}

/** Sets a setting of the view, with its app store action */
export function setView(app: AppStore, setting: string, value: unknown): void {
  switch (setting) {
    case "highlightedItemGroup":
      app.setHighlightedItemGroup(
        value as Parameters<AppStore["setHighlightedItemGroup"]>[0],
      );
      break;
    case "expandedImageIds":
      app.setExpandedImageIds(value as string[]);
      break;
    case "expandedLabelsIds":
      app.setExpandedLabelsIds(value as string[]);
      break;
    case "expandedPointsIds":
      app.setExpandedPointsIds(value as string[]);
      break;
    case "expandedShapesIds":
      app.setExpandedShapesIds(value as string[]);
      break;
    case "expandedTableIds":
      app.setExpandedTableIds(value as string[]);
      break;
  }
}

/**
 * The name of an entry of the project by its ID, to show actions readably
 *
 * @param project - The project
 * @param id - The ID
 * @returns The name, or `undefined` if no entry has that ID
 */
function nameById(project: ProjectStore, id: string): string | undefined {
  const lists: ListKind[] = [
    ...objectKinds,
    "layers",
    ...(Object.keys(mapLists) as MapKind[]),
  ];
  for (const kind of lists) {
    const entry = findById(project, kind, id);
    if (entry !== undefined) {
      return entry.name;
    }
  }
  return project.tables.find((table) => table.id === id)?.name;
}

/** The parts of a configuration, e.g. the "groupBy" of {"groupBy": ...} */
const configParts = ["source", "constant", "from", "groupBy", "random"];

/**
 * Whether a value is shown whole rather than part by part: a color, or a
 * configuration like {"groupBy": ...}, but not an object that only has a
 * part with such a name, like the group of setHighlightedItemGroup
 *
 * @param value - The value
 * @returns Whether it is shown whole
 */
function isWhole(value: unknown): boolean {
  return (
    isRgb(value) ||
    (isRecord(value) &&
      Object.keys(value).length > 0 &&
      Object.keys(value).every((key) => configParts.includes(key)))
  );
}

/**
 * The parts of a value that are set, one per line of the Tool row: e.g.
 * "channels[1].color: {...}"; empty entries ({} for a channel that stays)
 * are left out, colors and configurations are shown whole, and the IDs of
 * entries of the project by their names
 *
 * @param value - The value
 * @param path - Where it is
 * @param name - The name of an entry by its ID, if it has one
 * @returns The parts
 */
function setParts(
  value: unknown,
  path: string,
  name: (id: string) => string | undefined,
): string[] {
  if (Array.isArray(value) && value.some(isRecord)) {
    return value.flatMap((entry, index) =>
      setParts(entry, `${path}[${index}]`, name),
    );
  }
  if (isRecord(value) && !isWhole(value)) {
    return Object.entries(value).flatMap(([key, entry]) =>
      setParts(entry, path === "" ? key : `${path}.${key}`, name),
    );
  }
  const show = (entry: unknown) =>
    (typeof entry === "string" ? name(entry) : undefined) ??
    JSON.stringify(entry);
  const shown =
    Array.isArray(value) && value.every((entry) => typeof entry === "string")
      ? `[${value.map(show).join(", ")}]`
      : show(value);
  return [path === "" ? shown : `${path}: ${shown}`];
}

/**
 * Shows an action readably, with names instead of IDs, e.g.
 * 'updateImage(morphology_focus, {channels[1].color: {"r":255,"g":0,"b":0}})'
 * or 'setHighlightedItemGroup({annotatedObject.labelsId: cell_labels,
 * groupBy.column: "obs/leiden", group: "3"})'
 *
 * @param project - The project
 * @param args - The arguments of call_action
 * @returns The text
 */
export function describeAction(
  project: ProjectStore,
  args: ToolArguments,
): string {
  const actionArgs = Array.isArray(args.args) ? (args.args as unknown[]) : [];
  const name = (id: string) => nameById(project, id);
  const parts = actionArgs.map((arg) => {
    const shown = setParts(arg, "", name);
    return isRecord(arg) && !isWhole(arg)
      ? `{${shown.join(", ")}}`
      : shown.join(", ");
  });
  return `${String(args.action)}(${parts.join(", ")})`;
}

/** What an action changed, and what it left as it already was */
type Outcome = { changed: string[]; already: string[] };

/**
 * Shows a value in a tool result, e.g. red, 0.5, "DAPI" or true
 *
 * @param value - The value
 * @returns The text, cut after 150 characters
 */
function showValue(value: unknown): string {
  if (isRgb(value)) {
    return colorName(value);
  }
  if (typeof value === "number") {
    return String(round(value));
  }
  if (value === undefined) {
    return "not set";
  }
  const text = JSON.stringify(value);
  return text.length > 150 ? `${text.slice(0, 150)}...` : text;
}

/**
 * Compares what an action sets with the values from before, part by part like
 * {@link setParts}, e.g. "transform.rotation 0 → 90", or
 * "transform.rotation = 90" when it already was 90
 *
 * @param before - The value from before
 * @param after - What the action sets
 * @param path - Where it is
 * @param outcome - What changed and what did not, which this adds to
 */
function compareParts(
  before: unknown,
  after: unknown,
  path: string,
  outcome: Outcome,
): void {
  if (Array.isArray(after) && after.some(isRecord)) {
    const old: unknown[] = Array.isArray(before) ? before : [];
    after.forEach((entry: unknown, index) => {
      compareParts(old[index], entry, `${path}[${index}]`, outcome);
    });
    return;
  }
  if (isRecord(after) && !isWhole(after)) {
    for (const [key, entry] of Object.entries(after)) {
      const old = isRecord(before) ? before[key] : undefined;
      compareParts(old, entry, path === "" ? key : `${path}.${key}`, outcome);
    }
    return;
  }
  if (same(before, after)) {
    outcome.already.push(`${path} = ${showValue(after)}`);
  } else {
    outcome.changed.push(`${path} ${showValue(before)} → ${showValue(after)}`);
  }
}

/**
 * The result of an action for the model, which lists exactly what changed and
 * what already was so, so that the model cannot claim more, e.g. "Done.
 * Changed in he_image: opacity 1 → 0.5. Already so: visibility = true."
 *
 * @param name - What the action changed, e.g. the name of an object
 * @param outcome - What changed and what did not
 * @returns The text
 */
function outcomeText(name: string, outcome: Outcome): string {
  const already = outcome.already.join("; ");
  if (outcome.changed.length === 0) {
    return already === ""
      ? `Nothing changed: no change for ${name} was given.`
      : `Nothing changed: in ${name} already ${already}.`;
  }
  return (
    `Done. Changed in ${name}: ${outcome.changed.join("; ")}.` +
    (already === "" ? "" : ` Already so: ${already}.`)
  );
}

/**
 * Describes a highlighted group, e.g. 'group "3" of column "obs/leiden" of
 * cell_labels', or "none"
 *
 * @param project - The project
 * @param value - The group, as setHighlightedItemGroup takes it
 * @returns The text
 */
function describeHighlight(project: ProjectStore, value: unknown): string {
  if (!isRecord(value)) {
    return "none";
  }
  const id = isRecord(value.annotatedObject)
    ? Object.values(value.annotatedObject)[0]
    : undefined;
  const object =
    typeof id === "string" ? (nameById(project, id) ?? id) : "an object";
  const column = isRecord(value.groupBy) ? value.groupBy.column : undefined;
  return `group ${showValue(value.group)} of column ${showValue(column)} of ${object}`;
}

/**
 * The tool call_action: calls one allowed action of the app state, after
 * checking all its arguments, through {@link change} and the like, so that
 * "Undo" takes it back
 */
export async function callAction(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { action, store } = args;
  const actionArgs = Array.isArray(args.args) ? (args.args as unknown[]) : [];
  if (typeof action !== "string" || action === "") {
    return 'Error: give action, e.g. "updateImage".';
  }
  const refusal = refusedActions.find(([pattern]) => pattern.test(action));
  if (refusal !== undefined) {
    return `Error: ${action} is not allowed: ${refusal[1]}. Nothing changed.`;
  }
  const mapAction = /^(add|update)(\w+Map)$/.exec(action);
  const mapKind = (Object.keys(mapLists) as MapKind[]).find(
    (kind) => mapLists[kind].name === mapAction?.[2],
  );
  if (
    !allowedActionNames.includes(action) ||
    (mapAction !== null && mapKind === undefined)
  ) {
    return `Error: there is no action "${action}" that you can call. You can call: ${allowedActionNames.join(", ")}.`;
  }
  const storeName = action in viewActions ? "appStore" : "projectStore";
  if (store !== storeName) {
    return `Error: ${action} is an action of ${storeName}: call it with store "${storeName}".`;
  }
  const project = context.projectStore.getState();
  const { changes } = context;
  const [first, second] = actionArgs;

  const update = updateActions[action];
  if (update !== undefined) {
    const object =
      typeof first === "string"
        ? (findById(project, update.kind, first) as
            ProjectObject | Layer | undefined)
        : undefined;
    if (object === undefined) {
      return `Error: there is no entry of ${update.kind} with id ${JSON.stringify(first)}. Use an id from the app state.`;
    }
    if (!isRecord(second) || Object.keys(second).length === 0) {
      return 'Error: give the changes as the second argument, e.g. {"opacity": 0.5}.';
    }
    const { problems, values } = await checkUpdates(
      update.type,
      object,
      second,
      context,
      options,
    );
    if (problems.length > 0) {
      return `Error: nothing changed: ${problems.join("; ")}.`;
    }
    const current = object as unknown as Record<string, unknown>;
    const image = update.kind === "images" ? (object as Image) : undefined;
    const channels =
      image === undefined
        ? []
        : shownChannels(image, loadedImageData(context.dataStore, image));
    const outcome: Outcome = { changed: [], already: [] };
    for (const [key, value] of Object.entries(values)) {
      const part: Outcome = { changed: [], already: [] };
      if (key === "channels" && Array.isArray(second.channels)) {
        // each channel as the panel shows it, with its name
        second.channels.forEach((entry: unknown, c) => {
          const name = channels[c]?.name ?? `Channel ${c}`;
          compareParts(channels[c], entry, `channels[${c}] (${name})`, part);
        });
      } else {
        // the transform as the model gave it: only what changes
        const after = key === "transform" ? second.transform : value;
        compareParts(current[key], after, key, part);
      }
      // only what changes is changed, and taken back by "Undo"
      if (part.changed.length > 0) {
        change(project, changes, update.kind, object.id, key, value);
      }
      outcome.changed.push(...part.changed);
      outcome.already.push(...part.already);
    }
    let notes = "";
    const aboutChannels = ["channels", "channelViewMode", "activeChannel"].some(
      (key) => key in values,
    );
    if (outcome.changed.length > 0 && image !== undefined && aboutChannels) {
      // checked above: a view mode and a channel number
      const mode = (values.channelViewMode ?? image.channelViewMode) as
        string | undefined;
      const active = Number(values.activeChannel ?? image.activeChannel ?? 0);
      if (mode !== undefined && mode !== ImageChannelViewMode.composite) {
        const name = channels[active]?.name ?? `Channel ${active}`;
        notes += ` In the ${mode} view mode, only channel ${active} (${name}) is shown, whatever the visibility of the channels.`;
      }
    }
    if (
      outcome.changed.length > 0 &&
      !object.visibility &&
      values.visibility !== true
    ) {
      notes += ` ${object.name} is hidden: show it to see the change.`;
    }
    return `${outcomeText(object.name, outcome)}${notes}`;
  }

  if (mapKind !== undefined && mapAction?.[1] === "add") {
    if (!isRecord(first) || typeof first.id !== "string" || first.id === "") {
      return 'Error: give the map as the argument: {"id": ..., "name": ..., "values": {...}}.';
    }
    if (findById(project, mapKind, first.id) !== undefined) {
      return `Error: a map of ${mapKind} with id "${first.id}" already exists: call update${mapLists[mapKind].name}.`;
    }
    // name and values are checked also when they are missing
    const problem = checkMap(mapKind, {
      ...first,
      name: first.name,
      values: first.values,
    });
    if (problem !== "") {
      return `Error: nothing changed: ${problem}.`;
    }
    const map = first as GroupValueMap<unknown>;
    applyAddMap(project, mapKind, map);
    changes.push({
      type: "added",
      kind: mapKind,
      id: map.id,
      name: map.name,
      property: "",
      before: undefined,
      after: map,
    });
    const groups = Object.keys(map.values);
    return `Done. Added to ${mapKind} the map "${map.name}" (id ${map.id}), with ${groups.length} group${groups.length === 1 ? "" : "s"}: ${list(groups)}.`;
  }

  if (mapKind !== undefined) {
    const map =
      typeof first === "string" ? findById(project, mapKind, first) : undefined;
    if (map === undefined) {
      return `Error: there is no map of ${mapKind} with id ${JSON.stringify(first)}.`;
    }
    if (!isRecord(second) || Object.keys(second).length === 0) {
      return 'Error: give the changes as the second argument, e.g. {"values": {...}}.';
    }
    const problem = checkMap(mapKind, second);
    if (problem !== "") {
      return `Error: nothing changed: ${problem}.`;
    }
    const current = map as Record<string, unknown>;
    const outcome: Outcome = { changed: [], already: [] };
    for (const [key, value] of Object.entries(second)) {
      const part: Outcome = { changed: [], already: [] };
      compareParts(current[key], value, key, part);
      // new values replace the old ones: the groups left out are removed
      if (key === "values" && isRecord(value) && isRecord(current.values)) {
        for (const [group, old] of Object.entries(current.values)) {
          if (!(group in value)) {
            part.changed.push(`values.${group} ${showValue(old)} → not set`);
          }
        }
      }
      if (part.changed.length > 0) {
        change(project, changes, mapKind, map.id, key, value);
      }
      outcome.changed.push(...part.changed);
      outcome.already.push(...part.already);
    }
    return outcomeText(`the map "${map.name}"`, outcome);
  }

  const move = moveActions[action];
  if (move !== undefined) {
    const list: readonly { id: string; name: string }[] = project[move];
    const from = list.findIndex((entry) => entry.id === first);
    const entry = list[from];
    if (entry === undefined) {
      return `Error: there is no entry of ${move} with id ${JSON.stringify(first)}.`;
    }
    if (
      typeof second !== "number" ||
      !Number.isInteger(second) ||
      second < 0 ||
      second >= list.length
    ) {
      return `Error: the new index must be a number from 0 to ${list.length - 1}.`;
    }
    if (second === from) {
      return `Nothing changed: ${entry.name} is already at index ${from}.`;
    }
    applyMove(project, move, entry.id, second);
    changes.push({
      type: "moved",
      kind: move,
      id: entry.id,
      name: entry.name,
      property: "index",
      before: from,
      after: second,
    });
    return `Done. Moved ${entry.name} from index ${from} to ${second}.`;
  }

  const projectSetting = projectActions[action];
  if (projectSetting !== undefined) {
    const problem =
      projectSetting === "name"
        ? typeof first === "string" && first.trim() !== ""
          ? ""
          : "the name must be a text that is not empty"
        : checkValue("color", first, "the color");
    if (problem !== "") {
      return `Error: nothing changed: ${problem}.`;
    }
    const before = project[projectSetting];
    if (same(before, first)) {
      return `Nothing changed: the project ${projectSetting} already is ${showValue(first)}.`;
    }
    changes.push({
      kind: "project",
      id: "",
      name: "the project",
      property: projectSetting,
      before,
      after: first,
    });
    setProjectSetting(project, projectSetting, first);
    return `Done. Changed in the project: ${projectSetting} ${showValue(before)} → ${showValue(first)}.`;
  }

  const view = viewActions[action];
  if (view !== undefined) {
    const app = context.appStore.getState();
    let problem = "";
    if (view.list !== undefined) {
      const ids: readonly { id: string }[] = project[view.list];
      problem =
        Array.isArray(first) &&
        first.every((id) => ids.some((entry) => entry.id === id))
          ? ""
          : `give a list of ids of ${view.list}`;
    } else if (first !== null) {
      problem = await checkHighlight(project, context, first, options);
    }
    if (problem !== "") {
      return `Error: nothing changed: ${problem}.`;
    }
    const before = app[view.setting];
    const show = (value: unknown): string =>
      view.list === undefined
        ? describeHighlight(project, value)
        : `[${list((value as string[]).map((id) => nameById(project, id) ?? id))}]`;
    if (same(before, first)) {
      return `Nothing changed: ${view.setting} already is ${show(first)}.`;
    }
    changes.push({
      kind: "app",
      id: "",
      name: "the view",
      property: view.setting,
      before,
      after: first,
    });
    setView(app, view.setting, first);
    return `Done. Changed in the view: ${view.setting} ${show(before)} → ${show(first)}.`;
  }
  return `Error: there is no action "${action}" that you can call.`;
}
