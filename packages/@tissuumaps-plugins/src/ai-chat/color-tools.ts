import type {
  Color,
  ColorConfig,
  GroupValueMap,
  ProjectStore,
  VisibilityConfig,
} from "@tissuumaps/core";

import {
  colorName,
  colorNames,
  findPalette,
  groupPalettes,
  numberPalettes,
  othersColor,
  othersWords,
  toColor,
} from "./colors";
import {
  change,
  colorConfigOf,
  colorSetting,
  findById,
  findItemsTargets,
  groupByColumnOf,
  list,
  same,
  sameItemsAs,
  showIfHidden,
  visibilityConfigOf,
  visibilitySetting,
} from "./objects";
import { findColumn, findGroup, maxGroups, readColumn } from "./tables";
import type {
  Change,
  ItemsObject,
  ItemsTarget,
  MapKind,
  ToolArguments,
  ToolContext,
} from "./types";

/**
 * Shows a few groups, e.g. for an error: "0, 1, 2, ... (16 groups)"
 *
 * @param groups - The groups
 * @returns The text
 */
function someGroups(groups: string[]): string {
  return groups.length <= maxGroups
    ? groups.join(", ")
    : `${groups.slice(0, 10).join(", ")}, ... (${groups.length} groups)`;
}

/**
 * Sets the values of the color map or visibility map that the chat uses for
 * one object, and adds the map to the project the first time
 *
 * There is one map per object, reused every time, so that the project does
 * not fill up with maps. Changes to an existing map go through
 * {@link change}, so that "Undo" puts them back; a new map is remembered as
 * added, so that "Undo" deletes it.
 *
 * @typeParam TValue - The type of the map values
 * @param project - The project
 * @param changes - The changes of the current request
 * @param mapKind - The map list
 * @param map - The map with its new values
 */
function setChatMap<TValue extends Color | boolean>(
  project: ProjectStore,
  changes: Change[],
  mapKind: MapKind,
  map: GroupValueMap<TValue>,
): void {
  if (findById(project, mapKind, map.id) === undefined) {
    if (mapKind === "colorMaps") {
      project.addColorMap(map as GroupValueMap<Color>);
    } else {
      project.addVisibilityMap(map as GroupValueMap<boolean>);
    }
    // so that Undo deletes it again
    changes.push({
      type: "added",
      kind: mapKind,
      id: map.id,
      name: map.name,
      property: "",
      before: undefined,
      after: map,
    });
    return;
  }
  change(project, changes, mapKind, map.id, "values", map.values);
  change(project, changes, mapKind, map.id, "default", map.default);
}

/**
 * Makes a plan for each object; objects where it does not work are skipped,
 * so that e.g. category "labels" still colors the labels that have the column
 *
 * @typeParam TPlan - The type of a plan
 * @param targets - The objects
 * @param makePlan - Makes the plan for one object, or gives an error message
 * @param options - Optional abort signal
 * @returns The plans and the skipped objects, or an error message if it works
 * for none of them
 */
async function planEach<TPlan>(
  targets: ItemsTarget[],
  makePlan: (target: ItemsTarget) => Promise<TPlan | { error: string }>,
  options?: { signal?: AbortSignal },
): Promise<
  | { plans: TPlan[]; skipped: { name: string; error: string }[] }
  | { error: string }
> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const plans: TPlan[] = [];
  const skipped: { name: string; error: string }[] = [];
  for (const target of targets) {
    const plan = await makePlan(target);
    if (typeof plan === "object" && plan !== null && "error" in plan) {
      skipped.push({ name: target.object.name, error: plan.error });
    } else {
      plans.push(plan);
    }
  }
  if (plans.length === 0) {
    return {
      error: `${skipped.map((s) => s.error).join(" ")} Nothing was changed.`,
    };
  }
  return { plans, skipped };
}

/**
 * The end of a tool result about the skipped objects
 *
 * @param skipped - The skipped objects, with the reason
 * @returns The text, e.g. ' Not changed: nucleus_labels (it has no column "leiden").', or ""
 */
function skippedText(skipped: { name: string; error: string }[]): string {
  if (skipped.length === 0) {
    return "";
  }
  const reasons = skipped.map(
    (s) => `${s.name} (${s.error.replace(/^Error: /, "")})`,
  );
  return ` Not changed: ${reasons.join("; ")} Tell the user which objects were not changed and why.`;
}

/**
 * Runs the color_by_column tool: groups get a palette for groups, numbers a
 * color scale; without a palette or range, TissUUmaps' usual choice
 * ("distinct" or "viridis", and a scale from the lowest to the highest value)
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @param options - Optional abort signal
 * @returns The result for the model
 */
export async function colorByColumn(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const project = context.projectStore.getState();
  const palette = args["palette"];
  const range = args["range"];
  const validRange =
    Array.isArray(range) &&
    range.length === 2 &&
    range.every((value) => Number.isFinite(value)) &&
    Number(range[0]) < Number(range[1])
      ? ([Number(range[0]), Number(range[1])] as [number, number])
      : undefined;
  if (range !== undefined && validRange === undefined) {
    return "Error: range must be [lowest, highest], e.g. [0, 500].";
  }
  const found = findItemsTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const planned = await planEach(
    found.targets,
    async ({ kind, object }) => {
      const target = await findColumn(
        context.dataStore,
        context.tables,
        object,
        args["column"],
        { signal },
      );
      if ("error" in target) {
        return target;
      }
      const { numbers, groups } = await readColumn(target.data, target.column, {
        signal,
      });
      // a palette for numbers or a range also make few different numbers a scale
      const scale =
        groups === null ||
        (numbers &&
          (validRange !== undefined ||
            findPalette(palette, false) !== undefined));
      if (validRange !== undefined && !scale) {
        return {
          error: `Error: "${target.column}" has groups, not numbers, so it takes no range.`,
        };
      }
      const chosen =
        palette === undefined
          ? scale
            ? "viridis"
            : "distinct"
          : findPalette(palette, !scale);
      if (chosen === undefined) {
        const palettes = scale ? numberPalettes : groupPalettes;
        return {
          error: `Error: "${String(palette)}" is not a palette for ${scale ? "numbers" : "groups"}. Use one of: ${palettes.join(", ")}.`,
        };
      }
      const config: ColorConfig = scale
        ? {
            from: {
              ...target.ref,
              palette: chosen,
              ...(validRange && { range: validRange }),
            },
          }
        : { groupBy: { ...target.ref, map: undefined, palette: chosen } };
      const how = scale
        ? `the numbers in ${target.column} (palette ${chosen}, ${validRange ? `range ${validRange[0]} to ${validRange[1]}` : "from the lowest to the highest value"})`
        : `the ${groups?.length ?? 0} groups of ${target.column} (palette ${chosen})`;
      return { kind, object, config, how };
    },
    { signal },
  );
  if ("error" in planned) {
    return planned.error;
  }
  const done: string[] = [];
  const already: string[] = [];
  const shownToo: string[] = [];
  for (const { kind, object, config, how } of planned.plans) {
    if (same(colorConfigOf(object), config)) {
      already.push(`${object.name} by ${how}`);
      continue;
    }
    change(
      project,
      context.changes,
      kind,
      object.id,
      colorSetting(kind),
      config,
    );
    done.push(`${object.name} by ${how}`);
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  if (done.length === 0) {
    return (
      `Nothing changed: already colored like this: ${list(already)}. Tell the user it was already like this; do not say that you changed it.` +
      skippedText(planned.skipped)
    );
  }
  let text = `Done: now colored: ${list(done)}.`;
  if (already.length > 0) {
    text += ` Already colored like this before: ${list(already)}.`;
  }
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text + skippedText(planned.skipped);
}

/**
 * Runs the set_group_colors tool: gives chosen colors to some groups, and one
 * color to all the others (light grey if the user does not say)
 *
 * The colors go into the chat's color map of the object. Colors chosen
 * earlier for the same column are kept, so that "and cluster 5 in blue" adds
 * to "cluster 3 in red".
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @param options - Optional abort signal
 * @returns The result for the model
 */
export async function setGroupColors(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const project = context.projectStore.getState();
  const argsColors = args["colors"];
  const colors: Record<string, unknown> =
    typeof argsColors === "object" &&
    argsColors !== null &&
    !Array.isArray(argsColors)
      ? { ...argsColors }
      : {};
  let othersText = args["others"];
  // {"3": "red", "others": "grey"} means others: "grey"
  for (const word of othersWords) {
    if (word in colors) {
      othersText ??= colors[word];
      delete colors[word];
    }
  }
  if (Object.keys(colors).length === 0) {
    return 'Error: colors must give a color to each group, e.g. {"3": "red"}. To give one color to all the items, use set_color.';
  }
  const unknown = [...Object.values(colors), othersText ?? othersColor].filter(
    (color) => toColor(color) === undefined,
  );
  if (unknown.length > 0) {
    return `Error: unknown color "${String(unknown[0])}". Use one of: ${Object.keys(colorNames).join(", ")}, or "#rrggbb".`;
  }
  const others = toColor(othersText ?? othersColor) as Color;
  const found = findItemsTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const planned = await planEach(
    found.targets,
    async ({ kind, object }) => {
      const target = await findColumn(
        context.dataStore,
        context.tables,
        object,
        args["column"],
        { signal },
      );
      if ("error" in target) {
        return target;
      }
      const { groups } = await readColumn(target.data, target.column, {
        signal,
      });
      if (groups === null) {
        return {
          error: `Error: "${target.column}" has numbers, not groups. Use color_by_column for numbers.`,
        };
      }
      const mapId = `chat-colors/${kind}/${object.id}`;
      const map: GroupValueMap<Color> = {
        id: mapId,
        name: `Chat colors – ${object.name}`,
        values: {},
        default: others,
      };
      // the colors chosen earlier for the same column stay, unless the user
      // names new "others"
      const earlier = project.colorMaps.find((m) => m.id === mapId);
      const ref = { ...target.ref, map: mapId };
      if (earlier !== undefined && same(groupByColumnOf(object), ref)) {
        map.values = { ...earlier.values };
        if (othersText === undefined) {
          map.default = earlier.default;
        }
      }
      const named: string[] = [];
      for (const [wanted, colorText] of Object.entries(colors)) {
        const group = findGroup(groups, wanted);
        const color = toColor(colorText) as Color;
        if (group === undefined) {
          return {
            error: `Error: ${target.column} has no group "${wanted}". Its groups: ${someGroups(groups)}.`,
          };
        }
        map.values[group] = color;
        named.push(`${group} ${colorName(color)}`);
      }
      const config: ColorConfig = { groupBy: ref };
      const text = `${object.name} by ${target.column}: ${named.join(", ")}, all other groups ${colorName(map.default ?? others)}`;
      return { kind, object, map, config, text };
    },
    { signal },
  );
  if ("error" in planned) {
    return planned.error;
  }
  const shownToo: string[] = [];
  for (const { kind, object, map, config } of planned.plans) {
    setChatMap(project, context.changes, "colorMaps", map);
    if (!same(colorConfigOf(object), config)) {
      change(
        project,
        context.changes,
        kind,
        object.id,
        colorSetting(kind),
        config,
      );
    }
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  let text = `Done: now colored ${list(planned.plans.map((plan) => plan.text))}.`;
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text + skippedText(planned.skipped);
}

/**
 * Runs the set_color tool: gives one color to all the items of labels or points
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
export function setColor(args: ToolArguments, context: ToolContext): string {
  const project = context.projectStore.getState();
  const color = toColor(args["color"] ?? "");
  if (color === undefined) {
    return `Error: unknown color "${String(args["color"])}". Use one of: ${Object.keys(colorNames).join(", ")}, or "#rrggbb".`;
  }
  const found = findItemsTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const config: ColorConfig = { constant: { value: color } };
  const done: string[] = [];
  const already: string[] = [];
  const shownToo: string[] = [];
  for (const { kind, object } of found.targets) {
    if (same(colorConfigOf(object), config)) {
      already.push(object.name);
      continue;
    }
    change(
      project,
      context.changes,
      kind,
      object.id,
      colorSetting(kind),
      config,
    );
    done.push(object.name);
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  const name = colorName(color);
  if (done.length === 0) {
    return `Nothing changed: ${list(already)} already all ${name}. Tell the user it was already ${name}; do not say that you changed it.`;
  }
  let text = `Done: now all ${name}: ${list(done)}.`;
  if (already.length > 0) {
    text += ` Already ${name} before: ${list(already)}.`;
  }
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text;
}

/**
 * Runs the show_only_groups tool: shows only some groups of labels or points
 * and hides their other items, with the chat's visibility map of the object
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @param options - Optional abort signal
 * @returns The result for the model
 */
export async function showOnlyGroups(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const project = context.projectStore.getState();
  const argsGroups = args["groups"];
  const wantedGroups =
    typeof argsGroups === "string" ? [argsGroups] : argsGroups;
  if (!Array.isArray(wantedGroups) || wantedGroups.length === 0) {
    return 'Error: groups must list the groups to show, e.g. ["1", "3"].';
  }
  const found = findItemsTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const planned = await planEach(
    found.targets,
    async ({ kind, object }) => {
      const target = await findColumn(
        context.dataStore,
        context.tables,
        object,
        args["column"],
        { signal },
      );
      if ("error" in target) {
        return target;
      }
      const { groups } = await readColumn(target.data, target.column, {
        signal,
      });
      if (groups === null) {
        return {
          error: `Error: "${target.column}" has numbers, not groups: filters on numbers (e.g. "area > 50") are not possible.`,
        };
      }
      const map: GroupValueMap<boolean> = {
        id: `chat-filter/${kind}/${object.id}`,
        name: `Chat filter – ${object.name}`,
        values: {},
        default: false,
      };
      for (const wanted of wantedGroups) {
        const group = findGroup(groups, wanted);
        if (group === undefined) {
          return {
            error: `Error: ${target.column} has no group "${String(wanted)}". Its groups: ${someGroups(groups)}.`,
          };
        }
        map.values[group] = true;
      }
      const config: VisibilityConfig = {
        groupBy: { ...target.ref, map: map.id },
      };
      const text = `${object.name}: only ${Object.keys(map.values).join(", ")} of ${target.column}`;
      return { kind, object, map, config, text };
    },
    { signal },
  );
  if ("error" in planned) {
    return planned.error;
  }
  const shownToo: string[] = [];
  for (const { kind, object, map, config } of planned.plans) {
    setChatMap(project, context.changes, "visibilityMaps", map);
    if (!same(visibilityConfigOf(object), config)) {
      change(
        project,
        context.changes,
        kind,
        object.id,
        visibilitySetting(kind),
        config,
      );
    }
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  let text = `Done: now shown ${list(planned.plans.map((plan) => plan.text))}; their other items are hidden.`;
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text + skippedText(planned.skipped);
}

/**
 * Runs the show_all_groups tool: removes the group filter; objects without a
 * filter are left as they are, so that e.g. category "labels" does not show
 * other labels
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
export function showAllGroups(
  args: ToolArguments,
  context: ToolContext,
): string {
  const project = context.projectStore.getState();
  const found = findItemsTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const all: VisibilityConfig = { constant: { value: true } };
  const done: string[] = [];
  const without: string[] = [];
  const shownToo: string[] = [];
  for (const { kind, object } of found.targets) {
    if (same(visibilityConfigOf(object), all)) {
      without.push(object.name);
      continue;
    }
    change(
      project,
      context.changes,
      kind,
      object.id,
      visibilitySetting(kind),
      all,
    );
    done.push(object.name);
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  if (done.length === 0) {
    return `Nothing changed: ${list(without)} had no group filter, so all their items were already shown. Tell the user; do not say that you changed it.`;
  }
  let text = `Done: filter removed, all items shown again: ${list(done)}.`;
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  if (without.length > 0) {
    text += ` No filter, left as they were: ${list(without)}.`;
  }
  return text;
}

/**
 * Warns when another visible object shows the same items as an object whose
 * colors or filter changed, since it can cover the change (e.g. a filter on
 * "cells" while "cells-epcam" is visible)
 *
 * @param project - The project
 * @param someChanges - The changes of one tool call
 * @returns The warning for the user, or "" when there is nothing to warn about
 */
export function overlapWarning(
  project: ProjectStore,
  someChanges: Change[],
): string {
  const coverable = [
    "labelColor",
    "pointColor",
    "labelVisibility",
    "pointVisibility",
  ];
  const changed = someChanges.filter((c) => coverable.includes(c.property));
  const changedKeys = changed.map((c) => `${c.kind} ${c.id}`);
  const warnings = new Set<string>();
  for (const { kind, id } of changed) {
    if (kind !== "labels" && kind !== "points") {
      continue;
    }
    const object = findById(project, kind, id) as ItemsObject | undefined;
    if (object === undefined) {
      continue;
    }
    for (const other of sameItemsAs(project, kind, object)) {
      if (other.visibility && !changedKeys.includes(`${kind} ${other.id}`)) {
        warnings.add(
          `${other.name} is also visible and shows the same items as ${object.name}, in the same place: ` +
            `it can cover this change. Hide ${other.name} to see it.`,
        );
      }
    }
  }
  return [...warnings].join(" ");
}
