import type { Points } from "@tissuumaps/core";

import {
  change,
  findTargets,
  list,
  objectKinds,
  round,
  showIfHidden,
} from "./objects";
import type { ProjectObject, ToolArguments, ToolContext } from "./types";

/**
 * Runs the set_visibility tool
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
export function setVisibility(
  args: ToolArguments,
  context: ToolContext,
): string {
  const project = context.projectStore.getState();
  const visible = args["visible"];
  if (typeof visible !== "boolean") {
    return "Error: visible must be true or false.";
  }
  const found = findTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const state = visible ? "visible" : "hidden";
  const changed: string[] = [];
  const already: string[] = [];
  for (const { kind, object } of found.targets) {
    if (object.visibility === visible) {
      already.push(object.name);
    } else {
      change(project, context.changes, kind, object.id, "visibility", visible);
      changed.push(object.name);
    }
  }
  if (changed.length === 0) {
    // small models follow the last message best, so it says how to answer
    return (
      `Nothing changed: ${list(already)} already ${state}. ` +
      `Tell the user it was already ${state}; do not say that you ${visible ? "showed" : "hid"} it.`
    );
  }
  let text = `Done: now ${state}: ${list(changed)}.`;
  if (already.length > 0) {
    text += ` Already ${state} before: ${list(already)}.`;
  }
  return text;
}

/**
 * Runs the show_only tool: shows the given objects and hides all the others;
 * without ids and category, it hides everything
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
export function showOnly(args: ToolArguments, context: ToolContext): string {
  const project = context.projectStore.getState();
  const ids = args["ids"];
  const noIds = ids === undefined || (Array.isArray(ids) && ids.length === 0);
  const hideEverything =
    args["category"] === undefined && noIds && args["id"] === undefined;
  // as "section id", since an ID is only unique within its section
  let keep: string[] = [];
  if (!hideEverything) {
    const found = findTargets(project, args);
    if ("error" in found) {
      return found.error;
    }
    keep = found.targets.map(({ kind, object }) => `${kind} ${object.id}`);
  }
  const shown: string[] = [];
  const hidden: string[] = [];
  const alreadyVisible: string[] = [];
  const alreadyHidden: string[] = [];
  for (const kind of objectKinds) {
    const objects: readonly ProjectObject[] = project[kind];
    for (const object of objects) {
      const visible = keep.includes(`${kind} ${object.id}`);
      if (object.visibility === visible) {
        (visible ? alreadyVisible : alreadyHidden).push(object.name);
      } else {
        change(
          project,
          context.changes,
          kind,
          object.id,
          "visibility",
          visible,
        );
        (visible ? shown : hidden).push(object.name);
      }
    }
  }
  const lists =
    ` Shown now: ${list(shown)}. Hidden now: ${list(hidden)}.` +
    ` Already visible: ${list(alreadyVisible)}. Already hidden: ${list(alreadyHidden)}.`;
  if (shown.length + hidden.length > 0) {
    return `Done.${lists}`;
  }
  return (
    `Nothing changed: everything was already as requested.${lists}` +
    " Tell the user that it was already like this; do not say that you changed anything."
  );
}

/**
 * Runs the set_opacity tool, which changes the object's own opacity (not the
 * one of its layer)
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
export function setOpacity(args: ToolArguments, context: ToolContext): string {
  const project = context.projectStore.getState();
  const opacity = args["opacity"];
  if (typeof opacity !== "number" || opacity < 0 || opacity > 1) {
    return 'Error: opacity must be a number from 0 to 1 ("50% transparent" is 0.5, "30% transparent" is 0.7).';
  }
  const found = findTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const changed: string[] = [];
  const already: string[] = [];
  const shownToo: string[] = [];
  for (const { kind, object } of found.targets) {
    if (object.opacity === opacity) {
      already.push(object.name);
      continue;
    }
    change(project, context.changes, kind, object.id, "opacity", opacity);
    changed.push(`${object.name} (${round(object.opacity)} → ${opacity})`);
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  if (changed.length === 0) {
    return (
      `Nothing changed: ${list(already)} already had opacity ${opacity}. ` +
      "Tell the user it already had this opacity; do not say that you changed it."
    );
  }
  let text = `Done: opacity is now ${opacity} for ${list(changed)}.`;
  if (already.length > 0) {
    text += ` Already at opacity ${opacity}: ${list(already)}.`;
  }
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text;
}

/**
 * Runs the set_point_size tool: makes the points bigger or smaller by a
 * percentage of their current size
 *
 * The model only copies the percentage the user said and the plugin does the
 * math, which small models get wrong. The tool changes the point size factor,
 * which scales every point, so it also works when the size of each point comes
 * from a table column.
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
export function setPointSize(
  args: ToolArguments,
  context: ToolContext,
): string {
  const project = context.projectStore.getState();
  const percent = args["change_percent"];
  if (typeof percent !== "number" || percent <= -100) {
    return 'Error: change_percent must be a number above -100 ("10% smaller" is -10, "50% bigger" is 50).';
  }
  const found = findTargets(project, args, ["points"]);
  if ("error" in found) {
    return found.error;
  }
  const points = found.targets.map(({ object }) => object as Points);
  if (percent === 0) {
    return (
      `Nothing changed: a change of 0% keeps the size of ${list(points.map((p) => p.name))}. ` +
      "Tell the user the size did not change; do not say that you changed it."
    );
  }
  const changed: string[] = [];
  const shownToo: string[] = [];
  for (const object of points) {
    const newFactor = object.pointSizeFactor * (1 + percent / 100);
    change(
      project,
      context.changes,
      "points",
      object.id,
      "pointSizeFactor",
      newFactor,
    );
    changed.push(
      `${object.name} (point size factor ${round(object.pointSizeFactor)} → ${round(newFactor)})`,
    );
    if (showIfHidden(project, context.changes, "points", object)) {
      shownToo.push(object.name);
    }
  }
  const direction = percent > 0 ? "BIGGER" : "SMALLER";
  let text = `Done: now ${Math.abs(percent)}% ${direction} than before: ${list(changed)}.`;
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text;
}
