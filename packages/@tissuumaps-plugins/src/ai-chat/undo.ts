import type { AppStoreApi, ProjectStoreApi } from "@tissuumaps/core";

import {
  applyMove,
  deleteMap,
  setProjectSetting,
  setView,
} from "./call-action";
import { applyUpdate, findById, isMapKind, round } from "./objects";
import type { Change } from "./types";

/** Added to an undo line when the user had changed the value by hand */
export const handNote =
  " (it had been changed by hand after the chat changed it)";

/**
 * Describes how a value changed, e.g. "he_image opacity 0.5 → 1" or
 * "transcripts hidden → visible"
 *
 * @param change - The change
 * @param from - The value before
 * @param to - The value after
 * @returns The text
 */
export function describeValueChange(
  change: Change,
  from: unknown,
  to: unknown,
): string {
  if (change.type === "added") {
    return `${change.name} (the map added by the chat is deleted)`;
  }
  if (change.type === "moved") {
    return `${change.name} back at position ${Number(to) + 1}`;
  }
  if (change.kind === "app") {
    return change.property === "highlightedItemGroup"
      ? "the highlighted group"
      : "the expanded entries";
  }
  // the settings of colors and filters are too long to show
  if (change.property.endsWith("Color")) {
    return `${change.name} colors`;
  }
  if (change.property.endsWith("Visibility")) {
    return `${change.name} group filter`;
  }
  const format = (value: unknown): string => {
    if (change.property === "visibility") {
      return value === true ? "visible" : "hidden";
    }
    if (typeof value === "string") {
      return `"${value}"`;
    }
    return String(round(Number(value)));
  };
  const labels: Record<string, string> = {
    visibility: "",
    opacity: " opacity",
    pointSizeFactor: " point size factor",
  };
  const label = labels[change.property] ?? ` ${change.property}`;
  // e.g. a transform or the channels: only named
  if (typeof from === "object" || typeof to === "object") {
    return `${change.name}${label}`;
  }
  return `${change.name}${label} ${format(from)} → ${format(to)}`;
}

/**
 * What a change is about, as it is now: the value of the property, the
 * added map, or the index of the moved entry; `undefined` if the entry is
 * not in the project any more
 *
 * @param change - The change
 * @param projectStore - The project store
 * @param appStore - The app store
 * @returns The current value, or `undefined`
 */
export function currentValue(
  change: Change,
  projectStore: ProjectStoreApi,
  appStore: AppStoreApi,
): { value: unknown } | undefined {
  const project = projectStore.getState();
  if (change.kind === "project" || change.kind === "app") {
    const state: object =
      change.kind === "project" ? project : appStore.getState();
    return { value: (state as Record<string, unknown>)[change.property] };
  }
  if (change.type === "moved") {
    const index = project[change.kind].findIndex(
      (entry) => entry.id === change.id,
    );
    return index === -1 ? undefined : { value: index };
  }
  const entry = findById(project, change.kind, change.id);
  if (entry === undefined) {
    return undefined;
  }
  return {
    value:
      change.type === "added"
        ? entry
        : (entry as unknown as Record<string, unknown>)[change.property],
  };
}

/**
 * Puts back what one change changed, with the store actions
 *
 * @param change - The change
 * @param projectStore - The project store
 * @param appStore - The app store
 */
export function takeBack(
  change: Change,
  projectStore: ProjectStoreApi,
  appStore: AppStoreApi,
): void {
  const project = projectStore.getState();
  if (change.kind === "project") {
    setProjectSetting(project, change.property, change.before);
  } else if (change.kind === "app") {
    setView(appStore.getState(), change.property, change.before);
  } else if (change.type === "added" && isMapKind(change.kind)) {
    deleteMap(project, change.kind, change.id);
  } else if (change.type === "moved" && !isMapKind(change.kind)) {
    applyMove(project, change.kind, change.id, change.before as number);
  } else {
    applyUpdate(project, change.kind, change.id, {
      [change.property]: change.before,
    });
  }
}
