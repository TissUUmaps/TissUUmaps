import { ImageChannelViewMode, Marker } from "@tissuumaps/core";

import type { MapKind, ObjectKind } from "./types";

// call_action lets the strong models call the actions of the stores, which
// they know from the types (store-types.generated.ts). Only actions that can
// be taken back are allowed, and every argument is checked first: if anything
// is wrong, nothing changes and the model is told why.

/** The actions that update an entry of a list, with the entry's type */
export const updateActions: Record<
  string,
  { kind: ObjectKind | "layers"; type: string }
> = {
  updateLayer: { kind: "layers", type: "Layer" },
  updateImage: { kind: "images", type: "Image" },
  updateLabels: { kind: "labels", type: "Labels" },
  updatePoints: { kind: "points", type: "Points" },
  updateShapes: { kind: "shapes", type: "Shapes" },
};

/** The kinds of values that settings and maps hold */
export type ValueKind = "color" | "boolean" | "opacity" | "size" | "marker";

/** The lists of maps, with their name in the actions and their values */
export const mapLists: Record<MapKind, { name: string; value: ValueKind }> = {
  markerMaps: { name: "MarkerMap", value: "marker" },
  sizeMaps: { name: "SizeMap", value: "size" },
  colorMaps: { name: "ColorMap", value: "color" },
  visibilityMaps: { name: "VisibilityMap", value: "boolean" },
  opacityMaps: { name: "OpacityMap", value: "opacity" },
};

/** The actions that move an entry to another index, by list */
export const moveActions: Record<string, ObjectKind | "layers"> = {
  moveLayer: "layers",
  moveImage: "images",
  moveLabels: "labels",
  movePoints: "points",
  moveShapes: "shapes",
};

/** The settings of the view in the app store that call_action can set */
type ViewSetting =
  | "highlightedItemGroup"
  | "expandedImageIds"
  | "expandedLabelsIds"
  | "expandedPointsIds"
  | "expandedShapesIds"
  | "expandedTableIds";

/**
 * The actions of the app store that only change the view, with the setting
 * they set and, for the expanded entries, the list of the IDs they take
 */
export const viewActions: Record<
  string,
  { setting: ViewSetting; list?: ObjectKind | "tables" }
> = {
  setHighlightedItemGroup: { setting: "highlightedItemGroup" },
  setExpandedImageIds: { setting: "expandedImageIds", list: "images" },
  setExpandedLabelsIds: { setting: "expandedLabelsIds", list: "labels" },
  setExpandedPointsIds: { setting: "expandedPointsIds", list: "points" },
  setExpandedShapesIds: { setting: "expandedShapesIds", list: "shapes" },
  setExpandedTableIds: { setting: "expandedTableIds", list: "tables" },
};

/** The settings of the project itself that call_action can set */
export const projectActions: Record<string, "name" | "viewerBackgroundColor"> =
  {
    setName: "name",
    setViewerBackgroundColor: "viewerBackgroundColor",
  };

/** The actions the model may not call, with the reason it is told */
export const refusedActions: [RegExp, string][] = [
  [/^(delete|clear)/, "deleting cannot be taken back safely"],
  [
    /^add(Layer|Image|Labels|Points|Shapes|Table)$/,
    "new objects would need their data from a file",
  ],
  [
    /^updateTable$/,
    "a table only has a name and the source of its data, which cannot be changed",
  ],
  [/^set(OS|GL)Options$/, "the viewer options are not described in the types"],
  [
    /^setInteractionMode$/,
    "the drawing modes are not finished, and the viewer could not be panned any more",
  ],
  [/^setActivePanelId$/, "another panel could hide the chat"],
  [
    /^setImageChannelPreview$/,
    "to show one channel, call updateImage with channelViewMode and activeChannel",
  ],
  [
    /^(setWorkspace|setProjectOpen|register\w*DataProvider|setDark)$/,
    "it changes how TissUUmaps opens and loads data, or its settings",
  ],
];

/** The names of the actions the model may call, for its instructions */
export const allowedActionNames = [
  ...Object.keys(updateActions),
  ...Object.values(mapLists).flatMap(({ name }) => [
    `add${name}`,
    `update${name}`,
  ]),
  ...Object.keys(moveActions),
  ...Object.keys(projectActions),
  ...Object.keys(viewActions),
];

/** What a setting that is a configuration (e.g. labelColor) can hold */
export type ConfigRule = {
  value: ValueKind;
  maps: MapKind;
  /** Whether groupBy needs a map, which colors and markers do not */
  mapRequired: boolean;
  /** Colors take palettes, a range, and random colors */
  color?: boolean;
  /** Sizes take a unit */
  unit?: boolean;
};

/** The rule of the color settings */
const colorRule: ConfigRule = {
  value: "color",
  maps: "colorMaps",
  mapRequired: false,
  color: true,
};

/** The rule of the visibility settings */
const visibilityRule: ConfigRule = {
  value: "boolean",
  maps: "visibilityMaps",
  mapRequired: true,
};

/** The rule of the opacity settings */
const opacityRule: ConfigRule = {
  value: "opacity",
  maps: "opacityMaps",
  mapRequired: true,
};

/** The settings of objects that are configurations, with their rules */
export const configSettings: Record<string, ConfigRule> = {
  labelColor: colorRule,
  pointColor: colorRule,
  shapeFillColor: colorRule,
  shapeStrokeColor: colorRule,
  labelVisibility: visibilityRule,
  pointVisibility: visibilityRule,
  shapeVisibility: visibilityRule,
  shapeFillVisibility: visibilityRule,
  shapeStrokeVisibility: visibilityRule,
  labelOpacity: opacityRule,
  pointOpacity: opacityRule,
  shapeOpacity: opacityRule,
  shapeFillOpacity: opacityRule,
  shapeStrokeOpacity: opacityRule,
  pointSize: {
    value: "size",
    maps: "sizeMaps",
    mapRequired: true,
    unit: true,
  },
  pointMarker: { value: "marker", maps: "markerMaps", mapRequired: false },
};

/** The marker shapes, as their numbers */
export const markerValues: number[] = Object.values(Marker);

/** The ways the channels of an image can be shown */
export const channelViewModes: string[] = Object.values(ImageChannelViewMode);

/** The units of sizes */
export const coordinateSpaces = ["layer", "data", "world"];
