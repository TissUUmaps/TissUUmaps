import type {
  GroupValueMap,
  HighlightedItemGroup,
  Labels,
  OpacityConfig,
  Points,
  Shapes,
  VisibilityConfig,
} from "@tissuumaps/core";

/**
 * ID of the transient opacity map that shows only the highlighted group
 *
 * The map comes first, so a project map with this ID never replaces it. While
 * a group is highlighted, other objects that refer to such a project map see
 * the transient map instead. The app creates map IDs with `randomUUID`.
 */
const highlightOpacityMapId = "highlightedItemGroup";

export type HighlightableState = {
  labels: Labels[];
  points: Points[];
  shapes: Shapes[];
  opacityMaps: GroupValueMap<number>[];
};

/**
 * Overrides the opacity configuration of the highlighted group's object so that
 * only the highlighted group is shown
 *
 * The visibility configuration is overridden as well, so that a hidden group
 * is shown while it is highlighted; the other groups are hidden by their
 * opacity. Other objects, even those annotated by the same table, are left as
 * they are.
 *
 * The project itself is left untouched; the returned state is only handed to
 * the viewer.
 *
 * A new opacity map is built on every call, and the renderers compare maps by
 * identity, so callers memoize the result on the state and the highlighted
 * group. The collections without the object are returned as is, so that their
 * renderers are not woken up for a change they do not see.
 *
 * @param state - Objects and opacity maps of the project
 * @param highlightedItemGroup - The highlighted group, or `null` for none
 * @returns The state to render
 */
export function highlightItemGroup(
  state: HighlightableState,
  highlightedItemGroup: HighlightedItemGroup | null,
): HighlightableState {
  if (highlightedItemGroup === null) {
    return state;
  }
  const { annotatedObject, column, group } = highlightedItemGroup;
  const opacityConfig: OpacityConfig = {
    groupBy: { column, map: highlightOpacityMapId },
  };
  const visibilityConfig: VisibilityConfig = { constant: { value: true } };
  return {
    labels:
      "labelsId" in annotatedObject
        ? overrideObject(state.labels, annotatedObject.labelsId, (labels) => ({
            ...labels,
            labelVisibility: visibilityConfig,
            labelOpacity: opacityConfig,
          }))
        : state.labels,
    points:
      "pointsId" in annotatedObject
        ? overrideObject(state.points, annotatedObject.pointsId, (points) => ({
            ...points,
            pointVisibility: visibilityConfig,
            pointOpacity: opacityConfig,
          }))
        : state.points,
    shapes:
      "shapesId" in annotatedObject
        ? overrideObject(state.shapes, annotatedObject.shapesId, (shapes) => ({
            ...shapes,
            shapeVisibility: visibilityConfig,
            shapeOpacity: opacityConfig,
          }))
        : state.shapes,
    // maps are looked up by their first match, so that a project map with the
    // same ID does not win
    opacityMaps: [
      {
        id: highlightOpacityMapId,
        name: "Highlighted group",
        values: { [group]: 1 },
        default: 0,
      },
      ...state.opacityMaps,
    ],
  };
}

/**
 * Overrides one object of a collection
 *
 * @param objects - The objects of one collection
 * @param id - ID of the object to override
 * @param override - Returns the object with its configurations overridden
 * @returns The collection with the object overridden
 */
function overrideObject<TObject extends Labels | Points | Shapes>(
  objects: TObject[],
  id: string,
  override: (object: TObject) => TObject,
): TObject[] {
  return objects.map((object) =>
    object.id === id ? override(object) : object,
  );
}
