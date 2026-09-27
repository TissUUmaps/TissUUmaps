import { type Config, ConfigUtils, type Project } from "@tissuumaps/core";

import { useProjectStore } from "@/stores/project";

/**
 * Collects the IDs of the maps that some of the project's configurations refer
 * to
 *
 * Only reads the labels, points and shapes, whose configurations can refer to
 * a map, so that its component does not re-render on every other project
 * change.
 *
 * @param getConfigs - Returns the configurations to check
 * @returns The IDs of the maps that the configurations refer to
 */
export function useReferencedMapIds(
  getConfigs: (
    project: Pick<Project, "labels" | "points" | "shapes">,
  ) => Config<string>[],
): Set<string> {
  const labels = useProjectStore((state) => state.labels);
  const points = useProjectStore((state) => state.points);
  const shapes = useProjectStore((state) => state.shapes);
  return ConfigUtils.getGroupByMapIds(getConfigs({ labels, points, shapes }));
}
