import { describe, expect, it } from "vitest";

import { renderedDataObjectDefaults } from "./base";
import { pointsDefaults } from "./points";
import { createProject, projectDefaults } from "./project";

describe("createProject", () => {
  it("applies the defaults of undefined properties", () => {
    const project = createProject({
      name: "Project",
      viewerBackgroundColor: undefined,
      osOptions: {
        viewerOptions: { animationTime: undefined },
        viewerAnimationStartOptions: {},
        viewerAnimationFinishOptions: {},
      },
      points: [
        {
          id: "points",
          name: "Points",
          layer: "layer",
          visibility: undefined,
          pointColor: undefined,
          dataSource: { type: "csv", table: undefined },
        },
      ],
    });
    expect(project.viewerBackgroundColor).toEqual(
      projectDefaults.viewerBackgroundColor,
    );
    expect(project.osOptions.viewerOptions.animationTime).toBe(
      projectDefaults.osOptions.viewerOptions.animationTime,
    );
    expect(project.points[0]!.visibility).toBe(
      renderedDataObjectDefaults.visibility,
    );
    expect(project.points[0]!.pointColor).toEqual(pointsDefaults.pointColor);
    expect(project.points[0]!.dataSource).not.toHaveProperty("table");
  });
});
