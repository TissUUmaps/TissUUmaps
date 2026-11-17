import { describe, expect, it } from "vitest";

import { getCoordinateProperties } from "./getCoordinateProperties";

describe("getCoordinateProperties", () => {
  it("finds the coordinate controls in nested layouts", () => {
    expect(
      getCoordinateProperties({
        type: "VerticalLayout",
        elements: [
          { type: "Control", scope: "#/properties/table" },
          {
            type: "HorizontalLayout",
            elements: [
              {
                type: "Control",
                scope: "#/properties/xColumn",
                options: { tableColumn: true, coordinate: "x" },
              },
              {
                type: "Control",
                scope: "#/properties/yColumn",
                options: { tableColumn: true, coordinate: "y" },
              },
            ],
          },
        ],
      } as never),
    ).toEqual({ x: "xColumn", y: "yColumn" });
  });

  it("requires both coordinates", () => {
    expect(
      getCoordinateProperties({
        type: "VerticalLayout",
        elements: [
          {
            type: "Control",
            scope: "#/properties/x",
            options: { coordinate: "x" },
          },
        ],
      } as never),
    ).toBeNull();
  });
});
