import { describe, expect, it } from "vitest";

import { formatScaleBarLength } from "./formatScaleBarLength";

describe("formatScaleBarLength", () => {
  it("formats in the largest unit that keeps the value at least 1", () => {
    expect(formatScaleBarLength(0.2, "µm")).toBe("200 nm");
    expect(formatScaleBarLength(1, "µm")).toBe("1 µm");
    expect(formatScaleBarLength(500, "µm")).toBe("500 µm");
    expect(formatScaleBarLength(1000, "µm")).toBe("1 mm");
    expect(formatScaleBarLength(2e6, "µm")).toBe("2 m");
  });

  it("hides floating-point noise", () => {
    expect(formatScaleBarLength(0.1 + 0.2, "µm")).toBe("300 nm");
  });

  it("falls back to nanometers below one nanometer", () => {
    expect(formatScaleBarLength(5e-4, "µm")).toBe("0.5 nm");
  });

  it("formats pixels without converting them", () => {
    expect(formatScaleBarLength(200, "px")).toBe("200 px");
    expect(formatScaleBarLength(0.5, "px")).toBe("0.5 px");
  });
});
