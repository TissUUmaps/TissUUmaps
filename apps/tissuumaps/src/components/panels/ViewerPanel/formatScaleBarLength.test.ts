import { describe, expect, it } from "vitest";

import { formatScaleBarLength } from "./formatScaleBarLength";

describe("formatScaleBarLength", () => {
  it("formats in the largest unit that keeps the value at least 1", () => {
    expect(formatScaleBarLength(2e-7, "m")).toBe("200 nm");
    expect(formatScaleBarLength(1e-6, "m")).toBe("1 µm");
    expect(formatScaleBarLength(5e-4, "m")).toBe("500 µm");
    expect(formatScaleBarLength(1e-3, "m")).toBe("1 mm");
    expect(formatScaleBarLength(0.02, "m")).toBe("2 cm");
    expect(formatScaleBarLength(2, "m")).toBe("2 m");
    expect(formatScaleBarLength(5000, "m")).toBe("5 km");
  });

  it("hides floating-point noise", () => {
    expect(formatScaleBarLength((0.1 + 0.2) * 1e-6, "m")).toBe("300 nm");
  });

  it("falls back to nanometers below one nanometer", () => {
    expect(formatScaleBarLength(5e-10, "m")).toBe("0.5 nm");
  });

  it("formats pixels without converting them", () => {
    expect(formatScaleBarLength(200, "px")).toBe("200 px");
    expect(formatScaleBarLength(0.5, "px")).toBe("0.5 px");
  });
});
