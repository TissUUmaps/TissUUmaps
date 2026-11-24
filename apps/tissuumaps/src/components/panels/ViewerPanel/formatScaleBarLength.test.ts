import { describe, expect, it } from "vitest";

import { PhysicalUnit } from "@tissuumaps/core";

import { formatScaleBarLength } from "./formatScaleBarLength";

// the formatter follows the runtime locale, e.g. "0,5" in Swedish
const half = (0.5).toLocaleString();

describe("formatScaleBarLength", () => {
  it("formats in the largest unit that keeps the value at least 1", () => {
    expect(formatScaleBarLength(5e-10, PhysicalUnit.meter)).toBe("500 pm");
    expect(formatScaleBarLength(2e-7, PhysicalUnit.meter)).toBe("200 nm");
    expect(formatScaleBarLength(1e-6, PhysicalUnit.meter)).toBe("1 µm");
    expect(formatScaleBarLength(5e-4, PhysicalUnit.meter)).toBe("500 µm");
    expect(formatScaleBarLength(1e-3, PhysicalUnit.meter)).toBe("1 mm");
    expect(formatScaleBarLength(0.02, PhysicalUnit.meter)).toBe("20 mm");
    expect(formatScaleBarLength(2, PhysicalUnit.meter)).toBe("2 m");
    expect(formatScaleBarLength(5000, PhysicalUnit.meter)).toBe("5 km");
  });

  it("hides floating-point noise", () => {
    expect(formatScaleBarLength((0.1 + 0.2) * 1e-6, PhysicalUnit.meter)).toBe(
      "300 nm",
    );
  });

  it("falls back to picometers below one picometer", () => {
    expect(formatScaleBarLength(5e-13, PhysicalUnit.meter)).toBe(`${half} pm`);
  });

  it("formats any other unit without converting it", () => {
    expect(formatScaleBarLength(200, "units")).toBe("200 units");
    expect(formatScaleBarLength(0.5, "units")).toBe(`${half} units`);
    expect(formatScaleBarLength(5000, "arcsec")).toBe(
      `${(5000).toLocaleString()} arcsec`,
    );
  });
});
