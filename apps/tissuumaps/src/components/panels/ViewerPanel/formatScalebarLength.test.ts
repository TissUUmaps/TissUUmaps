import { describe, expect, it } from "vitest";

import { PhysicalUnit } from "@tissuumaps/core";

import { formatScalebarLength } from "./formatScalebarLength";

// the formatter follows the runtime locale, e.g. "0,5" in Swedish
const half = (0.5).toLocaleString();

describe("formatScalebarLength", () => {
  it("formats in the largest unit that keeps the value at least 1", () => {
    expect(formatScalebarLength(5e-10, PhysicalUnit.meter)).toBe("500 pm");
    expect(formatScalebarLength(2e-7, PhysicalUnit.meter)).toBe("200 nm");
    expect(formatScalebarLength(1e-6, PhysicalUnit.meter)).toBe("1 µm");
    expect(formatScalebarLength(5e-4, PhysicalUnit.meter)).toBe("500 µm");
    expect(formatScalebarLength(1e-3, PhysicalUnit.meter)).toBe("1 mm");
    expect(formatScalebarLength(0.02, PhysicalUnit.meter)).toBe("20 mm");
    expect(formatScalebarLength(2, PhysicalUnit.meter)).toBe("2 m");
    expect(formatScalebarLength(5000, PhysicalUnit.meter)).toBe("5 km");
  });

  it("hides floating-point noise", () => {
    expect(formatScalebarLength((0.1 + 0.2) * 1e-6, PhysicalUnit.meter)).toBe(
      "300 nm",
    );
  });

  it("falls back to picometers below one picometer", () => {
    expect(formatScalebarLength(5e-13, PhysicalUnit.meter)).toBe(`${half} pm`);
  });

  it("formats any other unit without converting it", () => {
    expect(formatScalebarLength(200, "units")).toBe("200 units");
    expect(formatScalebarLength(0.5, "units")).toBe(`${half} units`);
    expect(formatScalebarLength(5000, "arcsec")).toBe(
      `${(5000).toLocaleString()} arcsec`,
    );
  });
});
