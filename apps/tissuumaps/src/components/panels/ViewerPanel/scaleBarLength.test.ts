import { describe, expect, it } from "vitest";

import { formatScaleBarLength, getScaleBarLength } from "./scaleBarLength";

describe("getScaleBarLength", () => {
  it("rounds down to 1, 2 or 5 times a power of ten", () => {
    expect(getScaleBarLength(1.9)).toBe(1);
    expect(getScaleBarLength(4.99)).toBe(2);
    expect(getScaleBarLength(9.99)).toBe(5);
    expect(getScaleBarLength(149)).toBe(100);
    expect(getScaleBarLength(0.03)).toBeCloseTo(0.02);
  });

  it("keeps a length that is already round", () => {
    expect(getScaleBarLength(1)).toBe(1);
    expect(getScaleBarLength(200)).toBe(200);
    expect(getScaleBarLength(5000)).toBe(5000);
  });
});

describe("formatScaleBarLength", () => {
  it("formats in the largest unit that keeps the value at least 1", () => {
    expect(formatScaleBarLength(0.2)).toBe("200 nm");
    expect(formatScaleBarLength(1)).toBe("1 µm");
    expect(formatScaleBarLength(500)).toBe("500 µm");
    expect(formatScaleBarLength(1000)).toBe("1 mm");
    expect(formatScaleBarLength(2e6)).toBe("2 m");
  });

  it("hides floating-point noise", () => {
    expect(formatScaleBarLength(2 * 10 ** -1)).toBe("200 nm");
    expect(formatScaleBarLength(getScaleBarLength(0.03))).toBe("20 nm");
  });

  it("falls back to nanometers below one nanometer", () => {
    expect(formatScaleBarLength(5e-4)).toMatch(/ nm$/);
  });
});
