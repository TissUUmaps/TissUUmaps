import { describe, expect, it } from "vitest";

import { getScaleBarLength } from "./getScaleBarLength";

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
