import { describe, expect, it } from "vitest";

import { getScalebarLength } from "./getScalebarLength";

describe("getScalebarLength", () => {
  it("rounds down to 1, 2 or 5 times a power of ten", () => {
    expect(getScalebarLength(1.9)).toBe(1);
    expect(getScalebarLength(4.99)).toBe(2);
    expect(getScalebarLength(9.99)).toBe(5);
    expect(getScalebarLength(149)).toBe(100);
    expect(getScalebarLength(0.03)).toBeCloseTo(0.02);
  });

  it("keeps a length that is already round", () => {
    expect(getScalebarLength(1)).toBe(1);
    expect(getScalebarLength(200)).toBe(200);
    expect(getScalebarLength(5000)).toBe(5000);
  });
});
