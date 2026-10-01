import { describe, expect, it } from "vitest";

import { computeRowScrollOffset } from "./useCompressedRowVirtualizer";

describe("computeRowScrollOffset", () => {
  // rows of 10 px below a header of 20 px, in a viewport of 100 px
  const rowHeight = 10;
  const headerHeight = 20;
  const viewportHeight = 100;

  it("does not scroll to a row that is fully in view", () => {
    expect(
      computeRowScrollOffset(0, rowHeight, headerHeight, viewportHeight, 0),
    ).toBeNull();
    expect(
      computeRowScrollOffset(7, rowHeight, headerHeight, viewportHeight, 0),
    ).toBeNull();
  });

  it("scrolls a row below the viewport up to its bottom", () => {
    // row 8 spans [100, 110)
    expect(
      computeRowScrollOffset(8, rowHeight, headerHeight, viewportHeight, 0),
    ).toBe(10);
  });

  it("scrolls a row hidden under the header down to just below it", () => {
    // row 2 spans [40, 50), and the header covers [50, 70) at offset 50
    expect(
      computeRowScrollOffset(2, rowHeight, headerHeight, viewportHeight, 50),
    ).toBe(20);
    // the header covers [25, 45) at offset 25, half of row 2
    expect(
      computeRowScrollOffset(2, rowHeight, headerHeight, viewportHeight, 25),
    ).toBe(20);
  });
});
