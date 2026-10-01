import { describe, expect, it } from "vitest";

import { isImageURL } from "./OpenSeadragonImageDataProvider";

describe("isImageURL", () => {
  it("recognizes plain images by their file extension", () => {
    expect(isImageURL("https://example.org/data/slide.jpg")).toBe(true);
    expect(isImageURL("https://example.org/data/slide.JPEG?v=2#top")).toBe(
      true,
    );
    expect(isImageURL("http://localhost:5173/overview.png")).toBe(true);
  });

  it("leaves tile source descriptors alone", () => {
    expect(isImageURL("https://example.org/data/slide.dzi")).toBe(false);
    expect(isImageURL("https://example.org/iiif/slide/info.json")).toBe(false);
    expect(isImageURL("https://example.org/data/slide.jpg/info.json")).toBe(
      false,
    );
  });
});
