// @vitest-environment jsdom
// the provider module pulls in OpenSeadragon, which reads document on import
import { describe, expect, it } from "vitest";

import { OpenSeadragonImageDataProvider } from "./OpenSeadragonImageDataProvider";
import type { OpenSeadragonImageDataSource } from "./OpenSeadragonImageDataSource";

describe("OpenSeadragonImageDataProvider.normalize", () => {
  const provider = new OpenSeadragonImageDataProvider();

  it("keeps an inline tile source and drops the source", () => {
    const tileSource = { type: "image", url: "https://example.org/a.gif" };
    expect(
      provider.normalize(
        { type: "image", source: "a.png", tileSource },
        null,
        null,
      ),
    ).toMatchObject({ source: undefined, tileSource });
  });

  it("rejects a tile source given as a URL", () => {
    const dataSource = {
      type: "image",
      tileSource: "https://example.org/a.dzi",
    } as unknown as OpenSeadragonImageDataSource;
    expect(() => provider.normalize(dataSource, null, null)).toThrow(
      "Tile source 'https://example.org/a.dzi' must be given as source",
    );
  });
});
