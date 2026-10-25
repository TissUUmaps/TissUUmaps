import { expect, it } from "vitest";

import { generateStoreTypes } from "../../scripts/generate-store-types.ts";
import { storeProperties, storeTypes } from "./store-types.generated";

it("is up to date with the types of @tissuumaps/core (else run `pnpm run generate:store-types`)", () => {
  const { text, properties } = generateStoreTypes();
  expect(storeTypes).toBe(text);
  expect(storeProperties).toEqual(properties);
}, 120_000);
