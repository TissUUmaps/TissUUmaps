import { expect, it } from "vitest";

import { generateStoreTypes } from "../../scripts/generate-store-types.ts";
import { storeTypes } from "./store-types.generated";

it("is up to date with the types of @tissuumaps/core (else run `pnpm run generate:store-types`)", () => {
  expect(storeTypes).toBe(generateStoreTypes());
}, 120_000);
