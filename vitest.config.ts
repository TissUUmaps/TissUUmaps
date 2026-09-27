import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    projects: ["apps/*", "packages/*"],
    coverage: {
      // also report the source files that no test loads
      include: [
        "apps/tissuumaps/src/**/*.{js,jsx,ts,tsx}",
        "packages/*/src/**/*.{js,jsx,ts,tsx}",
      ],
      exclude: ["**/*.d.ts"],
    },
  },
});
