/// <reference types="vitest/config" />
import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import dts from "unplugin-dts/vite";
import {
  defaultClientConditions,
  defaultServerConditions,
  defineConfig,
} from "vite";

import packageJson from "./package.json" with { type: "json" };

// published dependencies and peers, which are external with their subpaths
const {
  dependencies = {},
  peerDependencies = {},
}: {
  dependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
} = packageJson;
const externalPackages = [
  ...Object.keys(dependencies),
  ...Object.keys(peerDependencies),
];

// the entry points besides the root one, which only re-exports them
const subpathEntries = {
  spatialdata: resolve(import.meta.dirname, "src/spatialdata/index.ts"),
};

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [
    dts({
      bundleTypes: true,
      tsconfigPath: resolve(import.meta.dirname, "tsconfig.ts59.json"),
      // the declarations are bundled per entry point, so the root one would
      // declare its own copies of the subpath entries' types, incompatible with
      // theirs (e.g. because of private members); re-export theirs instead
      afterBuild: () =>
        writeFile(
          resolve(import.meta.dirname, "dist/index.d.ts"),
          Object.keys(subpathEntries)
            .map((name) => `export * from "./${name}.js";\n`)
            .join(""),
        ),
    }),
  ],
  build: {
    minify: false,
    lib: {
      entry: {
        index: resolve(import.meta.dirname, "src/index.ts"),
        ...subpathEntries,
      },
      formats: ["es"],
    },
    rolldownOptions: {
      external: (id) =>
        externalPackages.some(
          (name) => id === name || id.startsWith(`${name}/`),
        ),
      checks: {
        pluginTimings: false,
      },
    },
  },
  test: {
    include: ["./src/**/*.test.js", "./src/**/*.test.ts"],
  },
  resolve: {
    conditions:
      mode === "production"
        ? [...defaultClientConditions]
        : ["tissuumaps-development", ...defaultClientConditions],
  },
  ssr: {
    resolve: {
      conditions:
        mode === "production"
          ? [...defaultServerConditions]
          : ["tissuumaps-development", ...defaultServerConditions],
    },
  },
}));
