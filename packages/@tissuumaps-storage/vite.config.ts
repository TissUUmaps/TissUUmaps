/// <reference types="vitest/config" />
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

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [
    dts({
      bundleTypes: true,
      tsconfigPath: resolve(import.meta.dirname, "tsconfig.ts59.json"),
    }),
  ],
  worker: {
    format: "es",
    // geotiff.js's worker imports its decoders dynamically; an inlined worker
    // cannot fetch chunks, so workers are built as a single file
    rolldownOptions: {
      output: {
        codeSplitting: false,
      },
    },
  },
  build: {
    lib: {
      entry: {
        index: resolve(import.meta.dirname, "src/index.ts"),
        csv: resolve(import.meta.dirname, "src/csv/index.ts"),
        geojson: resolve(import.meta.dirname, "src/geojson/index.ts"),
        hdf5: resolve(import.meta.dirname, "src/hierarchical/hdf5/index.ts"),
        "ome-zarr": resolve(import.meta.dirname, "src/ome-zarr/index.ts"),
        openseadragon: resolve(
          import.meta.dirname,
          "src/openseadragon/index.ts",
        ),
        parquet: resolve(import.meta.dirname, "src/parquet/index.ts"),
        table: resolve(import.meta.dirname, "src/table/index.ts"),
        tiff: resolve(import.meta.dirname, "src/tiff/index.ts"),
        zarr: resolve(import.meta.dirname, "src/hierarchical/zarr/index.ts"),
      },
      formats: ["es"],
    },
    // Only published dependencies and peers are external; the rest is bundled:
    // - Inline workers (`?worker&inline`) must be self-contained, so they bundle
    //   their deps (hyparquet, hyparquet-compressors, h5wasm), including what
    //   they use from @tissuumaps/core (which must therefore stay tree-shakable,
    //   see e.g. its palettes.ts).
    // - The git-hosted forks of @zarrita/storage and geotiff-tilesource are
    //   bundled and kept as devDependencies, so the published manifest has no
    //   git dependencies; revert once the changes are released upstream.
    // - omezarr-tilesource's peers (zarrita, ome-zarr.js) are declared as peers
    //   (and devDependencies) of this package, as their types are part of its
    //   public API (e.g. OMEZarr).
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
    typecheck: {
      tsconfig: resolve(import.meta.dirname, "tsconfig.test.json"),
    },
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
