/// <reference types="vitest/config" />
import { resolve } from "node:path";
import dts from "unplugin-dts/vite";
import {
  defaultClientConditions,
  defaultServerConditions,
  defineConfig,
} from "vite";

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
    // Worker-only deps (hyparquet, hyparquet-compressors, h5wasm) are
    // intentionally NOT externalized: the workers are imported with
    // `?worker&inline`, so they must be self-contained and their deps get
    // bundled into the inline worker.
    rolldownOptions: {
      external: [
        "@tissuumaps/core",
        "geotiff",
        "omezarr-tilesource",
        "openseadragon",
        "papaparse",
        "zarrita",
      ],
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
