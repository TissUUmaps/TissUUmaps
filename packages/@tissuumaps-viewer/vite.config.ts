/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
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
    react(),
  ],
  build: {
    lib: {
      entry: resolve(import.meta.dirname, "src/index.ts"),
      formats: ["es"],
      fileName: "index",
    },
    rolldownOptions: {
      external: [
        "@tissuumaps/core",
        "@tissuumaps/render",
        "react",
        "react/jsx-runtime",
        "react-dom",
      ],
      checks: {
        pluginTimings: false,
      },
    },
  },
  test: {
    include: [
      "./src/**/*.test.js",
      "./src/**/*.test.jsx",
      "./src/**/*.test.ts",
      "./src/**/*.test.tsx",
    ],
    typecheck: {
      tsconfig: resolve(import.meta.dirname, "tsconfig.test.json"),
    },
    environment: "jsdom",
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
