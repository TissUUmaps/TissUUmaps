/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
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
    react(),
  ],
  build: {
    lib: {
      entry: resolve(import.meta.dirname, "src/index.ts"),
      formats: ["es"],
      fileName: "index",
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
