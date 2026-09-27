/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  defaultClientConditions,
  defaultServerConditions,
  defineConfig,
} from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const packageJson = JSON.parse(
  readFileSync(resolve(import.meta.dirname, "./package.json"), "utf8"),
) as { version: string; repository: { url: string } };

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss(), mode === "production" && viteSingleFile()],
  // Zarr probes for optional files: the dev server must 404 them, not serve index.html.
  appType: "mpa",
  build: {
    chunkSizeWarningLimit: 2048,
    rolldownOptions: {
      checks: {
        pluginTimings: false,
      },
    },
  },
  test: {
    include: [
      "src/**/*.test.js",
      "src/**/*.test.jsx",
      "src/**/*.test.ts",
      "src/**/*.test.tsx",
    ],
    environment: "jsdom",
  },
  resolve: {
    conditions:
      mode === "production"
        ? [...defaultClientConditions]
        : ["tissuumaps-development", ...defaultClientConditions],
    // shadcn/ui
    alias: {
      "@": resolve(import.meta.dirname, "./src"),
    },
  },
  ssr: {
    resolve: {
      conditions:
        mode === "production"
          ? [...defaultServerConditions]
          : ["tissuumaps-development", ...defaultServerConditions],
    },
  },
  define: {
    __APP_VERSION__: JSON.stringify(packageJson.version),
    __APP_REPOSITORY_URL__: JSON.stringify(
      packageJson.repository.url.replace(/^git\+/, "").replace(/\.git$/, ""),
    ),
    "import.meta.env.VITE_CUSTOM_HTML": JSON.stringify(
      process.env.VITE_CUSTOM_HTML_FILE
        ? readFileSync(process.env.VITE_CUSTOM_HTML_FILE, "utf8")
        : "",
    ),
  },
}));
