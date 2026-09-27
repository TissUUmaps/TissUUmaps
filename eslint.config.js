import js from "@eslint/js";
import eslintConfigPrettier from "eslint-config-prettier/flat";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import { defineConfig, globalIgnores } from "eslint/config";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig([
  globalIgnores([
    "**/.docusaurus",
    "**/build",
    "**/coverage",
    "**/dist",
    "**/node_modules",
    "apps/docs/docs/api",
  ]),
  js.configs.recommended,
  tseslint.configs.recommendedTypeChecked,
  {
    files: ["**/*.{js,jsx,ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2025,
      globals: globals.browser,
      parserOptions: {
        projectService: {
          defaultProject: "tsconfig.node.json",
          allowDefaultProject: [
            "eslint.config.js",
            "vitest.config.ts",
            "apps/*/vite.config.ts",
            "packages/*/vite.config.ts",
          ],
        },
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "no-duplicate-imports": "error",
      "object-shorthand": "error",
      "@typescript-eslint/consistent-type-imports": "error",
      "@typescript-eslint/no-import-type-side-effects": "error",
    },
  },
  {
    files: [
      "apps/tissuumaps/**/*.{js,jsx,ts,tsx}",
      "packages/@tissuumaps-react/**/*.{js,jsx,ts,tsx}",
    ],
    extends: [reactHooks.configs.flat.recommended, reactRefresh.configs.vite],
  },
  eslintConfigPrettier,
]);
