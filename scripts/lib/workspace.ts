/** Helpers for reading the workspace packages */
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

/** The `package.json` fields that can hold dependencies */
export const dependencyFields = [
  "dependencies",
  "devDependencies",
  "peerDependencies",
  "optionalDependencies",
];

/**
 * Reads the names and versions of the publishable packages under `packages/`
 *
 * Like pnpm's workspace globs, only directories holding a `package.json` count
 * as packages.
 *
 * @param root - The repository root
 * @returns The package versions by package name
 */
export function readPackageVersions(root: string): Map<string, string> {
  const versions = new Map<string, string>();
  const packagesDir = join(root, "packages");
  for (const entry of readdirSync(packagesDir, { withFileTypes: true })) {
    const manifestPath = join(packagesDir, entry.name, "package.json");
    if (!entry.isDirectory() || !existsSync(manifestPath)) {
      continue;
    }
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
      name: string;
      version: string;
      private?: boolean;
    };
    if (!manifest.private) {
      versions.set(manifest.name, manifest.version);
    }
  }
  return versions;
}
