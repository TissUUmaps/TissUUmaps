/**
 * Prepares the checkout for building the application from the published
 * `@tissuumaps/*` packages instead of the workspace sources
 *
 * Run in CI only, never commit the result:
 * - `pnpm-workspace.yaml` keeps only the application as a workspace member, so
 *   pnpm cannot link the sibling packages and resolves them from the registry
 *   (`patchedDependencies`, the `minimumReleaseAge` exemption of our packages
 *   and the other settings are kept).
 * - The application's `workspace:` specifiers are pinned to the exact versions
 *   found in the checkout, which are the versions that were published from it.
 *
 * Usage: `node scripts/prepare-registry-build.ts` from the repository root,
 * followed by `pnpm install --no-frozen-lockfile` and
 * `pnpm --filter tissuumaps exec vite build`. Do not run the application's
 * `build` script: its `tsc -b` would follow the project references into the
 * unlinked package sources.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { dependencyFields, readPackageVersions } from "./lib/workspace.ts";

const root = process.cwd();
const versions = readPackageVersions(root);

const appManifestPath = join(root, "apps", "tissuumaps", "package.json");
const appManifest = JSON.parse(readFileSync(appManifestPath, "utf8")) as Record<
  string,
  unknown
>;
for (const field of dependencyFields) {
  const dependencies = appManifest[field] as Record<string, string> | undefined;
  if (!dependencies) {
    continue;
  }
  for (const [name, specifier] of Object.entries(dependencies)) {
    if (!specifier.startsWith("workspace:")) {
      continue;
    }
    const version = versions.get(name);
    if (version === undefined) {
      throw new Error(`${name} is a workspace dependency but not a package`);
    }
    dependencies[name] = version;
    console.log(`${name}: ${specifier} -> ${version}`);
  }
}
writeFileSync(appManifestPath, JSON.stringify(appManifest, null, 2) + "\n");

const workspacePath = join(root, "pnpm-workspace.yaml");
const workspace = readFileSync(workspacePath, "utf8");
// The block ends at the next top-level key; it may hold comments and blank lines
const packagesBlock = /^packages:\n(?:(?:[ \t]+[^\n]*|[ \t]*#[^\n]*)?\n)+/m;
if (!packagesBlock.test(workspace)) {
  throw new Error("pnpm-workspace.yaml has no packages block");
}
writeFileSync(
  workspacePath,
  workspace.replace(packagesBlock, "packages:\n  - apps/tissuumaps\n\n"),
);
console.log(
  `workspace members: apps/tissuumaps (${versions.size} packages unlinked)`,
);
