/**
 * Verifies that the application resolves the `@tissuumaps/*` packages from
 * the npm registry, not from the workspace
 *
 * Fails unless every package the application depends on is installed outside
 * `packages/`, has the version recorded in the checkout, and carries the
 * published manifest (which has no `tissuumaps-development` export condition).
 *
 * Usage: `node scripts/verify-registry-deps.ts` from the repository root,
 * after `scripts/prepare-registry-build.ts` and `pnpm install`.
 */
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { join, resolve, sep } from "node:path";

import { dependencyFields, readPackageVersions } from "./lib/workspace.ts";

const root = realpathSync(process.cwd());
const packagesDir = resolve(root, "packages") + sep;
const appDir = join(root, "apps", "tissuumaps");
const appManifest = JSON.parse(
  readFileSync(join(appDir, "package.json"), "utf8"),
) as Record<string, Record<string, string> | undefined>;
const appDependencies = new Set(
  dependencyFields.flatMap((field) => Object.keys(appManifest[field] ?? {})),
);
const errors: string[] = [];

for (const [name, version] of readPackageVersions(root)) {
  if (!appDependencies.has(name)) {
    continue;
  }
  const linkPath = join(appDir, "node_modules", name);
  if (!existsSync(linkPath)) {
    errors.push(`${name}: not installed`);
    continue;
  }
  const dir = realpathSync(linkPath);
  const manifest = JSON.parse(
    readFileSync(join(dir, "package.json"), "utf8"),
  ) as {
    version: string;
    exports?: Record<string, string | Record<string, string>>;
  };
  console.log(`${name}@${manifest.version} from ${dir}`);
  if (dir.startsWith(packagesDir)) {
    errors.push(`${name}: resolves to the workspace`);
  }
  if (manifest.version !== version) {
    errors.push(`${name}: version ${manifest.version}, expected ${version}`);
  }
  if (
    Object.values(manifest.exports ?? {}).some(
      (entry) => typeof entry === "object" && "tissuumaps-development" in entry,
    )
  ) {
    errors.push(`${name}: manifest carries the development export condition`);
  }
}

if (errors.length > 0) {
  for (const error of errors) {
    console.error(error);
  }
  process.exit(1);
}
