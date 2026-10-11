/**
 * Publishes the documentation of an application release to the website
 *
 * The website (`TissUUmaps/website`) holds the released documentation of every
 * MAJOR.MINOR line in `releases/<line>/`: `release.json` with the released
 * `version`, and `docs/` with the documentation, including the generated API
 * documentation. This script replaces a line's release with the given one,
 * unless the line holds a higher version, or a stable version while the given
 * one is a prerelease (see {@link shouldPublish}). Every line thus keeps its
 * highest stable version, or its highest prerelease while it has none.
 *
 * Run by the release workflow (`node scripts/gha-publish-docs.ts <version>
 * <docsDir> <releasesDir>`), which commits and pushes the website checkout.
 * The functions are exported for the tests, which import this module without
 * running it.
 */
import {
  cpSync,
  existsSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import semver from "semver";

/**
 * Whether a release replaces the one a line holds
 *
 * @param version - The version to publish
 * @param publishedVersion - The version the line holds, if any
 * @returns `true` unless the line holds a higher version, or a stable version
 * while `version` is a prerelease
 */
export function shouldPublish(
  version: string,
  publishedVersion: string | undefined,
): boolean {
  if (publishedVersion === undefined) {
    return true;
  }
  if (semver.prerelease(version) && !semver.prerelease(publishedVersion)) {
    return false;
  }
  return semver.gte(version, publishedVersion);
}

/**
 * Publishes the documentation of a release into the website's releases (see above)
 *
 * @param options.version - The released version, e.g. `4.1.0`
 * @param options.docsDir - The documentation, including the generated API documentation
 * @param options.releasesDir - The website's releases directory
 * @returns Whether the documentation was published
 */
export function publishDocs(options: {
  version: string;
  docsDir: string;
  releasesDir: string;
}): boolean {
  const { version, docsDir, releasesDir } = options;
  if (semver.valid(version) !== version) {
    throw new Error(`${version} is not a semantic version`);
  }
  const lineDir = join(
    releasesDir,
    `${semver.major(version)}.${semver.minor(version)}`,
  );
  const releaseFile = join(lineDir, "release.json");
  const publishedVersion = existsSync(releaseFile)
    ? (JSON.parse(readFileSync(releaseFile, "utf8")) as { version: string })
        .version
    : undefined;
  if (!shouldPublish(version, publishedVersion)) {
    console.log(
      `${lineDir} holds ${publishedVersion}; not publishing ${version}`,
    );
    return false;
  }
  rmSync(lineDir, { recursive: true, force: true });
  cpSync(docsDir, join(lineDir, "docs"), { recursive: true });
  writeFileSync(releaseFile, JSON.stringify({ version }, null, 2) + "\n");
  console.log(`published ${version} to ${lineDir}`);
  return true;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [version, docsDir, releasesDir] = process.argv.slice(2);
  if (!version || !docsDir || !releasesDir) {
    throw new Error(
      "Usage: node scripts/gha-publish-docs.ts <version> <docsDir> <releasesDir>",
    );
  }
  publishDocs({ version, docsDir, releasesDir });
}
