/**
 * Which versions of the application the site keeps
 *
 * Kept dependency-free so that the scripts run before `pnpm install`.
 */
import {
  type Version,
  compareVersions,
  formatVersion,
  isPrerelease,
} from "./semver.ts";

/**
 * Applies the retention rules to the released versions
 *
 * Of every MAJOR.MINOR line, the highest stable version is kept, plus the
 * highest prerelease above it (a prerelease of the next patch), or the
 * highest prerelease alone while the line has no stable version. A prerelease
 * is only kept while no stable version above it exists at all, so abandoned
 * prerelease lines do not accumulate. Every other version redirects to the
 * lowest kept version at or above it (a stable one for a stable version, so
 * that a prerelease never replaces a stable version for its users), or to
 * the latest version when there is none. The site root goes to the latest stable
 * version, or to the latest prerelease while no stable version exists.
 *
 * @param versions - The released versions, in any order
 * @param deployable - Whether a version can be deployed at all; the others
 * only ever get a redirect
 * @returns The versions to keep in ascending order, every other version mapped
 * to the kept version it redirects to, and the version the site root
 * redirects to (`null` without any version)
 */
export function planRetention(
  versions: Version[],
  deployable: (version: Version) => boolean = () => true,
): {
  retained: Version[];
  redirects: Record<string, string>;
  latest: Version | null;
} {
  const sorted = [...versions].sort(compareVersions);
  const lines = Map.groupBy(
    sorted.filter(deployable),
    (version) => `${version.major}.${version.minor}`,
  );
  const candidates = [...lines.values()].flatMap((lineVersions) => {
    const highestStable = lineVersions.findLast(
      (version) => !isPrerelease(version),
    );
    const highest = lineVersions.at(-1)!;
    return highestStable === undefined || highestStable === highest
      ? [highest]
      : [highestStable, highest];
  });
  const highestStable = candidates.findLast(
    (version) => !isPrerelease(version),
  );
  const retained = candidates.filter(
    (version) =>
      !isPrerelease(version) ||
      highestStable === undefined ||
      compareVersions(version, highestStable) > 0,
  );
  const latest = highestStable ?? retained.at(-1) ?? null;
  // A stable version never redirects to a prerelease while a stable one exists
  const redirects: Record<string, string> = {};
  for (const version of sorted) {
    const target =
      retained.find(
        (candidate) =>
          compareVersions(candidate, version) >= 0 &&
          (isPrerelease(version) || !isPrerelease(candidate)),
      ) ?? latest;
    if (target && !retained.includes(version)) {
      redirects[formatVersion(version)] = formatVersion(target);
    }
  }
  return { retained, redirects, latest };
}
