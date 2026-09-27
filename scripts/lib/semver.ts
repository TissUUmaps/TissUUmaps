/**
 * Semantic version handling for the release scripts
 *
 * Kept dependency-free so that the scripts run before `pnpm install`.
 */

/** A parsed semantic version; the prerelease identifiers are empty for a stable version */
export type Version = {
  major: number;
  minor: number;
  patch: number;
  prerelease: (string | number)[];
};

// Build metadata is not accepted: it is ignored by precedence, so two versions
// differing only in metadata would collide as site directories
const numeric = String.raw`0|[1-9]\d*`;
const identifier = String.raw`(?:${numeric}|\d*[A-Za-z-][0-9A-Za-z-]*)`;
const versionPattern = new RegExp(
  String.raw`^(${numeric})\.(${numeric})\.(${numeric})(?:-(${identifier}(?:\.${identifier})*))?$`,
);

/**
 * Parses a semantic version
 *
 * @param version - The version string, without a leading `v`
 * @returns The parsed version, or `null` if the string is not a valid semantic version
 */
export function parseVersion(version: string): Version | null {
  const match = versionPattern.exec(version);
  if (!match) {
    return null;
  }
  const [, major, minor, patch, prerelease] = match;
  return {
    major: Number(major),
    minor: Number(minor),
    patch: Number(patch),
    prerelease:
      prerelease === undefined
        ? []
        : prerelease
            .split(".")
            .map((part) => (/^\d+$/.test(part) ? Number(part) : part)),
  };
}

/**
 * Whether a version is a prerelease
 *
 * @param version - The parsed version
 * @returns `true` if the version has prerelease identifiers
 */
export function isPrerelease(version: Version): boolean {
  return version.prerelease.length > 0;
}

/**
 * Formats a parsed version back into its string form
 *
 * @param version - The parsed version
 * @returns The version string
 */
export function formatVersion(version: Version): string {
  const core = `${version.major}.${version.minor}.${version.patch}`;
  return isPrerelease(version)
    ? `${core}-${version.prerelease.join(".")}`
    : core;
}

function compareIdentifiers(a: string | number, b: string | number): number {
  if (typeof a === "number" && typeof b === "number") {
    return a - b;
  }
  // Numeric identifiers always have lower precedence than alphanumeric ones
  if (typeof a === "number") {
    return -1;
  }
  if (typeof b === "number") {
    return 1;
  }
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * Compares two versions by semantic version precedence
 *
 * A prerelease has lower precedence than the corresponding stable version;
 * prerelease identifiers are compared one by one, numerically where numeric,
 * and a shorter list of identifiers ranks below a longer one with the same
 * prefix.
 *
 * @param a - The first version
 * @param b - The second version
 * @returns A negative number if `a` ranks below `b`, a positive number if above, and `0` if equal
 */
export function compareVersions(a: Version, b: Version): number {
  return (
    a.major - b.major ||
    a.minor - b.minor ||
    a.patch - b.patch ||
    comparePrereleases(a.prerelease, b.prerelease)
  );
}

function comparePrereleases(
  a: (string | number)[],
  b: (string | number)[],
): number {
  if (a.length === 0 || b.length === 0) {
    return b.length - a.length;
  }
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const result = compareIdentifiers(a[i]!, b[i]!);
    if (result !== 0) {
      return result;
    }
  }
  return a.length - b.length;
}
