/**
 * Assembles the versioned GitHub Pages site from the release assets
 *
 * Every release of the application (`tissuumaps@<version>`) carries the built
 * application as the asset `tissuumaps-<version>.zip` and its documentation
 * as `tissuumaps-<version>-docs.zip`. Every MAJOR.MINOR line deploys one
 * version under `<MAJOR.MINOR>/` (see {@link selectVersionsToDeploy}); the
 * site root redirects to the latest line and `docs/` to its documentation (see
 * {@link resolveRedirectTarget}). Fails while no version can be deployed, so
 * that the deployed site is left as it is.
 *
 * Run by the release workflow (`node scripts/gha-assemble-pages.ts`), which
 * provides `GITHUB_REPOSITORY` and a `gh` token. Writes the site to `_site/`, the
 * directory `actions/upload-pages-artifact` uploads, and inserts
 * `.github/pages/custom.html` into every deployed application page (the
 * release assets stay free of it). Kept dependency-free so that it runs before
 * `pnpm install`. The functions are exported for the tests, which import this
 * module without running it.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

/** A parsed semantic version; the prerelease identifiers are empty for a stable version */
export type Version = {
  major: number;
  minor: number;
  patch: number;
  prerelease: (string | number)[];
};

/** The site description inlined into the redirect pages */
export type Site = {
  /** The path prefix of the site, e.g. `/TissUUmaps/` */
  prefix: string;
  /** The line the site root redirects to, e.g. `4.1` */
  latestLine: string;
  /** The MAJOR.MINOR lines that are deployed */
  deployedLines: string[];
};

/**
 * The marker in the application page (`apps/tissuumaps/index.html`) that the
 * custom HTML replaces; a deployed page without it fails the assembly, since
 * the custom HTML would otherwise silently be missing
 */
const customHTMLMarker = "<!-- GHA_CUSTOM_HTML -->";

/**
 * Parses a semantic version as created by changesets (without build metadata)
 *
 * @param version - The version string, e.g. `4.1.0-rc.1`
 * @returns The parsed version
 */
export function parseVersion(version: string): Version {
  const match = /^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$/.exec(version);
  if (!match) {
    throw new Error(`${version} is not a semantic version`);
  }
  const [, major, minor, patch, prerelease] = match;
  return {
    major: Number(major),
    minor: Number(minor),
    patch: Number(patch),
    prerelease: (prerelease?.split(".") ?? []).map((part) =>
      /^\d+$/.test(part) ? Number(part) : part,
    ),
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

/**
 * The MAJOR.MINOR line of a version, which names its directory on the site
 *
 * @param version - The parsed version
 * @returns The line, e.g. `4.1` for `4.1.2` and for `4.1.0-rc.1`
 */
export function formatVersionLine(version: Version): string {
  return `${version.major}.${version.minor}`;
}

/**
 * Compares two versions by semantic version precedence
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

/**
 * Compares two lists of prerelease identifiers by semantic version precedence
 *
 * An empty list (a stable version) ranks above any prerelease; otherwise the
 * identifiers are compared one by one, and a shorter list ranks below a longer
 * one with the same prefix.
 *
 * @param a - The first list of identifiers
 * @param b - The second list of identifiers
 * @returns A negative number if `a` ranks below `b`, a positive number if above, and `0` if equal
 */
export function comparePrereleases(
  a: (string | number)[],
  b: (string | number)[],
): number {
  if (a.length === 0 || b.length === 0) {
    return b.length - a.length;
  }
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const result = comparePrereleaseIdentifiers(a[i]!, b[i]!);
    if (result !== 0) {
      return result;
    }
  }
  return a.length - b.length;
}

/**
 * Compares two prerelease identifiers by semantic version precedence:
 * numerically if both are numeric, numeric ones below alphanumeric ones,
 * alphanumeric ones in ASCII order
 *
 * @param a - The first identifier
 * @param b - The second identifier
 * @returns A negative number if `a` ranks below `b`, a positive number if above, and `0` if equal
 */
export function comparePrereleaseIdentifiers(
  a: string | number,
  b: string | number,
): number {
  if (typeof a === "number" && typeof b === "number") {
    return a - b;
  }
  if (typeof a === "number" || typeof b === "number") {
    return typeof a === "number" ? -1 : 1;
  }
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * Selects the versions to deploy from the released versions
 *
 * The versions are grouped by MAJOR.MINOR line. Of every line, the highest
 * deployable stable version is the candidate, or the highest deployable
 * prerelease while the line has no deployable stable version; a line without
 * any deployable version has none. A prerelease candidate is only deployed if
 * it ranks above the highest stable candidate, so abandoned prerelease lines
 * do not accumulate.
 *
 * @param versions - The released versions, in any order
 * @param deployable - Whether a version can be deployed at all (has its assets)
 * @returns `versionsToDeploy`, at most one per line in ascending order, and
 * `latestVersion`, the version the site root redirects to: the highest stable
 * version to deploy, or the highest prerelease to deploy while there is none
 * (`null` without any version to deploy)
 */
export function selectVersionsToDeploy(
  versions: Version[],
  deployable: (version: Version) => boolean,
): { versionsToDeploy: Version[]; latestVersion: Version | null } {
  const lines = Map.groupBy(
    versions.toSorted(compareVersions),
    formatVersionLine,
  );
  const candidateLineVersions = [...lines.values()].flatMap((lineVersions) => {
    const deployableLineVersions = lineVersions.filter(deployable);
    const candidateLineVersion =
      deployableLineVersions.findLast((version) => !isPrerelease(version)) ??
      deployableLineVersions.at(-1);
    return candidateLineVersion === undefined ? [] : [candidateLineVersion];
  });
  const highestStableVersion = candidateLineVersions.findLast(
    (version) => !isPrerelease(version),
  );
  const versionsToDeploy = candidateLineVersions.filter(
    (version) =>
      !isPrerelease(version) ||
      highestStableVersion === undefined ||
      compareVersions(version, highestStableVersion) > 0,
  );
  const latestVersion = highestStableVersion ?? versionsToDeploy.at(-1) ?? null;
  return { versionsToDeploy, latestVersion };
}

/**
 * Resolves where a path of the site should redirect to
 *
 * Decides by the first segment of the path after the site prefix: `docs`
 * redirects to the latest line's documentation, keeping the rest of the path;
 * a deployed line never redirects (a missing page there is a genuine 404);
 * anything else, including the site root and the lines that are not deployed,
 * redirects to the latest line's root. Paths outside the prefix never
 * redirect.
 *
 * The redirect pages run this function in the browser, inlined through its
 * source text (with the types already stripped by Node), so it must not refer
 * to anything outside of itself.
 *
 * @param pathname - The requested path, e.g. `window.location.pathname`
 * @param site - The site description
 * @returns The target path, or `null` if the path should not redirect
 */
export function resolveRedirectTarget(
  pathname: string,
  site: Site,
): string | null {
  if (!pathname.startsWith(site.prefix)) {
    return null;
  }
  const relativePath = pathname.slice(site.prefix.length);
  const relativePathSeparatorIndex = relativePath.indexOf("/");
  const firstRelativePathSegment =
    relativePathSeparatorIndex === -1
      ? relativePath
      : relativePath.slice(0, relativePathSeparatorIndex);
  const relativePathAfterFirstSegment =
    relativePathSeparatorIndex === -1
      ? "/"
      : relativePath.slice(relativePathSeparatorIndex);
  if (firstRelativePathSegment === "docs") {
    return (
      site.prefix + site.latestLine + "/docs" + relativePathAfterFirstSegment
    );
  }
  if (site.deployedLines.includes(firstRelativePathSegment)) {
    return null;
  }
  return site.prefix + site.latestLine + "/";
}

/**
 * Renders a redirect page, which redirects through the inlined
 * {@link resolveRedirectTarget} (keeping the query string and the hash) and,
 * without JavaScript, through a meta refresh to `redirectTarget`
 *
 * @param site - The site description
 * @param redirectTarget - The meta refresh target, or `null` for none (the 404
 * page, which also serves genuine 404s)
 * @returns The HTML of the page
 */
export function renderRedirectPage(
  site: Site,
  redirectTarget: string | null,
): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="robots" content="noindex" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    ${redirectTarget === null ? "" : `<meta http-equiv="refresh" content="3; url=${redirectTarget}" />`}
    <title>TissUUmaps</title>
    <script>
      var SITE = ${JSON.stringify(site)};
      ${resolveRedirectTarget.toString()}
      var redirectTarget = resolveRedirectTarget(window.location.pathname, SITE);
      if (redirectTarget !== null) {
        window.location.replace(redirectTarget + window.location.search + window.location.hash);
      }
    </script>
  </head>
  <body>
    <p>${redirectTarget === null ? "This page does not exist." : "Redirecting to the current TissUUmaps release&hellip;"}</p>
    <p><a href="${site.prefix}${site.latestLine}/">Latest release</a></p>
  </body>
</html>
`;
}

/**
 * Lists the releases of a repository with the names of their assets
 *
 * @param repository - The repository, as `owner/name`
 * @returns All releases, including those of the packages
 */
export function fetchReleases(
  repository: string,
): { tag: string; assets: string[] }[] {
  // prettier-ignore
  return execFileSync(
    "gh",
    [
      "api",
      "--paginate", `repos/${repository}/releases`,
      "--jq", ".[] | {tag: .tag_name, assets: [.assets[].name]}",
    ],
    { encoding: "utf8" },
  )
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => JSON.parse(line) as { tag: string; assets: string[] });
}

/**
 * Assembles the site into `outDir` (see above)
 *
 * @param options.repository - The repository of the releases, as `owner/name`;
 * its name is the site's path prefix
 * @param options.releases - The repository's releases (see {@link fetchReleases})
 * @param options.outDir - The output directory, which must not exist yet
 * @param options.customHtml - The HTML to insert into every deployed application page
 * @param options.assetsDir - A directory holding the assets; downloaded from
 * the releases if omitted
 */
export function assemblePages(options: {
  repository: string;
  releases: { tag: string; assets: string[] }[];
  outDir: string;
  customHtml: string;
  assetsDir?: string;
}): void {
  const { repository, releases, outDir, customHtml, assetsDir } = options;
  const assetsOf = (version: Version): [app: string, docs: string] => [
    `tissuumaps-${formatVersion(version)}.zip`,
    `tissuumaps-${formatVersion(version)}-docs.zip`,
  ];
  const versions: Version[] = [];
  const versionsWithoutAssets = new Set<Version>();
  for (const release of releases) {
    if (!release.tag.startsWith("tissuumaps@")) {
      continue;
    }
    const version = parseVersion(release.tag.slice("tissuumaps@".length));
    versions.push(version);
    if (!assetsOf(version).every((asset) => release.assets.includes(asset))) {
      console.warn(`${release.tag} lacks its site assets`);
      versionsWithoutAssets.add(version);
    }
  }
  const { versionsToDeploy, latestVersion } = selectVersionsToDeploy(
    versions,
    (version) => !versionsWithoutAssets.has(version),
  );
  if (!latestVersion) {
    throw new Error("No release has its assets yet; nothing to deploy");
  }
  const site: Site = {
    prefix: `/${repository.split("/")[1]}/`,
    latestLine: formatVersionLine(latestVersion),
    deployedLines: versionsToDeploy.map(formatVersionLine),
  };
  mkdirSync(outDir);
  const zipDir = assetsDir ?? mkdtempSync(join(tmpdir(), "tissuumaps-site-"));
  for (const versionToDeploy of versionsToDeploy) {
    const [appAsset, docsAsset] = assetsOf(versionToDeploy);
    if (!assetsDir) {
      // prettier-ignore
      execFileSync("gh", [
        "release", "download", `tissuumaps@${formatVersion(versionToDeploy)}`,
        "--repo", repository,
        "--dir", zipDir,
        "--pattern", appAsset,
        "--pattern", docsAsset,
      ]);
    }
    const lineDir = join(outDir, formatVersionLine(versionToDeploy));
    // prettier-ignore
    execFileSync("unzip", [
      "-q", join(zipDir, appAsset),
      "-d", lineDir,
    ]);
    // prettier-ignore
    execFileSync("unzip", [
      "-q", join(zipDir, docsAsset),
      "-d", join(lineDir, "docs"),
    ]);
    const page = join(lineDir, "index.html");
    const html = readFileSync(page, "utf8");
    if (!html.includes(customHTMLMarker)) {
      throw new Error(`${page} has no ${customHTMLMarker} marker`);
    }
    writeFileSync(
      page,
      html.replace(customHTMLMarker, () => customHtml),
    );
    console.log(
      `deployed ${formatVersion(versionToDeploy)} as ${formatVersionLine(versionToDeploy)}`,
    );
  }
  for (const alias of ["", "docs/"]) {
    mkdirSync(join(outDir, alias), { recursive: true });
    writeFileSync(
      join(outDir, alias, "index.html"),
      renderRedirectPage(
        site,
        resolveRedirectTarget(site.prefix + alias, site),
      ),
    );
  }
  writeFileSync(join(outDir, "404.html"), renderRedirectPage(site, null));
  console.log(`latest: ${site.latestLine}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const repository = process.env.GITHUB_REPOSITORY!;
  assemblePages({
    repository,
    releases: fetchReleases(repository),
    outDir: "_site",
    customHtml: readFileSync(".github/pages/custom.html", "utf8"),
  });
}
