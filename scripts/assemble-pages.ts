/**
 * Assembles the versioned GitHub Pages site from the release assets
 *
 * Every release of the application (`tissuumaps@<version>`) carries the built
 * site of that version as the asset `tissuumaps-<version>-site.zip`. Only the
 * newest versions are deployed and the others redirect to them (see
 * `lib/retention.ts`); the site root redirects to the latest version and
 * `docs/` to its documentation. While no version is deployed, the site is a
 * copy of the fallback directory instead (the current application at its
 * root and its documentation under `docs/`, built from the default branch).
 *
 * Usage: `node scripts/assemble-pages.ts [options]`
 *   --out <dir>                output directory (default: `_site`)
 *   --prefix <path>            site path prefix (default: `/<repository name>/`)
 *   --repository <owner/repo>  repository of the releases (default: `$GITHUB_REPOSITORY`)
 *   --fallback-dir <dir>       the site to serve while no version is deployed
 *   --releases <file>          releases as JSON instead of the GitHub API (for testing)
 *   --assets-dir <dir>         directory holding the zips instead of downloading them (for testing)
 */
import { execFileSync } from "node:child_process";
import {
  cpSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "node:util";

import { type Site, resolveRedirect } from "./lib/redirect.ts";
import { planRetention } from "./lib/retention.ts";
import { type Version, formatVersion, parseVersion } from "./lib/semver.ts";

const tagOf = (version: Version) => `tissuumaps@${formatVersion(version)}`;
const assetOf = (version: Version) =>
  `tissuumaps-${formatVersion(version)}-site.zip`;

function run(command: string, args: string[]): string {
  return execFileSync(command, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  });
}

const { values: options } = parseArgs({
  options: {
    out: { type: "string", default: "_site" },
    prefix: { type: "string" },
    repository: { type: "string", default: process.env.GITHUB_REPOSITORY },
    "fallback-dir": { type: "string" },
    releases: { type: "string" },
    "assets-dir": { type: "string" },
  },
});
if (!options.repository) {
  throw new Error("--repository or GITHUB_REPOSITORY is required");
}
const repository = options.repository;
const prefix = options.prefix ?? `/${repository.split("/").at(-1)}/`;
if (!prefix.startsWith("/") || !prefix.endsWith("/")) {
  throw new Error(`--prefix must start and end with "/": ${prefix}`);
}
const outDir = options.out;

// The application's releases; one without its site asset (a failed upload)
// cannot be deployed and only gets a redirect
type Release = { tag: string; draft: boolean; assets: string[] };
const releases = options.releases
  ? (JSON.parse(readFileSync(options.releases, "utf8")) as Release[])
  : run("gh", [
      "api",
      "--paginate",
      `repos/${repository}/releases`,
      "--jq",
      ".[] | {tag: .tag_name, draft: .draft, assets: [.assets[].name]}",
    ])
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as Release);
const versions: Version[] = [];
const withoutAsset = new Set<Version>();
for (const release of releases) {
  const match = /^tissuumaps@(.+)$/.exec(release.tag);
  if (!match || release.draft) {
    continue;
  }
  const version = parseVersion(match[1]!);
  if (!version) {
    console.warn(`${release.tag} is not a semantic version tag; ignored`);
    continue;
  }
  versions.push(version);
  if (!release.assets.includes(assetOf(version))) {
    console.warn(`${release.tag} has no ${assetOf(version)}`);
    withoutAsset.add(version);
  }
}
const { retained, redirects, latest } = planRetention(
  versions,
  (version) => !withoutAsset.has(version),
);

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, ".nojekyll"), "");
if (latest === null && options["fallback-dir"]) {
  cpSync(options["fallback-dir"], outDir, { recursive: true });
  console.log(`no version deployed: serving ${options["fallback-dir"]}`);
} else {
  deployVersions();
  writeRedirectPages();
  console.log(
    `latest: ${latest ? formatVersion(latest) : "none"}; redirects: ${Object.keys(redirects).join(", ") || "none"}`,
  );
}

/** Unpacks each retained version's site asset into `<out>/<version>/` */
function deployVersions() {
  const downloadDir = options["assets-dir"]
    ? null
    : mkdtempSync(join(tmpdir(), "tissuumaps-site-"));
  const assetsDir = options["assets-dir"] ?? downloadDir!;
  try {
    for (const version of retained) {
      if (downloadDir) {
        run("gh", [
          "release",
          "download",
          tagOf(version),
          "--repo",
          repository,
          "--pattern",
          assetOf(version),
          "--dir",
          downloadDir,
        ]);
      }
      const versionDir = join(outDir, formatVersion(version));
      mkdirSync(versionDir);
      run("unzip", ["-q", join(assetsDir, assetOf(version)), "-d", versionDir]);
      console.log(`deployed ${formatVersion(version)}`);
    }
  } finally {
    if (downloadDir) {
      rmSync(downloadDir, { recursive: true, force: true });
    }
  }
}

/**
 * Writes the redirect pages: the root, the docs alias and the superseded
 * versions always redirect (also without JavaScript, via meta refresh); the
 * 404 page only redirects where the rules say so, since it also serves genuine
 * 404s. The rules are inlined from `lib/redirect.ts`.
 */
function writeRedirectPages() {
  const site: Site = {
    prefix,
    latest: latest && formatVersion(latest),
    retained: retained.map(formatVersion),
    redirects,
  };
  const releasesUrl = `https://github.com/${repository}/releases`;
  const page = readFileSync(
    join(import.meta.dirname, "templates", "redirect.html"),
    "utf8",
  )
    .replace("__SITE_JSON__", () => JSON.stringify(site))
    .replace("__REDIRECT_JS__", () =>
      stripTypeScriptTypes(
        readFileSync(join(import.meta.dirname, "lib", "redirect.ts"), "utf8"),
      )
        .replace(/^export /gm, "")
        .replace(/\/\*\*[\s\S]*?\*\/\n?/g, "")
        .replace(/^[ \t]*\n/gm, ""),
    )
    .replace(
      "__LINKS__",
      site.latest === null
        ? `<a href="${releasesUrl}">All releases</a>`
        : `<a href="${prefix}${site.latest}/">Latest release</a> &middot; <a href="${releasesUrl}">All releases</a>`,
    );
  // A page's meta refresh follows the same rules as its script (without the
  // query string, which only the script can carry over); `null` for 404.html
  const render = (pagePath: string | null) => {
    const target =
      pagePath === null ? null : resolveRedirect(prefix + pagePath, site);
    return page
      .replace(
        "__META_REFRESH__",
        target
          ? `<meta http-equiv="refresh" content="3; url=${target}" />`
          : "",
      )
      .replace(
        "__MESSAGE__",
        pagePath === null
          ? "This page does not exist."
          : target
            ? "Redirecting to the current TissUUmaps release&hellip;"
            : "No TissUUmaps release has been deployed yet.",
      );
  };
  writeFileSync(join(outDir, "index.html"), render(""));
  writeFileSync(join(outDir, "404.html"), render(null));
  for (const alias of ["docs", ...Object.keys(redirects)]) {
    mkdirSync(join(outDir, alias));
    writeFileSync(join(outDir, alias, "index.html"), render(`${alias}/`));
  }
}
