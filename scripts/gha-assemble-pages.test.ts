import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { runInNewContext } from "node:vm";

import {
  type Site,
  assemblePages,
  comparePrereleaseIdentifiers,
  comparePrereleases,
  compareVersions,
  formatVersion,
  formatVersionLine,
  isPrerelease,
  parseVersion,
  renderRedirectPage,
  resolveRedirectTarget,
  selectVersionsToDeploy,
} from "./gha-assemble-pages.ts";

const parse = parseVersion;

void test("parses stable and prerelease versions", () => {
  assert.deepEqual(parse("4.0.1"), {
    major: 4,
    minor: 0,
    patch: 1,
    prerelease: [],
  });
  assert.deepEqual(parse("4.0.0-beta.10"), {
    major: 4,
    minor: 0,
    patch: 0,
    prerelease: ["beta", 10],
  });
  assert.deepEqual(parse("4.0.0-0a.1-b.x").prerelease, ["0a", "1-b", "x"]);
  assert.throws(() => parseVersion("v4.0.0"), /not a semantic version/);
  assert.throws(() => parseVersion("4.0"), /not a semantic version/);
});

void test("formats versions", () => {
  assert.equal(formatVersion(parse("4.0.0-beta.0")), "4.0.0-beta.0");
  assert.equal(formatVersion(parse("4.1.0")), "4.1.0");
  assert.equal(isPrerelease(parse("4.0.0-rc.1")), true);
  assert.equal(isPrerelease(parse("4.0.0")), false);
});

void test("compares prerelease identifiers", () => {
  assert.ok(comparePrereleaseIdentifiers(2, 10) < 0);
  assert.equal(comparePrereleaseIdentifiers(1, 1), 0);
  assert.ok(comparePrereleaseIdentifiers(99, "alpha") < 0);
  assert.ok(comparePrereleaseIdentifiers("alpha", 0) > 0);
  assert.ok(comparePrereleaseIdentifiers("alpha", "beta") < 0);
  assert.equal(comparePrereleaseIdentifiers("rc", "rc"), 0);
});

void test("compares lists of prerelease identifiers", () => {
  assert.equal(comparePrereleases([], []), 0);
  assert.ok(comparePrereleases([], ["rc", 1]) > 0);
  assert.ok(comparePrereleases(["rc", 1], []) < 0);
  assert.ok(comparePrereleases(["beta"], ["beta", 0]) < 0);
  assert.ok(comparePrereleases(["beta", 2], ["beta", 10]) < 0);
  assert.equal(comparePrereleases(["rc", 1], ["rc", 1]), 0);
});

void test("orders versions by semantic version precedence", () => {
  const ordered = [
    "4.0.0-alpha",
    "4.0.0-alpha.1",
    "4.0.0-beta.0",
    "4.0.0-beta.1",
    "4.0.0-beta.2",
    "4.0.0-beta.10",
    "4.0.0-rc.1",
    "4.0.0",
    "4.0.1",
    "4.0.9",
    "4.1.0-rc.1",
    "4.1.0",
    "5.0.0",
  ].map(parse);
  for (let i = 0; i < ordered.length; i++) {
    for (let j = 0; j < ordered.length; j++) {
      const expected = Math.sign(i - j);
      assert.equal(
        Math.sign(compareVersions(ordered[i]!, ordered[j]!)),
        expected,
        `${formatVersion(ordered[i]!)} vs ${formatVersion(ordered[j]!)}`,
      );
    }
  }
});

/** Runs {@link selectVersionsToDeploy} on version strings */
function select(
  versions: string[],
  deployable: (version: string) => boolean = () => true,
) {
  const { versionsToDeploy, latestVersion } = selectVersionsToDeploy(
    versions.map(parseVersion),
    (version) => deployable(formatVersion(version)),
  );
  return {
    deployed: versionsToDeploy.map(formatVersion),
    latest: latestVersion === null ? null : formatVersion(latestVersion),
  };
}

void test("names lines by MAJOR.MINOR", () => {
  assert.equal(formatVersionLine(parseVersion("4.1.2")), "4.1");
  assert.equal(formatVersionLine(parseVersion("4.10.0-rc.1")), "4.10");
});

void test("deploys the highest stable version of each line", () => {
  assert.deepEqual(select(["4.0.1", "4.1.0", "4.0.0", "4.1.2", "4.1.1"]), {
    deployed: ["4.0.1", "4.1.2"],
    latest: "4.1.2",
  });
});

void test("deploys a line's prerelease only while the line has no stable version", () => {
  assert.deepEqual(select(["4.0.0-beta.0", "4.0.0-beta.1"]), {
    deployed: ["4.0.0-beta.1"],
    latest: "4.0.0-beta.1",
  });
  assert.deepEqual(
    select(["4.1.0-beta.0", "4.1.0-rc.1", "4.1.0", "4.1.1-rc.1"]).deployed,
    ["4.1.0"],
  );
});

void test("the root goes to the latest prerelease only while nothing stable exists", () => {
  assert.deepEqual(select(["4.0.3", "4.1.0-rc.1"]), {
    deployed: ["4.0.3", "4.1.0-rc.1"],
    latest: "4.0.3",
  });
});

void test("drops abandoned prerelease lines", () => {
  assert.deepEqual(
    select(["4.0.0-rc.1", "4.1.0", "4.2.0-rc.1", "4.3.0", "4.4.0-rc.1"]),
    {
      deployed: ["4.1.0", "4.3.0", "4.4.0-rc.1"],
      latest: "4.3.0",
    },
  );
});

void test("falls back to deployable versions within a line", () => {
  const deployable = (v: string) => !["4.0.2", "4.1.0-rc.2"].includes(v);
  assert.deepEqual(
    select(["4.0.1", "4.0.2", "4.1.0-rc.1", "4.1.0-rc.2"], deployable).deployed,
    ["4.0.1", "4.1.0-rc.1"],
  );
});

void test("does not deploy lines without a deployable version", () => {
  const deployable = (v: string) => !["4.1.0", "4.2.0-rc.1"].includes(v);
  assert.deepEqual(
    select(["4.0.0", "4.1.0", "4.2.0-rc.1", "4.3.0-rc.1"], deployable),
    {
      deployed: ["4.0.0", "4.3.0-rc.1"],
      latest: "4.0.0",
    },
  );
});

void test("handles no versions", () => {
  assert.deepEqual(select([]), { deployed: [], latest: null });
});

const site: Site = {
  prefix: "/TissUUmaps/",
  latestLine: "4.1",
  deployedLines: ["4.1", "4.2"],
};

void test("redirects the root, unknown paths and lines that are not deployed to the latest line", () => {
  assert.equal(resolveRedirectTarget("/TissUUmaps/", site), "/TissUUmaps/4.1/");
  assert.equal(
    resolveRedirectTarget("/TissUUmaps/live/", site),
    "/TissUUmaps/4.1/",
  );
  assert.equal(
    resolveRedirectTarget("/TissUUmaps/4.0/docs/intro/", site),
    "/TissUUmaps/4.1/",
  );
  assert.equal(
    resolveRedirectTarget("/TissUUmaps/4.1.2/docs/", site),
    "/TissUUmaps/4.1/",
  );
});

void test("redirects the docs alias to the latest documentation", () => {
  assert.equal(
    resolveRedirectTarget("/TissUUmaps/docs/", site),
    "/TissUUmaps/4.1/docs/",
  );
  assert.equal(
    resolveRedirectTarget("/TissUUmaps/docs/api/@tissuumaps/core/", site),
    "/TissUUmaps/4.1/docs/api/@tissuumaps/core/",
  );
});

void test("never redirects within deployed lines", () => {
  assert.equal(resolveRedirectTarget("/TissUUmaps/4.1/missing/", site), null);
  assert.equal(resolveRedirectTarget("/TissUUmaps/4.2/", site), null);
});

void test("does nothing outside the prefix", () => {
  assert.equal(resolveRedirectTarget("/Other/4.0/", site), null);
  assert.equal(resolveRedirectTarget("/TissUUmaps", site), null);
});

void test("renders redirect pages with a meta refresh only where given", () => {
  const redirecting = renderRedirectPage(site, "/TissUUmaps/4.1/");
  assert.match(
    redirecting,
    /<meta http-equiv="refresh" content="3; url=\/TissUUmaps\/4\.1\/" \/>/,
  );
  assert.match(redirecting, /Redirecting to the current TissUUmaps release/);
  assert.match(redirecting, /var SITE = \{"prefix":"\/TissUUmaps\/"/);
  assert.match(redirecting, /function resolveRedirectTarget\(/);
  const notFound = renderRedirectPage(site, null);
  assert.doesNotMatch(notFound, /http-equiv="refresh"/);
  assert.match(notFound, /This page does not exist\./);
});

/**
 * Creates the assets of the given releases in a temporary directory and
 * returns the options for assembling a site from them
 */
function fixture(
  releases: { tag: string; assets?: string[] }[],
  appPage = "<html><!-- GHA_CUSTOM_HTML --></html>",
) {
  const dir = mkdtempSync(join(tmpdir(), "assemble-pages-test-"));
  const sources = { app: join(dir, "app"), docs: join(dir, "docs") };
  mkdirSync(sources.app);
  mkdirSync(sources.docs);
  writeFileSync(join(sources.app, "index.html"), appPage);
  writeFileSync(join(sources.docs, "index.html"), "docs");
  const assetsDir = join(dir, "assets");
  mkdirSync(assetsDir);
  const withAssets = releases.map((release) => {
    const version = release.tag.replace(/^tissuumaps@/, "");
    const assets = release.assets ?? [
      `tissuumaps-${version}.zip`,
      `tissuumaps-${version}-docs.zip`,
    ];
    for (const asset of assets) {
      const source = asset.endsWith("-docs.zip") ? sources.docs : sources.app;
      execFileSync("zip", ["-q", join(assetsDir, asset), "index.html"], {
        cwd: source,
      });
    }
    return { tag: release.tag, assets };
  });
  const out = join(dir, "site");
  return {
    dir,
    out,
    options: {
      repository: "TissUUmaps/TissUUmaps",
      releases: withAssets,
      outDir: out,
      customHtml: "<script>matomo</script>",
      assetsDir,
    },
  };
}

void test("assembles the site from the release assets", () => {
  const { dir, out, options } = fixture([
    { tag: "tissuumaps@4.0.0-rc.1" },
    { tag: "tissuumaps@4.1.0" },
    { tag: "tissuumaps@4.1.1" },
    { tag: "tissuumaps@4.2.0", assets: [] },
    { tag: "tissuumaps@4.3.0-beta.0" },
    { tag: "@tissuumaps/core@0.1.0", assets: [] },
  ]);
  try {
    assemblePages(options);
    const read = (path: string) => readFileSync(join(out, path), "utf8");
    // Deployed lines, with the custom HTML in the application page only
    assert.equal(
      read("4.1/index.html"),
      "<html><script>matomo</script></html>",
    );
    assert.equal(read("4.1/docs/index.html"), "docs");
    assert.ok(existsSync(join(out, "4.3/index.html")));
    // Lines that are not deployed get no directory; the packages' releases are ignored
    assert.ok(!existsSync(join(out, "4.0")));
    assert.ok(!existsSync(join(out, "4.2")));
    // The root and the docs alias redirect to the latest stable line
    assert.match(read("index.html"), /url=\/TissUUmaps\/4\.1\/"/);
    assert.match(read("docs/index.html"), /url=\/TissUUmaps\/4\.1\/docs\/"/);
    // The 404 page only redirects through its script
    assert.doesNotMatch(read("404.html"), /http-equiv="refresh"/);
    const site = JSON.parse(
      /var SITE = (.*);/.exec(read("404.html"))![1]!,
    ) as Site;
    assert.deepEqual(site, {
      prefix: "/TissUUmaps/",
      latestLine: "4.1",
      deployedLines: ["4.1", "4.3"],
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

void test("the redirect pages run the inlined redirect rules", () => {
  const { dir, out, options } = fixture([
    { tag: "tissuumaps@4.0.0-rc.1" },
    { tag: "tissuumaps@4.1.0" },
  ]);
  try {
    assemblePages(options);
    const page = readFileSync(join(out, "404.html"), "utf8");
    const script = /<script>([\s\S]*?)<\/script>/.exec(page)![1]!;
    const redirect = (pathname: string) => {
      const targets: string[] = [];
      const location = {
        pathname,
        search: "?project=x",
        hash: "#a",
        replace: (url: string) => targets.push(url),
      };
      runInNewContext(script, { window: { location } });
      return targets[0] ?? null;
    };
    assert.equal(
      redirect("/TissUUmaps/docs/intro/"),
      "/TissUUmaps/4.1/docs/intro/?project=x#a",
    );
    assert.equal(redirect("/TissUUmaps/4.0/"), "/TissUUmaps/4.1/?project=x#a");
    assert.equal(redirect("/TissUUmaps/4.1/missing/"), null);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

void test("fails while no release has its assets, leaving the output untouched", () => {
  const { dir, out, options } = fixture([
    { tag: "tissuumaps@4.0.0", assets: ["tissuumaps-4.0.0.zip"] },
  ]);
  try {
    assert.throws(() => assemblePages(options), /No release has its assets/);
    assert.ok(!existsSync(out));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

void test("fails if an application page has no custom HTML marker", () => {
  const { dir, options } = fixture(
    [{ tag: "tissuumaps@4.0.0" }],
    "<html></html>",
  );
  try {
    assert.throws(
      () => assemblePages(options),
      /has no <!-- GHA_CUSTOM_HTML -->/,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

void test("fails if the output directory exists already", () => {
  const { dir, out, options } = fixture([{ tag: "tissuumaps@4.0.0" }]);
  try {
    mkdirSync(out);
    assert.throws(() => assemblePages(options), /EEXIST/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
