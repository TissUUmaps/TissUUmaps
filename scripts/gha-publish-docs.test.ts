import assert from "node:assert/strict";
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

import { publishDocs, shouldPublish } from "./gha-publish-docs.ts";

void test("publishes new lines, and newer or equal versions", () => {
  assert.equal(shouldPublish("4.1.0", undefined), true);
  assert.equal(shouldPublish("4.1.0-rc.1", undefined), true);
  assert.equal(shouldPublish("4.1.1", "4.1.0"), true);
  assert.equal(shouldPublish("4.1.0", "4.1.0"), true);
  assert.equal(shouldPublish("4.1.0-rc.2", "4.1.0-rc.1"), true);
  assert.equal(shouldPublish("4.1.0", "4.1.0-rc.2"), true);
});

void test("never replaces a higher version", () => {
  assert.equal(shouldPublish("4.1.0", "4.1.1"), false);
  assert.equal(shouldPublish("4.1.0-rc.1", "4.1.0-rc.2"), false);
});

void test("never replaces a stable version with a prerelease", () => {
  assert.equal(shouldPublish("4.1.1-rc.0", "4.1.0"), false);
});

/** Creates documentation and a releases directory in a temporary directory */
function fixture() {
  const dir = mkdtempSync(join(tmpdir(), "publish-docs-test-"));
  const docsDir = join(dir, "docs");
  mkdirSync(join(docsDir, "api"), { recursive: true });
  writeFileSync(join(docsDir, "intro.md"), "# Intro");
  writeFileSync(join(docsDir, "api", "index.md"), "# API");
  const releasesDir = join(dir, "releases");
  return { dir, docsDir, releasesDir };
}

void test("fails on an invalid version", () => {
  const { dir, docsDir, releasesDir } = fixture();
  try {
    for (const version of ["4.1", "v4.1.0", "4.1.0+build"]) {
      assert.throws(
        () => publishDocs({ version, docsDir, releasesDir }),
        /not a semantic version/,
      );
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

void test("publishes the documentation of a release into its line", () => {
  const { dir, docsDir, releasesDir } = fixture();
  try {
    assert.equal(publishDocs({ version: "4.1.0", docsDir, releasesDir }), true);
    const read = (path: string) =>
      readFileSync(join(releasesDir, path), "utf8");
    assert.deepEqual(JSON.parse(read("4.1/release.json")), {
      version: "4.1.0",
    });
    assert.equal(read("4.1/docs/intro.md"), "# Intro");
    assert.equal(read("4.1/docs/api/index.md"), "# API");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

void test("replaces the line's previous release", () => {
  const { dir, docsDir, releasesDir } = fixture();
  try {
    publishDocs({ version: "4.1.0", docsDir, releasesDir });
    rmSync(join(docsDir, "intro.md"));
    publishDocs({ version: "4.1.1", docsDir, releasesDir });
    assert.ok(!existsSync(join(releasesDir, "4.1/docs/intro.md")));
    assert.deepEqual(
      JSON.parse(readFileSync(join(releasesDir, "4.1/release.json"), "utf8")),
      { version: "4.1.1" },
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

void test("leaves the line untouched if it should not publish", () => {
  const { dir, docsDir, releasesDir } = fixture();
  try {
    publishDocs({ version: "4.1.0", docsDir, releasesDir });
    writeFileSync(join(docsDir, "intro.md"), "# Changed");
    assert.equal(
      publishDocs({ version: "4.1.1-rc.0", docsDir, releasesDir }),
      false,
    );
    assert.equal(
      readFileSync(join(releasesDir, "4.1/docs/intro.md"), "utf8"),
      "# Intro",
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
