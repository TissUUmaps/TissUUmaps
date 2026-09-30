import assert from "node:assert/strict";
import { test } from "node:test";

import {
  compareVersions,
  formatVersion,
  isPrerelease,
  parseVersion,
} from "./semver.ts";

const parse = (version: string) => parseVersion(version)!;

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
  assert.equal(parseVersion("v4.0.0"), null);
  assert.equal(parseVersion("4.0"), null);
  assert.equal(parseVersion("4.0.0rc1"), null);
  assert.equal(parseVersion("4.0.1+build.2"), null);
  assert.equal(parseVersion("4.0.0-a..b"), null);
  assert.equal(parseVersion("4.0.0-"), null);
  assert.equal(parseVersion("04.0.0"), null);
  assert.equal(parseVersion("4.0.0-beta.01"), null);
  assert.deepEqual(parse("4.0.0-0a.1-b.x").prerelease, ["0a", "1-b", "x"]);
});

void test("formats versions", () => {
  assert.equal(formatVersion(parse("4.0.0-beta.0")), "4.0.0-beta.0");
  assert.equal(formatVersion(parse("4.1.0")), "4.1.0");
  assert.equal(isPrerelease(parse("4.0.0-rc.1")), true);
  assert.equal(isPrerelease(parse("4.0.0")), false);
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
