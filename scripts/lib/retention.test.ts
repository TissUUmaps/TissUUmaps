import assert from "node:assert/strict";
import { test } from "node:test";

import { planRetention } from "./retention.ts";
import { formatVersion, parseVersion } from "./semver.ts";

function plan(...versions: string[]) {
  const retention = planRetention(versions.map((v) => parseVersion(v)!));
  return {
    retained: retention.retained.map(formatVersion),
    redirects: retention.redirects,
    latest: retention.latest === null ? null : formatVersion(retention.latest),
  };
}

void test("keeps the highest version of each line and redirects the others", () => {
  assert.deepEqual(plan("4.0.1", "4.1.0", "4.0.0", "4.1.2", "4.1.1"), {
    retained: ["4.0.1", "4.1.2"],
    redirects: { "4.0.0": "4.0.1", "4.1.0": "4.1.2", "4.1.1": "4.1.2" },
    latest: "4.1.2",
  });
});

void test("the root goes to the latest prerelease only while nothing stable exists", () => {
  assert.deepEqual(plan("4.0.0-beta.0", "4.0.0-beta.1"), {
    retained: ["4.0.0-beta.1"],
    redirects: { "4.0.0-beta.0": "4.0.0-beta.1" },
    latest: "4.0.0-beta.1",
  });
  assert.deepEqual(plan("4.0.3", "4.1.0-rc.1").latest, "4.0.3");
});

void test("prereleases redirect to the stable version above them, which keeps a prerelease above itself", () => {
  assert.deepEqual(
    plan("4.1.0-beta.0", "4.1.0-rc.1", "4.1.0", "4.1.1-rc.1", "4.1.1-rc.2"),
    {
      retained: ["4.1.0", "4.1.1-rc.2"],
      redirects: {
        "4.1.0-beta.0": "4.1.0",
        "4.1.0-rc.1": "4.1.0",
        "4.1.1-rc.1": "4.1.1-rc.2",
      },
      latest: "4.1.0",
    },
  );
});

void test("abandoned prerelease lines redirect to the next stable version", () => {
  assert.deepEqual(
    plan("4.0.0-rc.1", "4.1.0", "4.1.1-rc.1", "4.2.0", "4.3.0-rc.1"),
    {
      retained: ["4.1.0", "4.2.0", "4.3.0-rc.1"],
      redirects: { "4.0.0-rc.1": "4.1.0", "4.1.1-rc.1": "4.2.0" },
      latest: "4.2.0",
    },
  );
});

void test("undeployable versions only ever redirect", () => {
  const deployable = (v: string) => !["4.0.2", "4.1.0-rc.1"].includes(v);
  const retention = planRetention(
    ["4.0.1", "4.0.2", "4.1.0-rc.1", "4.1.0-rc.2"].map((v) => parseVersion(v)!),
    (version) => deployable(formatVersion(version)),
  );
  assert.deepEqual(retention.retained.map(formatVersion), [
    "4.0.1",
    "4.1.0-rc.2",
  ]);
  assert.deepEqual(retention.redirects, {
    "4.0.2": "4.0.1",
    "4.1.0-rc.1": "4.1.0-rc.2",
  });
});

void test("handles no versions", () => {
  assert.deepEqual(plan(), { retained: [], redirects: {}, latest: null });
});
