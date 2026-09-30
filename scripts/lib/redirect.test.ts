import assert from "node:assert/strict";
import { test } from "node:test";

import { type Site, resolveRedirect } from "./redirect.ts";

const site: Site = {
  prefix: "/TissUUmaps/",
  latest: "4.0.1",
  retained: ["4.0.1", "4.1.0-rc.2"],
  redirects: {
    "4.0.0-rc.1": "4.0.1",
    "4.0.0": "4.0.1",
    "4.1.0-rc.1": "4.1.0-rc.2",
  },
};

void test("redirects the root and unknown paths to the latest version", () => {
  assert.equal(resolveRedirect("/TissUUmaps/", site), "/TissUUmaps/4.0.1/");
  assert.equal(
    resolveRedirect("/TissUUmaps/live/", site),
    "/TissUUmaps/4.0.1/",
  );
  assert.equal(
    resolveRedirect("/TissUUmaps/live-dev/foo", site),
    "/TissUUmaps/4.0.1/",
  );
});

void test("redirects superseded versions keeping the rest of the path", () => {
  assert.equal(
    resolveRedirect("/TissUUmaps/4.0.0/", site),
    "/TissUUmaps/4.0.1/",
  );
  assert.equal(
    resolveRedirect("/TissUUmaps/4.0.0/docs/intro/", site),
    "/TissUUmaps/4.0.1/docs/intro/",
  );
  assert.equal(
    resolveRedirect("/TissUUmaps/4.1.0-rc.1/docs/api", site),
    "/TissUUmaps/4.1.0-rc.2/docs/api",
  );
});

void test("redirects the docs alias to the latest documentation", () => {
  assert.equal(
    resolveRedirect("/TissUUmaps/docs/", site),
    "/TissUUmaps/4.0.1/docs/",
  );
  assert.equal(
    resolveRedirect("/TissUUmaps/docs/api/@tissuumaps/core/", site),
    "/TissUUmaps/4.0.1/docs/api/@tissuumaps/core/",
  );
});

void test("never redirects within deployed versions", () => {
  assert.equal(resolveRedirect("/TissUUmaps/4.0.1/missing/", site), null);
  assert.equal(resolveRedirect("/TissUUmaps/4.1.0-rc.2/", site), null);
});

void test("does nothing outside the prefix or without a deployed version", () => {
  assert.equal(resolveRedirect("/Other/4.0.0/", site), null);
  assert.equal(resolveRedirect("/TissUUmaps", site), null);
  assert.equal(
    resolveRedirect("/TissUUmaps/", { ...site, latest: null, retained: [] }),
    null,
  );
});
