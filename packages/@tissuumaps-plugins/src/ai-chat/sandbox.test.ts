// @vitest-environment jsdom
// jsdom runs no scripts in the sandbox, which has no Web Workers anyway: these
// tests check the parts around it. What the sandbox blocks is checked in a
// real browser, with the skill sandbox-check.
import { afterEach, describe, expect, it } from "vitest";

import { runScript, sandboxPage } from "./sandbox";

const run = { code: "export default () => 1", args: {}, project: {} };
const answer = () => Promise.resolve(null);

afterEach(() => {
  document.body.replaceChildren();
});

describe("sandboxPage", () => {
  it("allows no network, no other scripts and no eval", () => {
    expect(sandboxPage()).toContain(
      `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' blob:; worker-src blob:">`,
    );
  });

  it("has one script element, which nothing in the code can end early", () => {
    const page = sandboxPage();
    const endTag = "</" + "script";
    expect(page.split("<script>")).toHaveLength(2);
    expect(page.split(endTag)).toHaveLength(2);
    expect(page.endsWith(`${endTag}>`)).toBe(true);
  });

  it("imports the script only with the import() written as text, which Vite does not rewrite", () => {
    const page = sandboxPage();
    expect(page.split("import(")).toHaveLength(2);
    expect(page).toContain("((url) => import(url));");
  });
});

describe("runScript", () => {
  it("runs the script in an iframe that may only run scripts, and removes it when the time is over", async () => {
    const running = runScript(run, answer, { timeLimit: 50 });
    const iframe = document.querySelector("iframe")!;
    expect(iframe.getAttribute("sandbox")).toBe("allow-scripts");
    expect(iframe.srcdoc).toBe(sandboxPage());
    await expect(running).rejects.toThrow(
      "the script took longer than 0.05 seconds, and was stopped",
    );
    expect(document.querySelector("iframe")).toBeNull();
  });

  it("listens only to its own iframe", async () => {
    const running = runScript(run, answer, { timeLimit: 50 });
    // a message from the page itself, which could pretend to be the sandbox
    window.dispatchEvent(
      new MessageEvent("message", {
        data: { type: "failed", error: "fake" },
        source: window,
      }),
    );
    await expect(running).rejects.toThrow("took longer");
  });

  it("stops at once when stopped, and does not start when already stopped", async () => {
    const stopper = new AbortController();
    const running = runScript(run, answer, {
      timeLimit: 10_000,
      signal: stopper.signal,
    });
    stopper.abort(new Error("Stop"));
    await expect(running).rejects.toThrow("Stop");
    expect(document.querySelector("iframe")).toBeNull();
    await expect(
      runScript(run, answer, { timeLimit: 10_000, signal: stopper.signal }),
    ).rejects.toThrow("Stop");
    expect(document.querySelector("iframe")).toBeNull();
  });
});
