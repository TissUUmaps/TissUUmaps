// @vitest-environment jsdom
// jsdom runs no scripts in the sandbox, and has no WebAssembly for Pyodide:
// these tests check the parts around it. What the sandbox blocks is checked
// in a real browser, with a Python script that tries to get out.
import { afterEach, describe, expect, it, vi } from "vitest";

import { lockFile } from "./python-files";
import { type PythonRun, startPythonKernel } from "./python-sandbox";
import type { ScriptOutcome } from "./sandbox";

const files = { get: vi.fn(() => Promise.resolve(new ArrayBuffer(0))) };

afterEach(() => {
  document.body.replaceChildren();
  files.get.mockClear();
});

describe("startPythonKernel", () => {
  it("starts Python in an iframe that may only run scripts, with no network and no eval", async () => {
    const controller = new AbortController();
    const starting = startPythonKernel(files, { signal: controller.signal });
    const iframe = document.querySelector("iframe")!;
    expect(iframe.getAttribute("sandbox")).toBe("allow-scripts");
    expect(iframe.srcdoc).toContain(
      `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' blob: 'wasm-unsafe-eval'; worker-src blob:">`,
    );
    // Pyodide is imported only with the import() written as text, which Vite
    // does not rewrite
    expect(iframe.srcdoc.split("import(")).toHaveLength(2);
    controller.abort(new Error("stopped"));
    await expect(starting).rejects.toThrow("stopped");
  });

  it("gives no files before its sandbox is ready, and listens only to its own iframe", async () => {
    const controller = new AbortController();
    const starting = startPythonKernel(files, { signal: controller.signal });
    window.postMessage({ type: "ready" }, "*");
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(files.get).not.toHaveBeenCalled();
    controller.abort(new Error("stopped"));
    await expect(starting).rejects.toThrow("stopped");
    expect(document.querySelector("iframe")).toBeNull();
  });

  it("does not start when it is stopped already", async () => {
    const controller = new AbortController();
    controller.abort(new Error("stopped"));
    await expect(
      startPythonKernel(files, { signal: controller.signal }),
    ).rejects.toThrow("stopped");
    expect(document.querySelector("iframe")).toBeNull();
  });
});

/** A list of packages of Pyodide, with scipy */
const lock = JSON.stringify({
  packages: {
    scipy: { name: "scipy", imports: ["scipy"] },
    numpy: { name: "numpy", imports: ["numpy"] },
  },
});

/** The error of Pyodide for a package of its own that is not loaded */
const scipyError = [
  "Traceback (most recent call last):",
  '  File "<script>", line 19, in <module>',
  '  File "/skill/_points2regions.py", line 3, in <module>',
  "    import scipy.sparse as sp",
  "ModuleNotFoundError: No module named 'scipy'",
  "The module 'scipy' is included in the Pyodide distribution, but it is not installed.",
  "You can install it by calling:",
  '  await micropip.install("scipy") in Python, or',
  '  await pyodide.loadPackage("scipy") in JavaScript',
].join("\n");

/**
 * Starts a kernel whose worker is played by the test: it gets the messages
 * of the kernel, and answers each run with the next outcome
 */
async function fakeKernel(
  outcomes: ((port: MessagePort, id: number) => Promise<ScriptOutcome>)[],
) {
  const seen: { type: string; run?: PythonRun; packages?: string[] }[] = [];
  const kernelFiles = {
    get: vi.fn((name: string) =>
      Promise.resolve(
        name === lockFile
          ? new TextEncoder().encode(lock).buffer
          : new ArrayBuffer(0),
      ),
    ),
  };
  const starting = startPythonKernel(kernelFiles);
  const iframe = document.querySelector("iframe")!;
  const port = await new Promise<MessagePort>((resolve) => {
    vi.spyOn(iframe.contentWindow!, "postMessage").mockImplementation(((
      _message: unknown,
      _origin: string,
      transfer: MessagePort[],
    ) => resolve(transfer[0]!)) as typeof window.postMessage);
    window.dispatchEvent(
      new MessageEvent("message", {
        data: { type: "ready" },
        source: iframe.contentWindow,
      }),
    );
  });
  port.onmessage = async ({ data }: MessageEvent) => {
    const message = data as {
      type: string;
      id: number;
      run?: PythonRun;
      packages?: string[];
    };
    if (message.type === "answer") {
      return;
    }
    seen.push(message);
    const value =
      message.type === "run" ? await outcomes.shift()!(port, message.id) : null;
    port.postMessage({ type: "done", id: message.id, value });
  };
  const kernel = await starting;
  return { kernel, seen };
}

/** Asks the kernel a question, as the code of the worker does */
function askFrom(port: MessagePort, question: object): Promise<void> {
  return new Promise((resolve) => {
    const listener = ({ data }: MessageEvent) => {
      if ((data as { type: string }).type === "answer") {
        port.removeEventListener("message", listener);
        resolve();
      }
    };
    port.addEventListener("message", listener);
    port.postMessage({ type: "question", id: 1000, question });
  });
}

describe("packages that the code needs", () => {
  const run: PythonRun = {
    code: "async def main(tm):\n    return 1",
    args: {},
    project: null,
    files: {
      "_points2regions.py": new TextEncoder().encode("import scipy").buffer,
    },
  };
  const options = { timeLimit: 60_000 };

  it("loads a package of Pyodide that the code needed, and runs it again once", async () => {
    const { kernel, seen } = await fakeKernel([
      () => Promise.resolve({ error: scipyError, logs: [] }),
      () => Promise.resolve({ result: "1", logs: ["done"] }),
    ]);
    const outcome = await kernel.run(run, () => Promise.resolve(null), options);
    expect(outcome.error).toBeUndefined();
    expect(outcome.result).toBe("1");
    expect(outcome.logs).toEqual([
      "(The code needed scipy, a package of Pyodide that its imports did not show: it was loaded, and the code ran again.)",
      "done",
    ]);
    const runs = seen.filter(({ type }) => type === "run");
    expect(runs.map(({ run: given }) => given?.packages)).toEqual([
      undefined,
      ["scipy"],
    ]);
    // the files of the skill went with both runs
    expect(
      runs.map(
        ({ run: given }) => given?.files?.["_points2regions.py"]?.byteLength,
      ),
    ).toEqual([12, 12]);
  });

  it("does not run again code that saved something, but loads the package", async () => {
    const { kernel, seen } = await fakeKernel([
      async (port) => {
        await askFrom(port, {
          kind: "save_column",
          table: "cells",
          column: "c",
          values: [],
        });
        return { error: scipyError, logs: [] };
      },
    ]);
    const outcome = await kernel.run(run, () => Promise.resolve({}), options);
    expect(seen.map(({ type }) => type)).toEqual(["start", "run", "load"]);
    expect(seen[2]!.packages).toEqual(["scipy"]);
    expect(outcome.error).toContain(
      "ModuleNotFoundError: No module named 'scipy'\nscipy is a package of Pyodide",
    );
    expect(outcome.error).toContain("it is loaded now: run the code again");
    expect(outcome.error).not.toContain("micropip.install");
  });

  it("says clearly that a package is not in Pyodide, without micropip", async () => {
    const { kernel, seen } = await fakeKernel([
      () =>
        Promise.resolve({
          error:
            "ModuleNotFoundError: No module named 'torch'\nThe module 'torch' is not included in the Pyodide distribution.\nYou can try to install it with micropip.install(\"torch\")",
          logs: [],
        }),
    ]);
    const outcome = await kernel.run(run, () => Promise.resolve(null), options);
    expect(seen.filter(({ type }) => type === "run")).toHaveLength(1);
    expect(outcome.error).toBe(
      "ModuleNotFoundError: No module named 'torch'\n'torch' is not a package of Pyodide, the Python of the sandbox, which can only import the packages of Pyodide and the files of the skill: micropip and pip cannot install anything here, since the sandbox has no network. If 'torch' is a module of the skill, import it as the files of the skill are laid out (their folder is on the path of Python); else do without it, e.g. with numpy, scipy, pandas or scikit-learn.",
    );
  });
});
