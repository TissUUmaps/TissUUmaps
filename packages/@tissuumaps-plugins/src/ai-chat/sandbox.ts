// Scripts of skills run in a sandbox. An iframe with the "sandbox" attribute
// gets its own, opaque origin: it cannot reach the page, its storage (where
// the API keys may be) or its stores. A Content Security Policy in it allows
// no network, no other scripts than its own, and no eval; the Web Worker that
// runs the script inherits that policy. The worker keeps the page responsive
// and can be stopped at any time, by removing the iframe. The script gets
// only what the panel sends it: its arguments, a copy of the project state,
// and the data it asks for through the object `tm`, which the panel answers
// with copies. Python uses the same page, with its own worker and policy
// (see python-sandbox.ts).

/** What a script asks the panel for, through `tm` */
export type ScriptQuestion =
  | { kind: "tables" }
  | { kind: "columns"; table: string; matrix?: string }
  | { kind: "values"; table: string; column: string }
  | { kind: "file"; path: string }
  | {
      kind: "save_column";
      table: string;
      column: string;
      values: unknown;
      ids?: unknown;
    }
  | { kind: "download"; name: unknown; text: unknown };

/** A script to run */
export type ScriptRun = {
  /** The code: an ES module whose default export is a function of `tm` */
  code: string;
  /** What the script gets as `tm.args` */
  args: unknown;
  /** What the script gets as `tm.project`: a copy of the project state */
  project: unknown;
};

/** What a script gave back */
export type ScriptOutcome = {
  /** What the script returned, as text; "" if it returned nothing */
  result?: string;
  /** Why the script failed */
  error?: string;
  /** What the script logged with console.log and the like */
  logs: string[];
  /**
   * The charts that Python code drew, for the user only: each image, which
   * should be a PNG (the panel checks it), and what it shows in words, for
   * the model
   */
  charts?: { png: ArrayBuffer; description: string }[];
};

/** The policy of the sandbox: no network, no other scripts, no eval */
const contentSecurityPolicy =
  "default-src 'none'; script-src 'unsafe-inline' blob:; worker-src blob:";

/**
 * The Web Worker that runs a script; it is sent to the sandbox as text, so
 * it must use nothing from outside itself
 *
 * @param load - Imports a module from its URL
 */
function workerMain(load: (url: string) => Promise<unknown>): void {
  self.onmessage = (
    event: MessageEvent<{ run: ScriptRun; port: MessagePort }>,
  ) => {
    self.onmessage = null;
    const { run, port } = event.data;
    const pending = new Map<
      number,
      { resolve: (value: unknown) => void; reject: (error: Error) => void }
    >();
    let nextId = 0;
    port.onmessage = (
      answer: MessageEvent<{ id: number; value?: unknown; error?: string }>,
    ) => {
      const { id, value, error } = answer.data;
      const question = pending.get(id);
      pending.delete(id);
      if (error === undefined) {
        question?.resolve(value);
      } else {
        question?.reject(new Error(error));
      }
    };
    const ask = (question: unknown) =>
      new Promise<unknown>((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve, reject });
        port.postMessage({ type: "question", id, question });
      });
    // the result over several lines, a logged value on one
    const toText = (value: unknown, indent?: number): string =>
      typeof value === "string"
        ? value
        : value === undefined
          ? ""
          : (JSON.stringify(
              value,
              (_key, item: unknown) =>
                ArrayBuffer.isView(item)
                  ? Array.from(item as unknown as ArrayLike<unknown>)
                  : item,
              indent,
              // e.g. a function, which JSON cannot hold
            ) ?? `(${typeof value})`);
    const logs: string[] = [];
    const log = (...values: unknown[]) => {
      logs.push(values.map((value) => toText(value)).join(" "));
    };
    globalThis.console = {
      ...globalThis.console,
      log,
      info: log,
      warn: log,
      error: log,
      debug: log,
    };
    const tm = Object.freeze({
      args: run.args,
      project: run.project,
      tables: () => ask({ kind: "tables" }),
      columns: (table: string, matrix?: string) =>
        ask({ kind: "columns", table, matrix }),
      read: (table: string, column: string) =>
        ask({ kind: "values", table, column }),
      readFile: (path: string) => ask({ kind: "file", path }),
      saveColumn: (
        table: string,
        column: string,
        values: unknown,
        options?: { ids?: unknown },
      ) =>
        ask({
          kind: "save_column",
          table,
          column,
          values,
          ...(options?.ids === undefined ? {} : { ids: options.ids }),
        }),
      download: (name: unknown, text: unknown) =>
        ask({ kind: "download", name, text }),
      progress: (text: unknown) =>
        port.postMessage({ type: "progress", text: toText(text) }),
    });
    void (async () => {
      try {
        const url = URL.createObjectURL(
          new Blob([run.code], { type: "text/javascript" }),
        );
        const script = (await load(url)) as { default?: unknown };
        if (typeof script.default !== "function") {
          throw new Error(
            'the script must "export default" a function of tm, e.g. export default async function (tm) { ... }',
          );
        }
        const result: unknown = await (
          script.default as (tools: typeof tm) => unknown
        )(tm);
        port.postMessage({ type: "done", result: toText(result, 2), logs });
      } catch (error) {
        port.postMessage({
          type: "failed",
          error: error instanceof Error ? error.message : String(error),
          logs,
        });
      }
    })();
  };
}

/**
 * The page of the sandbox, which starts the worker when the panel sends a
 * script; it is sent as text, so it must use nothing from outside itself
 *
 * @param workerSource - The code of the worker
 */
function sandboxMain(workerSource: string): void {
  const fail = (message: string) =>
    window.parent.postMessage({ type: "failed", error: message }, "*");
  window.addEventListener(
    "message",
    (event: MessageEvent<{ type?: string; run?: unknown }>) => {
      const port = event.ports[0];
      if (
        event.source !== window.parent ||
        event.data.type !== "run" ||
        port === undefined
      ) {
        return;
      }
      try {
        const url = URL.createObjectURL(
          new Blob([workerSource], { type: "text/javascript" }),
        );
        const worker = new Worker(url);
        worker.onerror = (error) => {
          error.preventDefault();
          fail(error.message || "the script could not start");
        };
        worker.postMessage({ run: event.data.run, port }, [port]);
      } catch (error) {
        fail(String(error));
      }
    },
  );
  window.parent.postMessage({ type: "ready" }, "*");
}

/**
 * The HTML of the sandbox page
 *
 * @param options - Another worker and policy than those of JavaScript, e.g.
 * those of Python; the worker gets the script and a port to the panel when
 * the panel sends them
 * @returns The page, for the srcdoc of the iframe
 */
export function sandboxPage(options?: {
  worker: (load: (url: string) => Promise<unknown>) => void;
  policy: string;
}): string {
  const { worker = workerMain, policy = contentSecurityPolicy } = options ?? {};
  // import() written as text: in development, Vite rewrites an import() of a
  // variable in the code it serves, with a function the sandbox does not have
  const workerSource = `(${worker.toString()})((url) => import(url));`;
  // "<" escaped, so that nothing in the code can end the script element
  const source = JSON.stringify(workerSource).replaceAll("<", "\\x3c");
  // the closing tag in two pieces: the app is built into one HTML file, with
  // this code in a script element, which the whole tag would end
  const endScript = "</" + "script>";
  return (
    '<!doctype html><meta charset="utf-8">' +
    `<meta http-equiv="Content-Security-Policy" content="${policy}">` +
    `<script>(${sandboxMain.toString()})(${source});${endScript}`
  );
}

/**
 * Runs a script in a new sandbox, which is removed afterwards
 *
 * @param run - The script, its arguments and the project state
 * @param answer - Answers the questions of the script, e.g. the values of a column
 * @param options - The longest time the script may run, in milliseconds;
 * what to do with its progress; an optional abort signal, which stops it
 * @returns What the script returned or why it failed, and what it logged
 * @throws If the script takes too long, or is stopped
 */
export function runScript(
  run: ScriptRun,
  answer: (question: ScriptQuestion) => Promise<unknown>,
  options: {
    timeLimit: number;
    onProgress?: (text: string) => void;
    signal?: AbortSignal;
  },
): Promise<ScriptOutcome> {
  const { timeLimit, onProgress, signal } = options;
  if (signal?.aborted) {
    return Promise.reject(signal.reason as Error);
  }
  return new Promise((resolve, reject) => {
    const iframe = document.createElement("iframe");
    iframe.setAttribute("sandbox", "allow-scripts");
    iframe.style.display = "none";
    iframe.srcdoc = sandboxPage();
    const channel = new MessageChannel();

    const finish = () => {
      clearTimeout(timer);
      window.removeEventListener("message", onPageMessage);
      signal?.removeEventListener("abort", onAbort);
      channel.port1.close();
      // also stops the worker, whatever it is doing
      iframe.remove();
    };
    const onAbort = () => {
      finish();
      reject(signal!.reason as Error);
    };
    const timer = setTimeout(() => {
      finish();
      reject(
        new Error(
          `the script took longer than ${timeLimit / 1000} seconds, and was stopped`,
        ),
      );
    }, timeLimit);

    const onPageMessage = (
      event: MessageEvent<{ type?: string; error?: string }>,
    ) => {
      if (event.source !== iframe.contentWindow) {
        return;
      }
      if (event.data.type === "ready") {
        // an opaque origin cannot be named: "*", but only to this iframe
        iframe.contentWindow!.postMessage({ type: "run", run }, "*", [
          channel.port2,
        ]);
      } else if (event.data.type === "failed") {
        finish();
        resolve({ error: event.data.error ?? "unknown error", logs: [] });
      }
    };
    channel.port1.onmessage = (
      event: MessageEvent<{
        type: string;
        id?: number;
        question?: ScriptQuestion;
        text?: string;
        result?: string;
        error?: string;
        logs?: string[];
      }>,
    ) => {
      const { type, id, question, text, result, error, logs = [] } = event.data;
      if (type === "question" && question !== undefined) {
        // also a value that cannot be copied to the sandbox is an error
        answer(question)
          .then((value) => channel.port1.postMessage({ id, value }))
          .catch((problem: unknown) =>
            channel.port1.postMessage({
              id,
              error:
                problem instanceof Error ? problem.message : String(problem),
            }),
          );
      } else if (type === "progress") {
        onProgress?.(text ?? "");
      } else if (type === "done") {
        finish();
        resolve({ result: result ?? "", logs });
      } else if (type === "failed") {
        finish();
        resolve({ error: error ?? "unknown error", logs });
      }
    };

    window.addEventListener("message", onPageMessage);
    signal?.addEventListener("abort", onAbort, { once: true });
    document.body.append(iframe);
  });
}
