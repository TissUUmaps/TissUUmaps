import { maxCharts } from "./charts";
import { type PythonFiles, coreFiles, lockFile } from "./python-files";
import {
  explainMissingModule,
  missingModule,
  pyodidePackages,
} from "./python-imports";
import {
  type ScriptOutcome,
  type ScriptQuestion,
  type ScriptRun,
  sandboxPage,
} from "./sandbox";

// Python runs in Pyodide, in a sandbox like that of JavaScript: an iframe
// with sandbox="allow-scripts" and a Content Security Policy without any
// network, with a Web Worker in it. The policy also allows WebAssembly, which
// Pyodide is made of, but still no eval. The worker never downloads anything:
// the panel gives it the files of Pyodide (see python-files.ts), and while
// Pyodide loads itself or a package, its fetch asks the panel for them, and
// can get nothing else. Starting Python takes seconds, so the sandbox stays,
// as a "kernel", for the next runs, each with fresh names; it is removed to
// stop a run.

/** No network, no eval; WebAssembly may be compiled, which Pyodide needs */
const contentSecurityPolicy =
  "default-src 'none'; script-src 'unsafe-inline' blob: 'wasm-unsafe-eval'; worker-src blob:";

/** The folder of the files of a skill, whose modules its code can import */
const skillFolder = "/skill";

/**
 * The Python side of the sandbox: runs the code of a script with fresh
 * names, and gives it tm, like tm of JavaScript but with numpy arrays
 */
const driver = [
  "import ast",
  "import importlib",
  "import inspect",
  "import json",
  "import keyword",
  "import os",
  "import shutil",
  "import sys",
  "",
  "import js",
  "from pyodide.ffi import to_js",
  "",
  "# figures are drawn into images, never into a window; plt.show() does nothing",
  'os.environ["MPLBACKEND"] = "Agg"',
  "import warnings",
  "",
  'warnings.filterwarnings("ignore", message=".*non-interactive.*")',
  "",
  "",
  "class _Tm:",
  '    """What a script gets: the data of the project, always as copies"""',
  "",
  "    def __init__(self, ask, progress, args, project):",
  "        self._ask = ask",
  "        self._progress = progress",
  "        self.args = args",
  "        self.project = project",
  "",
  "    async def _question(self, **question):",
  "        return await self._ask(to_js(question, dict_converter=js.Object.fromEntries))",
  "",
  "    async def tables(self):",
  '        return (await self._question(kind="tables")).to_py()',
  "",
  "    async def columns(self, table, matrix=None):",
  '        question = {"kind": "columns", "table": table}',
  "        if matrix is not None:",
  '            question["matrix"] = matrix',
  "        return (await self._question(**question)).to_py()",
  "",
  "    async def read(self, table, column):",
  "        import numpy as np",
  "",
  '        values = await self._question(kind="values", table=table, column=column)',
  "        if js.ArrayBuffer.isView(values):",
  "            return np.asarray(values.to_py())",
  "        return np.asarray(values.to_py(), dtype=object)",
  "",
  "    async def read_file(self, path):",
  '        return await self._question(kind="file", path=path)',
  "",
  "    async def save_column(self, table, column, values, ids=None):",
  "        import numpy as np",
  "",
  "        array = np.asarray(values).ravel()",
  '        if array.dtype.kind in "iuf":',
  "            # numbers: one copy, as 64-bit floats",
  '            data = js.Float64Array.new(to_js(array.astype("float64").tobytes()).buffer)',
  "        else:",
  "            data = to_js([None if value is None else str(value) for value in array.tolist()])",
  '        question = {"kind": "save_column", "table": table, "column": column, "values": data}',
  "        if ids is not None:",
  '            question["ids"] = to_js(np.asarray(ids).ravel().tolist())',
  "        return (await self._question(**question)).to_py()",
  "",
  "    async def download(self, name, text):",
  '        """Gives the user a file of text, e.g. a table as CSV"""',
  "        if not isinstance(text, str):",
  '            raise TypeError("give the text of the file, e.g. frame.to_csv()")',
  '        return (await self._question(kind="download", name=name, text=text)).to_py()',
  "",
  "    def progress(self, text):",
  "        self._progress(str(text))",
  "",
  "",
  "def _text(value):",
  "    if value is None:",
  '        return ""',
  "    if isinstance(value, str):",
  "        return value",
  '    pandas = sys.modules.get("pandas")',
  "    if pandas is not None and isinstance(value, (pandas.DataFrame, pandas.Series)):",
  "        return value.to_string()",
  '    if hasattr(value, "tolist"):',
  "        value = value.tolist()",
  "    return json.dumps(",
  '        value, indent=2, default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item)',
  "    )",
  "",
  "",
  `SKILL = "${skillFolder}"`,
  "# where a skill whose folder is a package can be imported by its name",
  'PACKAGES = SKILL + "_package"',
  "",
  "",
  "def _of_skill(module):",
  '    file = getattr(module, "__file__", None) or ""',
  '    folders = [str(path) for path in getattr(module, "__path__", None) or []]',
  "    return any(path.startswith(SKILL) for path in [file, *folders])",
  "",
  "",
  "def _clear_skill():",
  '    """Forgets the files and the modules of the skill of the run before"""',
  "    for name, module in list(sys.modules.items()):",
  "        if _of_skill(module):",
  "            del sys.modules[name]",
  "    sys.path[:] = [path for path in sys.path if not path.startswith(SKILL)]",
  "    for path in list(sys.path_importer_cache):",
  "        if path.startswith(SKILL):",
  "            del sys.path_importer_cache[path]",
  "    shutil.rmtree(SKILL, ignore_errors=True)",
  "    shutil.rmtree(PACKAGES, ignore_errors=True)",
  "    importlib.invalidate_caches()",
  "",
  "",
  "def _skill_paths(has_files, script, package):",
  '    """Where the modules of the skill are: as Python does, the folder of the',
  "    script first, then the folder of the skill, and its src folder if any;",
  '    a skill whose folder is a package can also be imported by its name"""',
  "    if not has_files:",
  '        return "[]"',
  "    paths = []",
  '    if "/" in script:',
  "        paths.append(os.path.join(SKILL, os.path.dirname(script)))",
  "    paths.append(SKILL)",
  '    if os.path.isdir(f"{SKILL}/src"):',
  '        paths.append(f"{SKILL}/src")',
  "    if (",
  "        package.isidentifier()",
  "        and not keyword.iskeyword(package)",
  '        and os.path.isfile(f"{SKILL}/__init__.py")',
  "    ):",
  "        os.makedirs(PACKAGES, exist_ok=True)",
  "        os.symlink(SKILL, os.path.join(PACKAGES, package))",
  "        paths.append(PACKAGES)",
  "    return json.dumps(paths)",
  "",
  "",
  "def _find_local(name, paths):",
  '    """The file of a module of the skill (\'\' for a folder without',
  '    __init__.py), or None for a module that is not of the skill"""',
  '    parts = name.split(".")',
  "    for root in paths:",
  "        base = os.path.join(root, *parts)",
  '        if os.path.isfile(base + ".py"):',
  '            return base + ".py"',
  '        if os.path.isfile(os.path.join(base, "__init__.py")):',
  '            return os.path.join(base, "__init__.py")',
  "        if os.path.isdir(base):",
  '            return ""',
  "    return None",
  "",
  "",
  "def _imports_of(source, package):",
  '    """The modules that some code imports, also those in functions"""',
  "    try:",
  "        tree = ast.parse(source)",
  "    except SyntaxError:",
  "        return []",
  "    names = []",
  "    for node in ast.walk(tree):",
  "        if isinstance(node, ast.Import):",
  "            names.extend(alias.name for alias in node.names)",
  "        elif isinstance(node, ast.ImportFrom):",
  '            module = node.module or ""',
  "            if node.level:",
  '                parts = package.split(".") if package else []',
  "                parts = parts[: len(parts) - node.level + 1]",
  '                module = ".".join(parts + ([module] if module else []))',
  "            if module:",
  "                names.append(module)",
  '                names.extend(f"{module}.{alias.name}" for alias in node.names if alias.name != "*")',
  "    return names",
  "",
  "",
  "def _imports_needed(code, paths):",
  '    """The modules that the code imports, and the modules of the skill that it',
  '    imports, which are not of the skill: the packages of Pyodide to load"""',
  "    paths = json.loads(paths)",
  "    outside = set()",
  "    seen = set()",
  '    todo = [(code, "")]',
  "    while todo:",
  "        source, package = todo.pop()",
  "        for name in _imports_of(source, package):",
  '            parts = name.split(".")',
  "            if _find_local(parts[0], paths) is None:",
  "                outside.add(parts[0])",
  "                continue",
  "            for end in range(1, len(parts) + 1):",
  '                module = ".".join(parts[:end])',
  "                path = _find_local(module, paths)",
  "                if path and path not in seen:",
  "                    seen.add(path)",
  '                    with open(path, encoding="utf-8", errors="replace") as file:',
  "                        text = file.read()",
  '                    inside = module if path.endswith("__init__.py") else module.rpartition(".")[0]',
  "                    todo.append((text, inside))",
  "    # the other Python files of the skill too: the code may import one in a",
  "    # way that its imports do not show (e.g. importlib, or the skill as a package)",
  "    outside.update(_skill_imports(paths))",
  '    return "\\n".join(f"import {name}" for name in sorted(outside))',
  "",
  "",
  "_NOT_CODE = {",
  '    "tests", "test", "testing", "docs", "doc", "examples", "example",',
  '    "benchmarks", "benchmark", "notebooks", ".git", ".github", "__pycache__",',
  "}",
  "",
  "",
  "def _skill_imports(paths):",
  '    """The modules outside the skill that its Python files import, without',
  '    its tests, examples and docs"""',
  "    outside = set()",
  "    if not paths or not os.path.isdir(SKILL):",
  "        return outside",
  "    for root, folders, names in os.walk(SKILL):",
  "        folders[:] = [name for name in folders if name.lower() not in _NOT_CODE]",
  "        for name in names:",
  '            if not name.endswith(".py") or name in ("setup.py", "conftest.py"):',
  "                continue",
  '            if name.startswith("test_") or name.endswith("_test.py"):',
  "                continue",
  "            try:",
  '                with open(os.path.join(root, name), encoding="utf-8", errors="replace") as file:',
  "                    tree = ast.parse(file.read())",
  "            except (OSError, SyntaxError, ValueError):",
  "                continue",
  "            for node in ast.walk(tree):",
  "                if isinstance(node, ast.Import):",
  "                    found = [alias.name for alias in node.names]",
  "                elif isinstance(node, ast.ImportFrom) and not node.level and node.module:",
  "                    found = [node.module]",
  "                else:",
  "                    continue",
  "                for module in found:",
  '                    top = module.split(".")[0]',
  "                    if _find_local(top, paths) is None:",
  "                        outside.add(top)",
  "    return outside",
  "",
  "",
  `MAX_CHARTS = ${maxCharts}`,
  "",
  "",
  "def _close_figures():",
  '    pyplot = sys.modules.get("matplotlib.pyplot")',
  "    if pyplot is not None:",
  '        pyplot.close("all")',
  "",
  "",
  "def _is_number(text):",
  "    try:",
  '        float(text.replace("−", "-"))',
  "        return True",
  "    except ValueError:",
  "        return False",
  "",
  "",
  "def _labels(texts, words_only=False, most=30):",
  '    """e.g. " [B cells, T cells]"; for an axis, only names, not numbers"""',
  "    names = [text.get_text() for text in texts if text.get_text()]",
  "    if not names or len(names) > most or (words_only and all(map(_is_number, names))):",
  '        return ""',
  '    return " [" + ", ".join(names) + "]"',
  "",
  "",
  "def _describe_axes(axes):",
  '    """What a plot shows, in words, for the model, which does not see it"""',
  '    colorbar = getattr(axes, "_colorbar", None)',
  "    if colorbar is not None:",
  "        label = axes.get_ylabel() or axes.get_xlabel()",
  '        text = f"color scale {label!r}" if label else "color scale"',
  "        low, high = colorbar.norm.vmin, colorbar.norm.vmax",
  "        if low is not None and high is not None:",
  '            text += f" from {low:.4g} to {high:.4g}"',
  "        return text",
  "    parts = []",
  "    if axes.get_title():",
  "        parts.append(repr(axes.get_title()))",
  "    for name, label, limits, ticks in [",
  '        ("x", axes.get_xlabel(), axes.get_xlim(), axes.get_xticklabels()),',
  '        ("y", axes.get_ylabel(), axes.get_ylim(), axes.get_yticklabels()),',
  "    ]:",
  '        text = f"{name} axis {label!r}" if label else f"{name} axis"',
  '        parts.append(f"{text} from {limits[0]:.4g} to {limits[1]:.4g}{_labels(ticks, True)}")',
  "    held = []",
  "    if axes.lines:",
  '        held.append(f"{len(axes.lines)} line(s)")',
  "    bars = sum(len(container) for container in axes.containers)",
  "    if bars:",
  '        held.append(f"{bars} bar(s)")',
  "    if axes.collections:",
  '        held.append(f"{len(axes.collections)} collection(s) of marks, e.g. scatter points")',
  "    if axes.images:",
  '        held.append(f"{len(axes.images)} image(s), e.g. a heatmap")',
  "    if held:",
  '        parts.append("with " + ", ".join(held))',
  "    legend = axes.get_legend()",
  "    if legend is not None:",
  '        parts.append("legend" + _labels(legend.get_texts()))',
  '    return "; ".join(parts)',
  "",
  "",
  "def _charts():",
  '    """The figures that the code left open, as PNG images, each with what it',
  '    shows in words; then they are closed"""',
  "    import io",
  "",
  '    pyplot = sys.modules.get("matplotlib.pyplot")',
  "    if pyplot is None:",
  "        return []",
  "    charts = []",
  "    for number in pyplot.get_fignums()[:MAX_CHARTS]:",
  "        figure = pyplot.figure(number)",
  "        # at most about 2000 pixels on a side",
  "        dpi = min(100, 2000 / max(*figure.get_size_inches(), 1))",
  "        image = io.BytesIO()",
  "        # the warnings of matplotlib itself are not for the model",
  "        with warnings.catch_warnings():",
  '            warnings.simplefilter("ignore")',
  '            figure.savefig(image, format="png", dpi=dpi, bbox_inches="tight")',
  '        title = figure._suptitle.get_text() if figure._suptitle is not None else ""',
  "        plots = [_describe_axes(axes) for axes in figure.axes]",
  '        description = (f"{title!r}: " if title else "") + " | ".join(plots)',
  '        charts.append({"png": to_js(image.getvalue()), "description": description})',
  "    hidden = len(pyplot.get_fignums()) - len(charts)",
  "    _close_figures()",
  "    if hidden > 0:",
  '        print(f"{hidden} more figure(s) not shown: at most {MAX_CHARTS} per run")',
  "    return charts",
  "",
  "",
  "async def _run(code, ask, progress, args, project, paths, script):",
  "    # no figure of the run before",
  "    _close_figures()",
  "    for path in reversed(json.loads(paths)):",
  "        sys.path.insert(0, path)",
  '    namespace = {"__name__": "__script__"}',
  "    if script:",
  '        namespace["__file__"] = f"{SKILL}/{script}"',
  '    exec(compile(code, "<script>", "exec"), namespace)',
  '    main = namespace.get("main")',
  "    if not callable(main):",
  "        raise ValueError(",
  '            "the script must define a function main(tm), e.g. async def main(tm): ..."',
  "        )",
  "    result = main(_Tm(ask, progress, json.loads(args), json.loads(project)))",
  "    if inspect.isawaitable(result):",
  "        result = await result",
  "    return to_js(",
  '        {"result": _text(result), "charts": _charts()},',
  "        dict_converter=js.Object.fromEntries,",
  "    )",
].join("\n");

/** Python code to run, with the files of its skill, if any */
export type PythonRun = ScriptRun & {
  /** The files of the skill, by their path in it; its code can import them */
  files?: Record<string, ArrayBuffer>;
  /** The path of the script in the skill, e.g. "scripts/stats.py" */
  path?: string;
  /**
   * More modules whose packages of Pyodide to load first, e.g. one that the
   * code needed in a run before, though its imports did not show it
   */
  packages?: string[];
  /**
   * The name under which a skill whose folder is a Python package can be
   * imported, e.g. "points2regions" (see skillPackageName)
   */
  packageName?: string;
};

/** What the worker needs of Pyodide */
type Pyodide = {
  runPython: (code: string) => unknown;
  loadPackagesFromImports: (
    code: string,
    options: {
      messageCallback: (message: string) => void;
      errorCallback: (message: string) => void;
    },
  ) => Promise<unknown>;
  setStdout: (options: { batched: (line: string) => void }) => void;
  setStderr: (options: { batched: (line: string) => void }) => void;
  globals: {
    get: (name: string) => ((...args: unknown[]) => Promise<string>) & {
      destroy: () => void;
    };
  };
  FS: {
    mkdirTree: (path: string) => void;
    writeFile: (path: string, data: Uint8Array) => void;
  };
};

/** What the panel sends to the worker */
type ToWorker =
  | { type: "answer"; id: number; value?: unknown; error?: string }
  | {
      type: "start";
      id: number;
      files: Record<string, ArrayBuffer>;
      lock: string;
      driver: string;
    }
  | {
      type: "run";
      id: number;
      run: PythonRun;
    }
  | { type: "load"; id: number; packages: string[] };

/** What the worker sends to the panel */
type FromWorker =
  | { type: "file"; id: number; name: string }
  | { type: "question"; id: number; question: ScriptQuestion }
  | { type: "progress"; text: string }
  | { type: "running"; id: number }
  | { type: "done"; id: number; value?: ScriptOutcome }
  | { type: "failed"; id: number; error: string };

/**
 * The Web Worker that runs Python; it is sent to the sandbox as text, so it
 * must use nothing from outside itself
 *
 * @param load - Imports a module from its URL
 */
function workerMain(load: (url: string) => Promise<unknown>): void {
  // a made-up address: Pyodide asks for its files there, and gets them from
  // memory or from the panel
  const base = "https://pyodide.invalid/";
  const realFetch = self.fetch.bind(self);
  const files = new Map<string, ArrayBuffer>();
  const pending = new Map<
    number,
    { resolve: (value: unknown) => void; reject: (error: Error) => void }
  >();
  let nextId = 0;
  let port: MessagePort;
  let pyodide: Pyodide;

  const ask = (message: { type: string } & Record<string, unknown>) =>
    new Promise<unknown>((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      port.postMessage({ ...message, id });
    });

  // only while Pyodide loads itself or packages, never while a script runs
  const servePyodideFiles = (on: boolean) => {
    self.fetch = on
      ? async (url: RequestInfo | URL) => {
          const address =
            url instanceof Request
              ? url.url
              : url instanceof URL
                ? url.href
                : url;
          if (!address.startsWith(base)) {
            throw new TypeError(`Python has no network: ${address}`);
          }
          const name = address.slice(base.length);
          if (!files.has(name)) {
            files.set(name, (await ask({ type: "file", name })) as ArrayBuffer);
          }
          return new Response(files.get(name)!.slice(0), {
            headers: {
              "Content-Type": name.endsWith(".wasm")
                ? "application/wasm"
                : "application/octet-stream",
            },
          });
        }
      : realFetch;
  };

  const moduleUrl = (buffer: ArrayBuffer) =>
    URL.createObjectURL(new Blob([buffer], { type: "text/javascript" }));

  // the last lines of a Python error, from the first line of the script
  const shortError = (error: unknown) => {
    const lines = String(
      error instanceof Error ? error.message : error,
    ).trimEnd();
    const all = lines.split("\n");
    const from = all.findIndex((line) => line.includes('File "<script>"'));
    return (from >= 0 ? all.slice(from) : all.slice(-1)).join("\n");
  };

  async function start(message: Extract<ToWorker, { type: "start" }>) {
    for (const [name, buffer] of Object.entries(message.files)) {
      files.set(name, buffer);
    }
    const { loadPyodide } = (await load(
      moduleUrl(files.get("pyodide.mjs")!),
    )) as { loadPyodide: (options: object) => Promise<Pyodide> };
    const asm = (await load(moduleUrl(files.get("pyodide.asm.mjs")!))) as {
      default: unknown;
    };
    servePyodideFiles(true);
    try {
      pyodide = await loadPyodide({
        indexURL: base,
        packageBaseUrl: base,
        stdLibURL: `${base}python_stdlib.zip`,
        lockFileContents: message.lock,
        createPyodideModule: asm.default,
        stdout: () => {},
        stderr: () => {},
      });
    } finally {
      servePyodideFiles(false);
    }
    pyodide.runPython(message.driver);
  }

  // calls a function of the driver, which gives back text
  const callDriver = (name: string, ...values: unknown[]) => {
    const driverFunction = pyodide.globals.get(name);
    try {
      return driverFunction(...values) as unknown as string;
    } finally {
      driverFunction.destroy();
    }
  };

  async function run(message: Extract<ToWorker, { type: "run" }>) {
    const {
      code,
      args,
      project,
      files: scriptFiles,
      path = "",
      packages = [],
      packageName = "",
    } = message.run;
    const logs: string[] = [];
    // the files of a skill, which its code can import (the same folder as
    // skillFolder); nothing of the skill of the run before stays
    const folder = "/skill";
    callDriver("_clear_skill");
    for (const [file, buffer] of Object.entries(scriptFiles ?? {})) {
      const full = `${folder}/${file}`;
      pyodide.FS.mkdirTree(full.slice(0, full.lastIndexOf("/")));
      pyodide.FS.writeFile(full, new Uint8Array(buffer));
    }
    const paths = callDriver(
      "_skill_paths",
      scriptFiles !== undefined,
      path,
      packageName,
    );
    // the packages of Pyodide that the code imports, also through the
    // modules of the skill, from the panel
    servePyodideFiles(true);
    try {
      await pyodide.loadPackagesFromImports(
        callDriver("_imports_needed", code, paths) +
          packages.map((name) => `\nimport ${name}`).join(""),
        {
          messageCallback: () => {},
          errorCallback: (text) => logs.push(text),
        },
      );
    } finally {
      servePyodideFiles(false);
    }
    port.postMessage({ type: "running", id: message.id });
    pyodide.setStdout({ batched: (line) => logs.push(line) });
    pyodide.setStderr({ batched: (line) => logs.push(line) });
    const runner = pyodide.globals.get("_run");
    try {
      const { result, charts } = (await runner(
        code,
        (question: unknown) => ask({ type: "question", question }),
        (text: string) => port.postMessage({ type: "progress", text }),
        JSON.stringify(args ?? null),
        JSON.stringify(project ?? null),
        paths,
        path,
      )) as unknown as {
        result: string;
        charts: { png: Uint8Array; description: string }[];
      };
      return {
        result,
        logs,
        // each image in a buffer of its own, which the panel checks
        charts: charts.map(({ png, description }) => ({
          png: png.slice().buffer,
          description,
        })),
      };
    } catch (error) {
      return { error: shortError(error), logs };
    } finally {
      runner.destroy();
    }
  }

  // loads packages of Pyodide, e.g. one that code needed but did not show
  async function loadPackages(
    message: Extract<ToWorker, { type: "load" }>,
  ): Promise<null> {
    servePyodideFiles(true);
    try {
      await pyodide.loadPackagesFromImports(
        message.packages.map((name) => `import ${name}`).join("\n"),
        { messageCallback: () => {}, errorCallback: () => {} },
      );
    } finally {
      servePyodideFiles(false);
    }
    return null;
  }

  self.onmessage = (event: MessageEvent<{ port: MessagePort }>) => {
    self.onmessage = null;
    port = event.data.port;
    port.onmessage = async ({ data }: MessageEvent<ToWorker>) => {
      if (data.type === "answer") {
        const request = pending.get(data.id);
        pending.delete(data.id);
        if (data.error === undefined) {
          request?.resolve(data.value);
        } else {
          request?.reject(new Error(data.error));
        }
        return;
      }
      try {
        const value =
          data.type === "start"
            ? await start(data)
            : data.type === "load"
              ? await loadPackages(data)
              : await run(data);
        port.postMessage({ type: "done", id: data.id, value });
      } catch (error) {
        port.postMessage({
          type: "failed",
          id: data.id,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    };
  };
}

/** Python, started in a sandbox that stays for the next runs */
export type PythonKernel = {
  /**
   * Runs Python code: loads the packages it imports, then runs its function
   * main(tm). The time limit starts once the packages are loaded.
   *
   * @param run - The code (with its files, for a script of a skill), its
   * arguments and the project state
   * @param answer - Answers the questions of tm, e.g. the values of a column
   * @param options - The longest time the code may run, in milliseconds;
   * what to do with its progress, and when it starts to run; an optional
   * abort signal, which stops it
   * @returns What the code returned or why it failed, and what it printed
   * @throws If the code takes too long or is stopped: then the kernel is
   * stopped too
   */
  run: (
    run: PythonRun,
    answer: (question: ScriptQuestion) => Promise<unknown>,
    options: {
      timeLimit: number;
      onProgress?: (text: string) => void;
      onRunning?: () => void;
      signal?: AbortSignal;
    },
  ) => Promise<ScriptOutcome>;
  /** Stops Python, whatever it is doing, and removes its sandbox */
  stop: () => void;
  /** Whether it was stopped, and cannot run anything any more */
  isStopped: () => boolean;
};

/**
 * Starts Python in a new sandbox: gives it the files of Pyodide, which may
 * be downloaded first
 *
 * @param files - The files of Python, see python-files.ts
 * @param options - Optional abort signal, which stops the start
 * @returns The kernel
 * @throws If Python cannot start, e.g. a file cannot be downloaded, or if
 * it is stopped
 */
export async function startPythonKernel(
  files: PythonFiles,
  options?: { signal?: AbortSignal },
): Promise<PythonKernel> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const iframe = document.createElement("iframe");
  iframe.setAttribute("sandbox", "allow-scripts");
  iframe.style.display = "none";
  iframe.srcdoc = sandboxPage({
    worker: workerMain,
    policy: contentSecurityPolicy,
  });
  const channel = new MessageChannel();
  // stops the downloads of a kernel that is stopped
  const downloads = new AbortController();
  const calls = new Map<
    number,
    { resolve: (value: unknown) => void; reject: (error: Error) => void }
  >();
  let nextId = 0;
  let stopped = false;
  /** What the run in progress does with the questions and the progress */
  let current: {
    answer: (question: ScriptQuestion) => Promise<unknown>;
    onProgress?: (text: string) => void;
    onRunning: () => void;
  } | null = null;

  const call = (message: object, transfer: Transferable[] = []) =>
    new Promise<unknown>((resolve, reject) => {
      if (stopped) {
        reject(new Error("Python was stopped"));
        return;
      }
      const id = nextId++;
      calls.set(id, { resolve, reject });
      channel.port1.postMessage({ ...message, id }, transfer);
    });

  const stop = (reason: Error = new Error("Python was stopped")) => {
    if (stopped) {
      return;
    }
    stopped = true;
    window.removeEventListener("message", onPageMessage);
    downloads.abort(reason);
    channel.port1.close();
    // also stops the worker, whatever it is doing
    iframe.remove();
    for (const pending of calls.values()) {
      pending.reject(reason);
    }
    calls.clear();
  };

  const reply = (id: number, work: () => Promise<unknown>) => {
    work()
      .then((value) => {
        channel.port1.postMessage(
          { type: "answer", id, value },
          value instanceof ArrayBuffer ? [value] : [],
        );
      })
      .catch((problem: unknown) => {
        // also a value that cannot be copied to the sandbox is an error
        channel.port1.postMessage({
          type: "answer",
          id,
          error: problem instanceof Error ? problem.message : String(problem),
        });
      });
  };

  channel.port1.onmessage = ({ data }: MessageEvent<FromWorker>) => {
    if (data.type === "file") {
      reply(data.id, () => files.get(data.name, { signal: downloads.signal }));
    } else if (data.type === "question") {
      const answer = current?.answer;
      reply(data.id, () =>
        answer === undefined
          ? Promise.reject(new Error("nothing is running"))
          : answer(data.question),
      );
    } else if (data.type === "progress") {
      current?.onProgress?.(data.text);
    } else if (data.type === "running") {
      current?.onRunning();
    } else {
      const pending = calls.get(data.id);
      calls.delete(data.id);
      if (data.type === "done") {
        pending?.resolve(data.value);
      } else {
        pending?.reject(new Error(data.error));
      }
    }
  };

  /** The package of Pyodide of each module that it can import */
  let byImport = new Map<string, string>();

  let started: () => void;
  let failed: (error: Error) => void;
  const ready = new Promise<void>((resolve, reject) => {
    started = resolve;
    failed = reject;
  });
  function onPageMessage(
    event: MessageEvent<{ type?: string; error?: string }>,
  ): void {
    if (event.source !== iframe.contentWindow) {
      return;
    }
    if (event.data.type === "ready") {
      // an opaque origin cannot be named: "*", but only to this iframe
      iframe.contentWindow!.postMessage({ type: "run", run: null }, "*", [
        channel.port2,
      ]);
      started();
    } else if (event.data.type === "failed") {
      const error = new Error(
        `Python could not start: ${event.data.error ?? "unknown error"}`,
      );
      failed(error);
      stop(error);
    }
  }
  window.addEventListener("message", onPageMessage);
  const onAbort = () => {
    const reason = signal!.reason as Error;
    failed(reason);
    stop(reason);
  };
  signal?.addEventListener("abort", onAbort, { once: true });
  document.body.append(iframe);

  try {
    await ready;
    const given: Record<string, ArrayBuffer> = {};
    for (const name of coreFiles) {
      given[name] = await files.get(name, { signal: downloads.signal });
    }
    const lock = new TextDecoder().decode(
      await files.get(lockFile, { signal: downloads.signal }),
    );
    byImport = pyodidePackages(lock).byImport;
    await call(
      { type: "start", files: given, lock, driver },
      Object.values(given),
    );
  } catch (error) {
    stop(error instanceof Error ? error : new Error(String(error)));
    throw signal?.aborted ? (signal.reason as Error) : error;
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }

  /** Runs code once in the sandbox, with the time limit */
  const runOnce: PythonKernel["run"] = (
    run,
    answer,
    { timeLimit, onProgress, onRunning, signal: stopRun },
  ) => {
    if (stopRun?.aborted) {
      return Promise.reject(stopRun.reason as Error);
    }
    let timer: ReturnType<typeof setTimeout> | undefined;
    const onStop = () => stop(stopRun!.reason as Error);
    stopRun?.addEventListener("abort", onStop, { once: true });
    current = {
      answer,
      onProgress,
      // the time limit starts once the packages are there
      onRunning: () => {
        onRunning?.();
        timer = setTimeout(() => {
          stop(
            new Error(
              `the script took longer than ${timeLimit / 1000} seconds, and was stopped`,
            ),
          );
        }, timeLimit);
      },
    };
    const transfer = Object.values(run.files ?? {});
    return (
      call({ type: "run", run }, transfer) as Promise<ScriptOutcome>
    ).finally(() => {
      clearTimeout(timer);
      stopRun?.removeEventListener("abort", onStop);
      current = null;
    });
  };

  return {
    stop: () => stop(),
    isStopped: () => stopped,
    run: async (run, answer, options) => {
      // whether the code saved a column or a file, which a second run would
      // save again
      let changed = false;
      const watched = (question: ScriptQuestion) => {
        if (question.kind === "save_column" || question.kind === "download") {
          changed = true;
        }
        return answer(question);
      };
      // the files go to the sandbox; a second run needs them again
      const copy = (given: PythonRun): PythonRun =>
        given.files === undefined
          ? given
          : {
              ...given,
              files: Object.fromEntries(
                Object.entries(given.files).map(([path, buffer]) => [
                  path,
                  buffer.slice(0),
                ]),
              ),
            };
      const first = await runOnce(copy(run), watched, options);
      const missing = missingModule(first.error);
      if (missing === undefined || first.error === undefined) {
        return first;
      }
      const pyodide = byImport.get(missing.top);
      if (pyodide === undefined || run.packages?.includes(missing.top)) {
        return {
          ...first,
          error: explainMissingModule(first.error, byImport),
        };
      }
      // a package of Pyodide that the imports of the code did not show (e.g.
      // imported by a module of the skill in another way): loaded, and the
      // code runs again, once, unless it saved something already
      options.onProgress?.(`loading ${pyodide}, which the code needs`);
      if (changed) {
        await call({ type: "load", packages: [missing.top] });
        return {
          ...first,
          error: explainMissingModule(first.error, byImport, { loaded: true }),
        };
      }
      const second = await runOnce(
        copy({ ...run, packages: [...(run.packages ?? []), missing.top] }),
        watched,
        options,
      );
      return {
        ...second,
        logs: [
          `(The code needed ${pyodide}, a package of Pyodide that its imports did not show: it was loaded, and the code ran again.)`,
          ...second.logs,
        ],
        ...(second.error === undefined
          ? {}
          : { error: explainMissingModule(second.error, byImport) }),
      };
    },
  };
}
