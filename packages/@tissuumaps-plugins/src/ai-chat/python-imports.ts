import { notebookCode } from "./notebooks";

// Before a repository is added as a skill, the user is told which Python
// packages its code needs, and whether Pyodide has them: Python in the
// sandbox can only import the packages of Pyodide. The imports are found by
// reading the code (Python is not started for this), so the list is a good
// guess rather than a proof: e.g. an import in a branch that never runs
// counts too. The packages that requirements.txt or pyproject.toml list are
// added to the list.

/**
 * The modules that need no package: those of the standard library of Python
 * (sys.stdlib_module_names of Python 3.15, without the private ones), and
 * those that Pyodide has in itself
 */
const builtInModules = new Set(
  (
    "__future__ abc annotationlib antigravity argparse array ast asyncio atexit base64 bdb binascii bisect " +
    "builtins bz2 cProfile calendar cmath cmd code codecs codeop collections colorsys compileall " +
    "compression concurrent configparser contextlib contextvars copy copyreg csv ctypes curses " +
    "dataclasses datetime dbm decimal difflib dis doctest email encodings ensurepip enum errno " +
    "faulthandler fcntl filecmp fileinput fnmatch fractions ftplib functools gc genericpath getopt " +
    "getpass gettext glob graphlib grp gzip hashlib heapq hmac html http idlelib imaplib importlib " +
    "inspect io ipaddress itertools json keyword linecache locale logging lzma mailbox marshal math " +
    "mimetypes mmap modulefinder msvcrt multiprocessing netrc nt ntpath nturl2path numbers opcode " +
    "operator optparse os pathlib pdb pickle pickletools pkgutil platform plistlib poplib posix " +
    "posixpath pprint profile profiling pstats pty pwd py_compile pyclbr pydoc pydoc_data pyexpat " +
    "queue quopri random re readline reprlib resource rlcompleter runpy sched secrets select " +
    "selectors shelve shlex shutil signal site smtplib socket socketserver sqlite3 ssl stat " +
    "statistics string stringprep struct subprocess symtable sys sysconfig syslog tabnanny tarfile " +
    "tempfile termios textwrap this threading time timeit tkinter token tokenize tomllib trace " +
    "traceback tracemalloc tty turtle turtledemo types typing unicodedata unittest urllib uuid venv " +
    "warnings wave weakref webbrowser winreg winsound wsgiref xml xmlrpc zipapp zipfile zipimport " +
    "zlib zoneinfo js pyodide pyodide_js"
  ).split(" "),
);

/** The packages of Pyodide, from its list of packages (pyodide-lock.json) */
export type PyodidePackages = {
  /** The package of each module that can be imported, e.g. "sklearn" → "scikit-learn" */
  byImport: Map<string, string>;
  /** The names of the packages, normalized (see {@link packageName}) */
  names: Set<string>;
};

/** A package that the code needs */
export type NeededPackage = {
  /** The module that the code imports, e.g. "sklearn", or the name of a listed package */
  name: string;
  /** The package of Pyodide that has it, e.g. "scikit-learn"; none if Pyodide does not have it */
  pyodide?: string;
  /** The files that import it */
  files: string[];
  /** Whether requirements.txt or pyproject.toml lists it */
  listed: boolean;
};

/** What the Python code of a folder needs */
export type PythonNeeds = {
  /** Whether the folder has Python code (.py files or notebooks) */
  hasPython: boolean;
  /** The packages, outside the standard library and the folder itself */
  packages: NeededPackage[];
  /**
   * The files that cannot work, because they need a package that Pyodide does
   * not have, by themselves or through another file of the folder
   */
  broken: { path: string; missing: string[] }[];
  /** Why the packages of Pyodide are not known, if they are not */
  pyodideError?: string;
};

/** An import of some code: `from <level dots><module> import <names>` */
type PythonImport = { module: string; level: number; names: string[] };

/**
 * The normalized name of a package, as pip compares them (PEP 503)
 *
 * @param name - e.g. "Scikit_Learn"
 * @returns e.g. "scikit-learn"
 */
export function packageName(name: string): string {
  return name.toLowerCase().replace(/[-_.]+/g, "-");
}

/**
 * Reads the list of the packages of Pyodide
 *
 * @param lockText - The text of pyodide-lock.json
 * @returns The packages, and the modules that each one has
 */
export function pyodidePackages(lockText: string): PyodidePackages {
  const lock = JSON.parse(lockText) as {
    packages?: Record<string, { name?: string; imports?: string[] }>;
  };
  const byImport = new Map<string, string>();
  const names = new Set<string>();
  for (const [key, entry] of Object.entries(lock.packages ?? {})) {
    const name = entry.name ?? key;
    names.add(packageName(name));
    for (const module of entry.imports ?? []) {
      byImport.set(module, name);
    }
  }
  return { byImport, names };
}

/**
 * Finds the imports of some Python code, without running it: the lines that
 * start with `import` or `from … import`, also inside functions and `try`
 *
 * @param source - The code
 * @returns The imports, in their order
 */
export function findImports(source: string): PythonImport[] {
  // without the strings over several lines (e.g. docstrings), and with the
  // lines that a backslash continues joined
  const code = source
    .replace(/("""|''')[\s\S]*?\1/g, '""')
    .replace(/\\\r?\n/g, " ");
  const lines = code.split(/\r?\n/);
  const imports: PythonImport[] = [];
  const nameOf = (part: string) =>
    part
      .trim()
      .split(/\s+as\s+/)[0]!
      .trim();
  for (let index = 0; index < lines.length; index++) {
    for (const statement of lines[index]!.replace(/#.*$/, "").split(";")) {
      const plain = /^\s*import\s+(.+)$/.exec(statement);
      if (plain !== null) {
        for (const module of plain[1]!.split(",").map(nameOf)) {
          if (/^[A-Za-z_][\w.]*$/.test(module)) {
            imports.push({ module, level: 0, names: [] });
          }
        }
        continue;
      }
      const from = /^\s*from\s+(\.*)\s*([\w.]*)\s+import\s+(.*)$/.exec(
        statement,
      );
      if (from === null) {
        continue;
      }
      let names = from[3]!;
      // a list of names in parentheses, over several lines
      if (names.trimStart().startsWith("(")) {
        while (!names.includes(")") && index + 1 < lines.length) {
          index++;
          names += ` ${lines[index]!.replace(/#.*$/, "")}`;
        }
      }
      imports.push({
        module: from[2]!,
        level: from[1]!.length,
        names: names
          .replace(/[()]/g, " ")
          .split(",")
          .map(nameOf)
          .filter((name) => /^[A-Za-z_]\w*$/.test(name)),
      });
    }
  }
  return imports;
}

/**
 * The packages that requirements.txt or pyproject.toml lists
 *
 * @param path - The path of the file
 * @param text - Its text
 * @returns The names of the packages, e.g. ["numpy", "scikit-learn"]
 */
export function listedPackages(path: string, text: string): string[] {
  const file = path.split("/").at(-1)!.toLowerCase();
  const nameOf = (spec: string) =>
    /^\s*([A-Za-z0-9][A-Za-z0-9._-]*)/.exec(spec)?.[1];
  const names: string[] = [];
  if (/^requirements.*\.txt$/.test(file)) {
    for (const line of text.split(/\r?\n/)) {
      const spec = line.replace(/(^|\s)#.*$/, "").trim();
      // options such as -r other.txt, and packages from an address
      if (spec === "" || spec.startsWith("-") || spec.includes("://")) {
        continue;
      }
      const name = nameOf(spec);
      if (name !== undefined) {
        names.push(name);
      }
    }
  } else if (file === "pyproject.toml") {
    let section = "";
    let inList = false;
    for (const line of text.split(/\r?\n/)) {
      const plain = line.replace(/(^|\s)#.*$/, "").trim();
      const header = /^\[([^\]]+)\]$/.exec(plain);
      if (header !== null && !inList) {
        section = header[1]!.trim();
        continue;
      }
      if (section === "project" && /^dependencies\s*=/.test(plain)) {
        inList = true;
      }
      if (inList) {
        for (const [, spec] of plain.matchAll(/["']([^"']+)["']/g)) {
          const name = nameOf(spec!);
          if (name !== undefined) {
            names.push(name);
          }
        }
        if (plain.includes("]")) {
          inList = false;
        }
        continue;
      }
      // Poetry: one package per line, e.g. numpy = "^1.26"
      const key = /^([A-Za-z0-9][A-Za-z0-9._-]*)\s*=/.exec(plain)?.[1];
      if (
        section === "tool.poetry.dependencies" &&
        key !== undefined &&
        key.toLowerCase() !== "python"
      ) {
        names.push(key);
      }
    }
  }
  return [...new Set(names)];
}

/**
 * Finds what the Python code of a folder needs: the packages it imports or
 * lists, whether Pyodide has them, and which files cannot work without them
 *
 * @param files - The text of the files of the folder, by path in it
 * @param pyodide - The packages of Pyodide, or why they are not known
 * @returns What the code needs
 */
export function analyzePython(
  files: Map<string, string>,
  pyodide: PyodidePackages | Error,
): PythonNeeds {
  const pythonFiles = new Set(
    [...files.keys()].filter((path) => path.endsWith(".py")),
  );
  // the folders that hold Python modules, e.g. "anhood" and "anhood/utils"
  const folders = new Set<string>();
  for (const path of pythonFiles) {
    const parts = path.split("/").slice(0, -1);
    for (let end = 1; end <= parts.length; end++) {
      folders.add(parts.slice(0, end).join("/"));
    }
  }
  const join = (folder: string, dotted: string) =>
    [folder, dotted.replaceAll(".", "/")]
      .filter((part) => part !== "")
      .join("/");
  const moduleFile = (folder: string, dotted: string) => {
    const base = join(folder, dotted);
    return pythonFiles.has(`${base}.py`)
      ? `${base}.py`
      : pythonFiles.has(`${base}/__init__.py`)
        ? `${base}/__init__.py`
        : undefined;
  };
  const isLocal = (folder: string, top: string) =>
    pythonFiles.has(`${join(folder, top)}.py`) ||
    folders.has(join(folder, top));
  // the files of a module and of the packages above it, e.g. a/__init__.py
  const localFiles = (folder: string, imported: PythonImport) => {
    const found: string[] = [];
    const parts = imported.module === "" ? [] : imported.module.split(".");
    for (let end = 1; end <= parts.length; end++) {
      found.push(moduleFile(folder, parts.slice(0, end).join(".")) ?? "");
    }
    for (const name of imported.names) {
      found.push(moduleFile(folder, [...parts, name].join(".")) ?? "");
    }
    return found.filter((path) => path !== "");
  };

  // what each file imports: from outside, and of the folder
  const outside = new Map<string, Set<string>>();
  const inside = new Map<string, Set<string>>();
  const sources = [...files].filter(
    ([path]) => path.endsWith(".py") || path.endsWith(".ipynb"),
  );
  for (const [path, text] of sources) {
    const folder = path.split("/").slice(0, -1).join("/");
    const code = path.endsWith(".ipynb") ? notebookCode(text) : text;
    const external = new Set<string>();
    const local = new Set<string>();
    for (const imported of findImports(code)) {
      if (imported.level > 0) {
        const parts = folder === "" ? [] : folder.split("/");
        const base = parts
          .slice(0, parts.length - imported.level + 1)
          .join("/");
        for (const file of localFiles(base, imported)) {
          local.add(file);
        }
        continue;
      }
      const top = imported.module.split(".")[0]!;
      // as Python: the folder of the script, then the top of the folder and src
      const root = [folder, "", "src"].find((candidate) =>
        isLocal(candidate, top),
      );
      if (root !== undefined) {
        for (const file of localFiles(root, imported)) {
          local.add(file);
        }
      } else if (!builtInModules.has(top)) {
        external.add(top);
      }
    }
    local.delete(path);
    outside.set(path, external);
    inside.set(path, local);
  }

  const known = pyodide instanceof Error ? undefined : pyodide;
  const packages = new Map<string, NeededPackage>();
  for (const [path, external] of outside) {
    for (const name of external) {
      const entry = packages.get(name) ?? {
        name,
        ...(known?.byImport.has(name)
          ? { pyodide: known.byImport.get(name)! }
          : {}),
        files: [],
        listed: false,
      };
      entry.files.push(path);
      packages.set(name, entry);
    }
  }
  for (const [path, text] of files) {
    for (const listed of listedPackages(path, text)) {
      const normalized = packageName(listed);
      const inPyodide = [...(known?.byImport.values() ?? [])].find(
        (name) => packageName(name) === normalized,
      );
      const same = [...packages.values()].find(
        (entry) =>
          packageName(entry.name) === normalized ||
          (inPyodide !== undefined && entry.pyodide === inPyodide),
      );
      if (same !== undefined) {
        same.listed = true;
      } else {
        packages.set(listed, {
          name: listed,
          ...(inPyodide !== undefined || known?.names.has(normalized)
            ? { pyodide: inPyodide ?? listed }
            : {}),
          files: [],
          listed: true,
        });
      }
    }
  }

  // a file cannot work without a missing package, or a file that cannot work
  const missing = new Map<string, Set<string>>();
  if (known !== undefined) {
    for (const [path, external] of outside) {
      missing.set(
        path,
        new Set([...external].filter((name) => !known.byImport.has(name))),
      );
    }
    for (let changed = true; changed;) {
      changed = false;
      for (const [path, local] of inside) {
        const own = missing.get(path)!;
        for (const file of local) {
          for (const name of missing.get(file) ?? []) {
            if (!own.has(name)) {
              own.add(name);
              changed = true;
            }
          }
        }
      }
    }
  }
  return {
    hasPython: sources.length > 0,
    packages: [...packages.values()].sort((a, b) =>
      a.name.localeCompare(b.name),
    ),
    broken: [...missing]
      .filter(([, names]) => names.size > 0)
      .map(([path, names]) => ({ path, missing: [...names].sort() }))
      .sort((a, b) => a.path.localeCompare(b.path)),
    ...(pyodide instanceof Error ? { pyodideError: pyodide.message } : {}),
  };
}

/**
 * The module that Python could not import, from the text of the error
 *
 * @param error - The error of the code, e.g. "ModuleNotFoundError: No module named 'scipy.sparse'"
 * @returns The module, e.g. "scipy.sparse", and its top, e.g. "scipy"; none
 * for another error
 */
export function missingModule(
  error: string | undefined,
): { module: string; top: string } | undefined {
  const module = /ModuleNotFoundError: No module named '([\w.]+)'/.exec(
    error ?? "",
  )?.[1];
  return module === undefined
    ? undefined
    : { module, top: module.split(".")[0]! };
}

/**
 * Explains an error of a module that Python could not import, in place of
 * the advice of Pyodide (micropip, loadPackage), which the sandbox cannot
 * follow: it has no network, and the code cannot load packages itself
 *
 * @param error - The error of the code
 * @param byImport - The package of Pyodide of each module it can import
 * @param options - loaded: the package of Pyodide was loaded after the
 * error, so that it is there for the next run
 * @returns The error, explained; the same error if it is another one
 */
export function explainMissingModule(
  error: string,
  byImport: Map<string, string>,
  options?: { loaded?: boolean },
): string {
  const missing = missingModule(error);
  if (missing === undefined) {
    return error;
  }
  const lines = error.split("\n");
  const at = lines.findIndex((line) =>
    line.includes("ModuleNotFoundError: No module named"),
  );
  const kept = lines.slice(0, at + 1).join("\n");
  const pyodide = byImport.get(missing.top);
  if (pyodide !== undefined) {
    return `${kept}\n${pyodide} is a package of Pyodide, the Python of the sandbox, but it was not loaded before the code ran${
      options?.loaded === true
        ? "; it is loaded now: run the code again. It was not run again by itself, since it had already saved a column or a file."
        : "."
    }`;
  }
  return `${kept}\n'${missing.top}' is not a package of Pyodide, the Python of the sandbox, which can only import the packages of Pyodide and the files of the skill: micropip and pip cannot install anything here, since the sandbox has no network. If '${missing.top}' is a module of the skill, import it as the files of the skill are laid out (their folder is on the path of Python); else do without it, e.g. with numpy, scipy, pandas or scikit-learn.`;
}

/**
 * The name under which Python can import a skill whose folder is a package
 * (it has __init__.py at its top, often with relative imports, e.g. the
 * folder points2regions of a repository): the name of the skill, with _ for
 * its dashes
 *
 * @param skill - The skill: its name and its files
 * @returns The name, e.g. "points2regions"; none if its folder is not a package
 */
export function skillPackageName(skill: {
  name: string;
  files: Map<string, unknown>;
}): string | undefined {
  if (!skill.files.has("__init__.py")) {
    return undefined;
  }
  const name = skill.name.replace(/[^A-Za-z0-9_]/g, "_");
  return /^[A-Za-z_]\w*$/.test(name) ? name : `_${name}`;
}
