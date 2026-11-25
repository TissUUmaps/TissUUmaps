import { describe, expect, it } from "vitest";

import {
  analyzePython,
  explainMissingModule,
  findImports,
  listedPackages,
  missingModule,
  packageName,
  pyodidePackages,
} from "./python-imports";

/** A small list of packages, like that of Pyodide */
const lock = JSON.stringify({
  packages: {
    numpy: { name: "numpy", imports: ["numpy"] },
    scipy: { name: "scipy", imports: ["scipy"] },
    pandas: { name: "pandas", imports: ["pandas"] },
    matplotlib: { name: "matplotlib", imports: ["matplotlib", "pylab"] },
    "scikit-learn": { name: "scikit-learn", imports: ["sklearn"] },
    openblas: { name: "openblas", imports: [] },
  },
});

describe("python-imports", () => {
  it("reads the packages of Pyodide", () => {
    const packages = pyodidePackages(lock);
    expect(packages.byImport.get("sklearn")).toBe("scikit-learn");
    expect(packages.names.has("openblas")).toBe(true);
    expect(packageName("Scikit_Learn")).toBe("scikit-learn");
  });

  it("finds the imports of Python code, also in functions, but not in strings", () => {
    const code = [
      '"""A module.',
      "",
      "import fake_in_docstring",
      '"""',
      "import numpy as np, scipy.sparse as sp",
      "from typing import Sequence  # a comment",
      "from .utils.helpers import (",
      "    first,",
      "    second as other,  # why",
      ")",
      "from .. import parent",
      "import os; import json",
      "def plot():",
      "    import matplotlib.pyplot as plt",
      "x = 'import not_an_import'",
    ].join("\n");
    expect(findImports(code)).toEqual([
      { module: "numpy", level: 0, names: [] },
      { module: "scipy.sparse", level: 0, names: [] },
      { module: "typing", level: 0, names: ["Sequence"] },
      { module: "utils.helpers", level: 1, names: ["first", "second"] },
      { module: "", level: 2, names: ["parent"] },
      { module: "os", level: 0, names: [] },
      { module: "json", level: 0, names: [] },
      { module: "matplotlib.pyplot", level: 0, names: [] },
    ]);
  });

  it("reads the packages that requirements.txt and pyproject.toml list", () => {
    expect(
      listedPackages(
        "requirements.txt",
        "numpy>=1.24\n# a comment\nscikit-learn==1.4 ; python_version > '3.8'\n-r other.txt\n" +
          "git+https://github.com/x/y.git\nsquidpy[interactive]\n",
      ),
    ).toEqual(["numpy", "scikit-learn", "squidpy"]);
    expect(
      listedPackages(
        "src/pyproject.toml",
        [
          "[build-system]",
          'requires = ["setuptools"]',
          "[project]",
          'name = "anhood"',
          "dependencies = [",
          '  "numpy>=1.24",',
          "  'scipy',",
          "]",
          "[tool.poetry.dependencies]",
          'python = "^3.10"',
          'pandas = "^2.0"',
        ].join("\n"),
      ),
    ).toEqual(["numpy", "scipy", "pandas"]);
    expect(listedPackages("README.md", "numpy")).toEqual([]);
  });

  it("tells which packages the code needs, whether Pyodide has them, and which files cannot work", () => {
    // the files of anhood, in short
    const files = new Map([
      ["anhood/__init__.py", "from .anhood import anhood\n"],
      [
        "anhood/anhood.py",
        "import numpy as np\nimport pandas as pd\nfrom .utils.analytical_nhood_enrichment import compute\n",
      ],
      [
        "anhood/utils/analytical_nhood_enrichment.py",
        "from typing import Literal\nimport numpy as np\nimport scipy.sparse as sp\nfrom scipy.spatial import cKDTree\n",
      ],
      ["anhood/squidpy_wrapper/__init__.py", "from .anhood import anhood\n"],
      [
        "anhood/squidpy_wrapper/anhood.py",
        "import numpy as np\nfrom anndata import AnnData\nfrom squidpy.gr._utils import _save_data\n" +
          "from ..utils.analytical_nhood_enrichment import analytical_nhood_enrichment\n",
      ],
      [
        "example.ipynb",
        JSON.stringify({
          cells: [
            {
              cell_type: "code",
              source:
                "%load_ext autoreload\nimport squidpy as sq\nfrom anhood import anhood\nimport seaborn as sns",
            },
          ],
        }),
      ],
      ["scripts/run.py", "from helpers import go\nimport sklearn\n"],
      ["scripts/helpers.py", "from anhood.squidpy_wrapper import anhood\n"],
      ["requirements.txt", "numpy\nscikit-learn\nnumba\n"],
      ["README.md", "import nothing"],
    ]);
    const needs = analyzePython(files, pyodidePackages(lock));
    expect(needs.hasPython).toBe(true);
    expect(needs.packages).toEqual([
      {
        name: "anndata",
        files: ["anhood/squidpy_wrapper/anhood.py"],
        listed: false,
      },
      { name: "numba", files: [], listed: true },
      {
        name: "numpy",
        pyodide: "numpy",
        files: [
          "anhood/anhood.py",
          "anhood/utils/analytical_nhood_enrichment.py",
          "anhood/squidpy_wrapper/anhood.py",
        ],
        listed: true,
      },
      {
        name: "pandas",
        pyodide: "pandas",
        files: ["anhood/anhood.py"],
        listed: false,
      },
      {
        name: "scipy",
        pyodide: "scipy",
        files: ["anhood/utils/analytical_nhood_enrichment.py"],
        listed: false,
      },
      {
        name: "seaborn",
        files: ["example.ipynb"],
        listed: false,
      },
      {
        name: "sklearn",
        pyodide: "scikit-learn",
        files: ["scripts/run.py"],
        listed: true,
      },
      {
        name: "squidpy",
        files: ["anhood/squidpy_wrapper/anhood.py", "example.ipynb"],
        listed: false,
      },
    ]);
    // also through other files: helpers.py imports the wrapper, run.py helpers.py
    expect(needs.broken).toEqual([
      {
        path: "anhood/squidpy_wrapper/__init__.py",
        missing: ["anndata", "squidpy"],
      },
      {
        path: "anhood/squidpy_wrapper/anhood.py",
        missing: ["anndata", "squidpy"],
      },
      { path: "example.ipynb", missing: ["seaborn", "squidpy"] },
      { path: "scripts/helpers.py", missing: ["anndata", "squidpy"] },
      { path: "scripts/run.py", missing: ["anndata", "squidpy"] },
    ]);
  });

  it("says so when the packages of Pyodide are not known", () => {
    const needs = analyzePython(
      new Map([["main.py", "import numpy"]]),
      new Error("the page is not secure"),
    );
    expect(needs).toEqual({
      hasPython: true,
      packages: [{ name: "numpy", files: ["main.py"], listed: false }],
      broken: [],
      pyodideError: "the page is not secure",
    });
    expect(
      analyzePython(new Map([["README.md", "# Hi"]]), pyodidePackages(lock)),
    ).toEqual({ hasPython: false, packages: [], broken: [] });
  });
});

describe("missingModule and explainMissingModule", () => {
  it("finds the module that Python could not import", () => {
    expect(
      missingModule("ModuleNotFoundError: No module named 'scipy.sparse'"),
    ).toEqual({ module: "scipy.sparse", top: "scipy" });
    expect(missingModule("ZeroDivisionError: division by zero")).toBe(
      undefined,
    );
    expect(missingModule(undefined)).toBe(undefined);
  });

  it("drops the advice of Pyodide and says what to do in the sandbox", () => {
    const { byImport } = pyodidePackages(lock);
    const explained = explainMissingModule(
      [
        '  File "<script>", line 3, in main',
        "ModuleNotFoundError: No module named 'sklearn'",
        "The module 'scikit-learn' is included in the Pyodide distribution, but it is not installed.",
        "You can install it by calling:",
        '  await micropip.install("scikit-learn") in Python, or',
      ].join("\n"),
      byImport,
    );
    expect(explained).toBe(
      [
        '  File "<script>", line 3, in main',
        "ModuleNotFoundError: No module named 'sklearn'",
        "scikit-learn is a package of Pyodide, the Python of the sandbox, but it was not loaded before the code ran.",
      ].join("\n"),
    );
    expect(explained).not.toContain("micropip.install");
    // another error stays as it is
    expect(explainMissingModule("ValueError: no", byImport)).toBe(
      "ValueError: no",
    );
  });
});
