import { describe, expect, it } from "vitest";

import { languageProblem } from "./code-language";

const javascript =
  "export default async function (tm) {\n  return (await tm.tables()).length;\n}";
const python =
  "import numpy as np\n\nasync def main(tm):\n    return len(await tm.tables())";

describe("languageProblem", () => {
  it("says when JavaScript is sent as Python, and the other way round", () => {
    expect(languageProblem(javascript, "python")).toBe(
      'this looks like JavaScript, not Python: pass language "javascript", or write Python with async def main(tm): ...',
    );
    expect(languageProblem(python, "javascript")).toBe(
      'this looks like Python, not JavaScript: pass language "python", or write JavaScript with export default async function (tm) { ... }',
    );
  });

  it("accepts code in its own language, and code it cannot tell", () => {
    expect(languageProblem(javascript, "javascript")).toBeUndefined();
    expect(languageProblem(python, "python")).toBeUndefined();
    expect(languageProblem("x", "python")).toBeUndefined();
    // JavaScript that carries Python in a string is still JavaScript
    expect(
      languageProblem(
        `const PYTHON = \`${python}\`;\nexport default async function (tm) { return PYTHON; }`,
        "javascript",
      ),
    ).toBeUndefined();
    // Python with a dict that looks like an object is still Python
    expect(
      languageProblem(
        'async def main(tm):\n    result = {"a": 1}\n    return result',
        "python",
      ),
    ).toBeUndefined();
  });
});
