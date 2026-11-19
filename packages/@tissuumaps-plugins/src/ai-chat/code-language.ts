// Code for the sandbox comes with its language, "javascript" or "python".
// When the code clearly is in the other language (e.g. a plugin that sends
// "export default async function (tm)" as Python), the error says so at
// once, instead of a syntax error of the wrong language.

/** The languages that the sandbox runs */
export type CodeLanguage = "javascript" | "python";

/** Signs that code is JavaScript */
const javascriptSigns = [
  /\bexport\s+default\b/,
  /^\s*(?:const|let|var)\s+[\w{[]/m,
  /^\s*(?:async\s+)?function\b/m,
  /\bconsole\.log\s*\(/,
  /\bawait\s+tm\.[a-z]+[A-Z]\w*\s*\(/,
];

/** Signs that code is Python */
const pythonSigns = [
  /^\s*(?:async\s+)?def\s+\w+\s*\([^)]*\)\s*(?:->[^:]+)?:/m,
  /^\s*import\s+[\w.]+(?:\s+as\s+\w+)?\s*$/m,
  /^\s*from\s+[\w.]+\s+import\s+/m,
  /^\s*print\s*\(/m,
  /\bawait\s+tm\.[a-z]+_\w*\s*\(/,
];

/**
 * Whether code shows signs of a language
 *
 * @param code - The code
 * @param signs - The signs of the language
 * @returns Whether one of them is there
 */
function hasSigns(code: string, signs: RegExp[]): boolean {
  return signs.some((sign) => sign.test(code));
}

/**
 * Says when code is clearly in another language than the one it was given
 * with
 *
 * @param code - The code
 * @param language - The language it was given with
 * @returns Why it cannot run as that language, or nothing if it may
 */
export function languageProblem(
  code: string,
  language: CodeLanguage,
): string | undefined {
  const javascript = hasSigns(code, javascriptSigns);
  const python = hasSigns(code, pythonSigns);
  if (language === "python" && javascript && !python) {
    return 'this looks like JavaScript, not Python: pass language "javascript", or write Python with async def main(tm): ...';
  }
  if (
    language === "javascript" &&
    python &&
    !/\bexport\s+default\b/.test(code)
  ) {
    return 'this looks like Python, not JavaScript: pass language "python", or write JavaScript with export default async function (tm) { ... }';
  }
  return undefined;
}
