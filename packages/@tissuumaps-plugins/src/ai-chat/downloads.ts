// Code in the sandbox can give the user a file of text, e.g. a table of
// results as CSV, with tm.download(name, text): the chat shows the file with
// "Save" and "Copy", and nothing is saved unless the user clicks. Only text,
// with an ending that says what it is, so that a file from the sandbox is
// never a program or a page.

/** The endings of the files that code can give, and their types */
export const downloadTypes: Record<string, string> = {
  ".csv": "text/csv",
  ".tsv": "text/tab-separated-values",
  ".txt": "text/plain",
  ".json": "application/json",
  ".md": "text/markdown",
};

/** The largest file that code can give, in bytes */
export const maxDownloadBytes = 10_000_000;

/**
 * Checks the name of a file that code gives to the user
 *
 * @param given - The name, e.g. "zscores.csv"
 * @returns The name, or why it is not valid
 */
export function downloadFileName(
  given: unknown,
): { name: string } | { error: string } {
  const name = typeof given === "string" ? given.trim() : "";
  const ending = /\.[^.]+$/.exec(name)?.[0].toLowerCase() ?? "";
  if (!Object.hasOwn(downloadTypes, ending)) {
    return {
      error: `the file must end with ${Object.keys(downloadTypes).join(", ")}, e.g. "zscores.csv"`,
    };
  }
  const stem = name.slice(0, -ending.length);
  const valid =
    name.length <= 100 &&
    /^[\p{L}\p{N}][\p{L}\p{N} ._()-]*$/u.test(stem) &&
    !/[. ]$/.test(stem) &&
    // names that Windows keeps for itself
    !/^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$/i.test(stem);
  return valid
    ? { name }
    : {
        error: `${JSON.stringify(name)} is not a valid file name: use letters, digits, spaces, "-", "_", "." and "( )", without folders, and at most 100 characters`,
      };
}
