// The code of a plugin made by the chat is imported as a module in the page,
// from a blob: address, as TissUUmaps loads a plugin from a file. Kept apart,
// so that the tests, whose page cannot import modules, can replace it.

/**
 * Imports the code of a plugin as a module, in the page
 *
 * @param code - The code
 * @returns The module
 * @throws If the code cannot be imported, e.g. a syntax error
 */
export async function importModule(code: string): Promise<unknown> {
  const url = URL.createObjectURL(
    new Blob([code], { type: "text/javascript" }),
  );
  try {
    return (await import(/* @vite-ignore */ url)) as unknown;
  } finally {
    URL.revokeObjectURL(url);
  }
}
