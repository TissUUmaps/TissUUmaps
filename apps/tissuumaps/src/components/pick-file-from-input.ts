/**
 * Lets the user pick a file through a file input, which works in every
 * browser but yields no file handle
 *
 * A fresh input is used for every call, so that picking the same file again
 * still fires a change event.
 *
 * @param accept - The file extensions to offer
 * @returns The picked file, or `null` if the user cancelled
 */
export function pickFileFromInput(accept: string[]): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept.join(",");
    input.addEventListener("change", () => resolve(input.files?.[0] ?? null), {
      once: true,
    });
    input.addEventListener("cancel", () => resolve(null), { once: true });
    input.click();
  });
}
