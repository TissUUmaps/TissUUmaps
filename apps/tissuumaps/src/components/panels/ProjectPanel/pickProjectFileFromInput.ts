import { projectFileExtensions } from "@/data/io/workspace";

/**
 * Lets the user pick a project file through a file input, which works in
 * every browser but yields no file handle
 *
 * A fresh input is used for every call, so that picking the same file again
 * still fires a change event.
 *
 * @returns The picked file, or `null` if the user cancelled
 */
export function pickProjectFileFromInput(): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = projectFileExtensions.join(",");
    input.addEventListener("change", () => resolve(input.files?.[0] ?? null), {
      once: true,
    });
    input.addEventListener("cancel", () => resolve(null), { once: true });
    input.click();
  });
}
