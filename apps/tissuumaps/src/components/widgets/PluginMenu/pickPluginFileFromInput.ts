/** The file extensions of plugin modules */
const pluginFileExtensions = [".js", ".mjs"];

/**
 * Lets the user pick a plugin module file through a file input
 *
 * A fresh input is used for every call, so that picking the same file again
 * still fires a change event.
 *
 * @returns The picked file, or `null` if the user cancelled
 */
export function pickPluginFileFromInput(): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = pluginFileExtensions.join(",");
    input.addEventListener("change", () => resolve(input.files?.[0] ?? null), {
      once: true,
    });
    input.addEventListener("cancel", () => resolve(null), { once: true });
    input.click();
  });
}
