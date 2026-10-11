import { afterEach, describe, expect, it, vi } from "vitest";

import { pickPluginFileFromInput } from "./pickPluginFileFromInput";

// jsdom opens no dialog, so the stub acts as the user on click
function stubFileInputClick(onClick: (input: HTMLInputElement) => void): void {
  vi.spyOn(HTMLInputElement.prototype, "click").mockImplementation(function (
    this: HTMLInputElement,
  ) {
    onClick(this);
  });
}

describe("pickPluginFileFromInput", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns the chosen file", async () => {
    const file = new File(["export default {};"], "plugin.js");
    stubFileInputClick((input) => {
      Object.defineProperty(input, "files", { value: [file] });
      input.dispatchEvent(new Event("change"));
    });
    await expect(pickPluginFileFromInput()).resolves.toBe(file);
  });

  it("offers plugin module files", async () => {
    let accept: string | undefined;
    stubFileInputClick((input) => {
      accept = input.accept;
      input.dispatchEvent(new Event("cancel"));
    });
    await pickPluginFileFromInput();
    expect(accept).toBe(".js,.mjs");
  });

  it("returns null when the user cancels", async () => {
    stubFileInputClick((input) => {
      input.dispatchEvent(new Event("cancel"));
    });
    await expect(pickPluginFileFromInput()).resolves.toBeNull();
  });
});
