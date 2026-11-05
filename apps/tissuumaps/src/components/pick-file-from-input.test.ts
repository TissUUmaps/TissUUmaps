import { afterEach, describe, expect, it, vi } from "vitest";

import { pickFileFromInput } from "./pick-file-from-input";

// jsdom opens no dialog, so the stub acts as the user on click
function stubFileInputClick(onClick: (input: HTMLInputElement) => void): void {
  vi.spyOn(HTMLInputElement.prototype, "click").mockImplementation(function (
    this: HTMLInputElement,
  ) {
    onClick(this);
  });
}

describe("pickFileFromInput", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns the chosen file", async () => {
    const file = new File(["{}"], "project.tm4");
    stubFileInputClick((input) => {
      Object.defineProperty(input, "files", { value: [file] });
      input.dispatchEvent(new Event("change"));
    });
    await expect(pickFileFromInput([".tm4"])).resolves.toBe(file);
  });

  it("offers the given file extensions", async () => {
    let accept: string | undefined;
    stubFileInputClick((input) => {
      accept = input.accept;
      input.dispatchEvent(new Event("cancel"));
    });
    await pickFileFromInput([".js", ".mjs"]);
    expect(accept).toBe(".js,.mjs");
  });

  it("returns null when the user cancels", async () => {
    stubFileInputClick((input) => {
      input.dispatchEvent(new Event("cancel"));
    });
    await expect(pickFileFromInput([".tm4"])).resolves.toBeNull();
  });
});
