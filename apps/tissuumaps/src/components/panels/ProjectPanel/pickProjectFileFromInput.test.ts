import { afterEach, describe, expect, it, vi } from "vitest";

import { pickProjectFileFromInput } from "./pickProjectFileFromInput";

// jsdom opens no dialog, so the stub acts as the user on click
function stubFileInputClick(onClick: (input: HTMLInputElement) => void): void {
  vi.spyOn(HTMLInputElement.prototype, "click").mockImplementation(function (
    this: HTMLInputElement,
  ) {
    onClick(this);
  });
}

describe("pickProjectFileFromInput", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns the chosen file", async () => {
    const file = new File(["{}"], "project.tm4");
    stubFileInputClick((input) => {
      Object.defineProperty(input, "files", { value: [file] });
      input.dispatchEvent(new Event("change"));
    });
    await expect(pickProjectFileFromInput([".tm4"])).resolves.toBe(file);
  });

  it("offers the given project file extensions", async () => {
    let accept: string | undefined;
    stubFileInputClick((input) => {
      accept = input.accept;
      input.dispatchEvent(new Event("cancel"));
    });
    await pickProjectFileFromInput([".tm4", ".json"]);
    expect(accept).toBe(".tm4,.json");
  });

  it("returns null when the user cancels", async () => {
    stubFileInputClick((input) => {
      input.dispatchEvent(new Event("cancel"));
    });
    await expect(pickProjectFileFromInput([".tm4"])).resolves.toBeNull();
  });
});
