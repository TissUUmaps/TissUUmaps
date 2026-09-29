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
    const file = new File(["{}"], "project.tmap");
    stubFileInputClick((input) => {
      Object.defineProperty(input, "files", { value: [file] });
      input.dispatchEvent(new Event("change"));
    });
    await expect(pickProjectFileFromInput()).resolves.toBe(file);
  });

  it("offers project files", async () => {
    let accept: string | undefined;
    stubFileInputClick((input) => {
      accept = input.accept;
      input.dispatchEvent(new Event("cancel"));
    });
    await pickProjectFileFromInput();
    expect(accept).toBe(".tmap,.json");
  });

  it("returns null when the user cancels", async () => {
    stubFileInputClick((input) => {
      input.dispatchEvent(new Event("cancel"));
    });
    await expect(pickProjectFileFromInput()).resolves.toBeNull();
  });
});
