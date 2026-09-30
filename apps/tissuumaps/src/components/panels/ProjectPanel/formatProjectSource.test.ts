import { describe, expect, it } from "vitest";

import { formatProjectSource } from "./formatProjectSource";

const baseUrl = "https://example.org/app/index.html";

describe("formatProjectSource", () => {
  it("formats a workspace file relative to the folder", () => {
    expect(formatProjectSource("/a/project.tm4", "data", baseUrl)).toBe(
      "data › a/project.tm4",
    );
  });

  it("names the folder generically without a workspace", () => {
    expect(formatProjectSource("/project.tm4", null, baseUrl)).toBe(
      "Folder › project.tm4",
    );
  });

  it("formats a URL under the app relative to the app", () => {
    expect(
      formatProjectSource(
        "https://example.org/app/data/project.tm4",
        null,
        baseUrl,
      ),
    ).toBe("data/project.tm4");
  });

  it("formats another URL by host and path", () => {
    expect(
      formatProjectSource("https://other.org/project.tm4", null, baseUrl),
    ).toBe("other.org/project.tm4");
  });

  it("keeps the query string", () => {
    expect(
      formatProjectSource("https://other.org/load?project=a", null, baseUrl),
    ).toBe("other.org/load?project=a");
  });

  it("keeps a source that is not a URL", () => {
    expect(formatProjectSource("project.tm4", null, baseUrl)).toBe(
      "project.tm4",
    );
  });
});
