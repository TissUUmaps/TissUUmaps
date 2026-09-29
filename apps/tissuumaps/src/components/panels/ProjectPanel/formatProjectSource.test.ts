import { describe, expect, it } from "vitest";

import { formatProjectSource } from "./formatProjectSource";

const baseUrl = "https://example.org/app/index.html";

describe("formatProjectSource", () => {
  it("formats a workspace file relative to the folder", () => {
    expect(formatProjectSource("/a/project.tmap", "data", baseUrl)).toBe(
      "data › a/project.tmap",
    );
  });

  it("names the folder generically without a workspace", () => {
    expect(formatProjectSource("/project.tmap", null, baseUrl)).toBe(
      "Folder › project.tmap",
    );
  });

  it("formats a URL under the app relative to the app", () => {
    expect(
      formatProjectSource(
        "https://example.org/app/data/project.json",
        null,
        baseUrl,
      ),
    ).toBe("data/project.json");
  });

  it("formats another URL by host and path", () => {
    expect(
      formatProjectSource("https://other.org/project.json", null, baseUrl),
    ).toBe("other.org/project.json");
  });

  it("keeps the query string", () => {
    expect(
      formatProjectSource("https://other.org/load?project=a", null, baseUrl),
    ).toBe("other.org/load?project=a");
  });

  it("keeps a source that is not a URL", () => {
    expect(formatProjectSource("project.json", null, baseUrl)).toBe(
      "project.json",
    );
  });
});
