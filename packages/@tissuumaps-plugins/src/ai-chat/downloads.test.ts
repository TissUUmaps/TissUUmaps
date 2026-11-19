import { describe, expect, it } from "vitest";

import { downloadFileName } from "./downloads";

describe("downloadFileName", () => {
  it("takes names of text files", () => {
    expect(downloadFileName(" zscores.csv ")).toEqual({ name: "zscores.csv" });
    expect(downloadFileName("Z-scores (leiden).TSV")).toEqual({
      name: "Z-scores (leiden).TSV",
    });
    expect(downloadFileName("résumé.md")).toEqual({ name: "résumé.md" });
  });

  it("refuses other files, folders and names that Windows keeps", () => {
    for (const name of ["run.exe", "page.html", "zscores", 3]) {
      expect(downloadFileName(name)).toEqual({
        error:
          'the file must end with .csv, .tsv, .txt, .json, .md, e.g. "zscores.csv"',
      });
    }
    for (const name of [
      "../zscores.csv",
      "a/b.csv",
      "con.txt",
      ".csv",
      "x .csv",
    ]) {
      expect(downloadFileName(name)).toMatchObject({
        error: expect.stringContaining("is not a valid file name") as string,
      });
    }
    expect(downloadFileName(`${"a".repeat(97)}.csv`)).toMatchObject({
      error: expect.stringContaining("at most 100 characters") as string,
    });
  });
});
