// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { renderMarkdown } from "./markdown";

/** The cells of each row of the first table, as text */
function cellsOf(box: HTMLElement): string[][] {
  return [...box.querySelectorAll("tr")].map((row) =>
    [...row.children].map((cell) => cell.textContent),
  );
}

describe("renderMarkdown", () => {
  it("shows a table with its header, aligned columns and light borders", () => {
    const box = renderMarkdown(
      [
        "Cells per cluster:",
        "",
        "| Cluster | Cells | Share |",
        "|---|---|:-:|",
        "| 0 | 1,204 | 12% |",
        "| **10** | 87 | 1% |",
        "",
        "Done.",
      ].join("\n"),
    );
    const table = box.querySelector("table");
    expect(table).not.toBeNull();
    expect(cellsOf(box)).toEqual([
      ["Cluster", "Cells", "Share"],
      ["0", "1,204", "12%"],
      ["10", "87", "1%"],
    ]);
    expect(table?.querySelectorAll("th")).toHaveLength(3);
    // numbers to the right, as the dashes say otherwise
    const [cluster, cells, share] = [...(table?.querySelectorAll("td") ?? [])];
    expect(cluster?.style.textAlign).toBe("right");
    expect(cells?.style.textAlign).toBe("right");
    expect(share?.style.textAlign).toBe("center");
    expect(table?.querySelector("b")?.textContent).toBe("10");
    // only the theme's colors, which change with the light and dark theme
    expect(table?.style.border).toContain("var(--border)");
    expect(table?.querySelector("tr")?.style.background).toBe("var(--muted)");
    // no "|" or "---" left as text
    expect(box.textContent).not.toContain("|");
    expect(box.textContent).not.toContain("---");
    expect(box.textContent).toContain("Done.");
  });

  it("reads rows without the outer |, with too few or too many cells, and | inside cells", () => {
    const box = renderMarkdown(
      [
        "Gene | Note",
        ":--- | ---",
        "EPCAM | `a|b`",
        "KRT8 |",
        "VIM | x \\| y | extra",
      ].join("\n"),
    );
    expect(cellsOf(box)).toEqual([
      ["Gene", "Note"],
      ["EPCAM", "a|b"],
      ["KRT8", ""],
      ["VIM", "x | y"],
    ]);
    expect(box.querySelector("td")?.style.textAlign).toBe("left");
  });

  it("leaves lines with | that are not a table as text, also while a table arrives", () => {
    expect(renderMarkdown("a | b\nc | d").querySelector("table")).toBeNull();
    expect(renderMarkdown("| a | b |").querySelector("table")).toBeNull();
    // a header and a line of dashes with another number of columns
    expect(
      renderMarkdown("| a | b |\n|---|").querySelector("table"),
    ).toBeNull();
    // a horizontal line is no table
    expect(renderMarkdown("x\n---").querySelector("table")).toBeNull();
  });

  it("never turns the text of a table into HTML", () => {
    const box = renderMarkdown(
      "| a | b |\n|---|---|\n| <img src=x onerror=alert(1)> | <script>x</script> |",
    );
    expect(box.querySelector("img, script")).toBeNull();
    expect(box.querySelector("td")?.textContent).toBe(
      "<img src=x onerror=alert(1)>",
    );
  });

  it("makes links only to http and https addresses, opened in a new tab", () => {
    const box = renderMarkdown(
      "See [the paper](https://example.org/a_(b)?q=1) and [bad](javascript:alert(1)).",
    );
    const links = box.querySelectorAll("a");
    expect(links).toHaveLength(1);
    expect(links[0]?.textContent).toBe("the paper");
    expect(links[0]?.getAttribute("href")).toBe(
      "https://example.org/a_(b)?q=1",
    );
    expect(links[0]?.target).toBe("_blank");
    expect(links[0]?.rel).toBe("noopener noreferrer");
    expect(box.textContent).toContain("[bad](javascript:alert(1))");
  });
});
