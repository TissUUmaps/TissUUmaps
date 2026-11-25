// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  bars,
  formatNumber,
  groupName,
  heatmap,
  legend,
  niceTicks,
  numberOrder,
  scatter,
  signedColor,
} from "./plugin-charts";

/** The texts of the SVG elements with a data attribute, in order */
function texts(element: HTMLElement, attribute: string): string[] {
  return [...element.querySelectorAll(`[${attribute}]`)].map(
    (part) => part.textContent,
  );
}

/** The tooltip of a chart, line by line */
function tooltipOf(element: HTMLElement): string[] {
  const tooltip = element.querySelector<HTMLElement>('[role="tooltip"]')!;
  return tooltip.style.display === "none"
    ? []
    : [...tooltip.children].map((line) => line.textContent);
}

/** Moves the mouse onto x, y of the SVG (jsdom puts it at 0, 0) */
function pointAt(
  element: HTMLElement,
  x: number,
  y: number,
  type = "mousemove",
) {
  element
    .querySelector("svg")!
    .dispatchEvent(
      new MouseEvent(type, { clientX: x, clientY: y, bubbles: true }),
    );
}

/** The middle of the cell of a heatmap, by its row and column */
function cellMiddle(element: HTMLElement, row: string, column: string) {
  const cell = element.querySelector(`[data-cell="${row}|${column}"]`)!;
  const x = Number(cell.getAttribute("x"));
  const y = Number(cell.getAttribute("y"));
  const size = Number(cell.getAttribute("width"));
  return { x: x + size / 2, y: y + size / 2, fill: cell.getAttribute("fill") };
}

afterEach(() => {
  document.documentElement.classList.remove("dark");
  document.documentElement.removeAttribute("style");
  document.body.replaceChildren();
});

describe("numbers and colors", () => {
  it("names groups as whole numbers and sorts them as numbers", () => {
    expect(groupName(3)).toBe("3");
    expect(groupName("3.0")).toBe("3");
    expect(groupName("T cells")).toBe("T cells");
    const names = ["10", "2", "0", "1"];
    expect(numberOrder(names).map((index) => names[index])).toEqual([
      "0",
      "1",
      "2",
      "10",
    ]);
    // names that are not all numbers keep their order
    expect(numberOrder(["b", "a", "1"])).toEqual([0, 1, 2]);
  });

  it("writes numbers and ticks for people", () => {
    expect(formatNumber(12.345)).toBe("12.3");
    expect(formatNumber(-0.5)).toBe("−0.5");
    expect(formatNumber(3)).toBe("3");
    expect(formatNumber(Number.NaN)).toBe("n/a");
    expect(niceTicks(0, 97)).toEqual([0, 20, 40, 60, 80, 100]);
    expect(niceTicks(-3, 3)).toContain(0);
  });

  it("has a gray at 0, blue below and red above, light and dark", () => {
    expect(signedColor(0, 5, false)).toBe("#f0efec");
    expect(signedColor(0, 5, true)).toBe("#383835");
    const red = signedColor(5, 5, false);
    const blue = signedColor(-5, 5, false);
    expect(parseInt(red.slice(1, 3), 16)).toBeGreaterThan(
      parseInt(red.slice(5, 7), 16),
    );
    expect(parseInt(blue.slice(5, 7), 16)).toBeGreaterThan(
      parseInt(blue.slice(1, 3), 16),
    );
  });
});

describe("chat.heatmap", () => {
  const rows = ["10", "2", "0", "1.0"];
  /** z-scores: row i, column j */
  const values = [
    [4, -1, 0.5, 0],
    [-1, 2, null, -4],
    [0.5, null, 3, 1],
    [0, -4, 1, 1],
  ];

  it("shows the names in number order, a color bar centered on 0, and the theme", async () => {
    document.documentElement.style.setProperty("--background", "#ffffff");
    document.documentElement.style.setProperty("--foreground", "#0a0a0a");
    const place = document.createElement("div");
    document.body.append(place);
    heatmap(place, {
      rows,
      values,
      title: "Neighborhood enrichment",
      rowTitle: "Cluster",
      columnTitle: "Neighbor cluster",
      valueTitle: "z-score",
    });
    expect(texts(place, "data-row-label")).toEqual(["0", "1", "2", "10"]);
    expect(texts(place, "data-column-label")).toEqual(["0", "1", "2", "10"]);
    expect(texts(place, "data-legend-title")).toEqual(["z-score"]);
    expect(texts(place, "data-legend-tick")).toEqual([
      "4",
      "2",
      "0",
      "−2",
      "−4",
    ]);
    // the background of the theme, the gray of 0, the muted color where missing
    const svg = place.querySelector("svg")!;
    expect(svg.firstElementChild?.getAttribute("fill")).toBe("#ffffff");
    expect(cellMiddle(place, "10", "1").fill).toBe("#f0efec");
    expect(cellMiddle(place, "2", "2").fill).not.toBe("#f0efec");
    expect(
      [...place.querySelectorAll("text")].every(
        (text) => text.getAttribute("fill") !== "",
      ),
    ).toBe(true);
    expect(
      [...place.querySelectorAll("button")].map((b) => b.textContent),
    ).toEqual(["Save PNG", "Save CSV"]);

    // drawn again in the colors of the dark theme
    document.documentElement.style.setProperty("--background", "#141414");
    document.documentElement.classList.add("dark");
    await vi.waitFor(() =>
      expect(cellMiddle(place, "10", "1").fill).toBe("#383835"),
    );
    expect(
      place.querySelector("svg")!.firstElementChild?.getAttribute("fill"),
    ).toBe("#141414");
  });

  it("shows the values under the mouse, and gives a click to the plugin", () => {
    const place = document.createElement("div");
    const clicks: unknown[] = [];
    heatmap(place, {
      rows,
      values,
      rowTitle: "Cluster",
      columnTitle: "Neighbor cluster",
      valueTitle: "z-score",
      tooltip: (cell: { rowIndex: number }) => `row ${cell.rowIndex} given`,
      onClick: (cell: unknown) => clicks.push(cell),
    });
    const middle = cellMiddle(place, "2", "10");
    pointAt(place, middle.x, middle.y);
    expect(tooltipOf(place)).toEqual([
      "z-score: −1",
      "Cluster 2, neighbor cluster 10",
      "row 1 given",
    ]);
    pointAt(place, middle.x, middle.y, "click");
    expect(clicks).toEqual([
      { row: "2", column: "10", value: -1, rowIndex: 1, columnIndex: 0 },
    ]);
    expect(place.querySelector('[data-outline="selected"]')).not.toBeNull();
    const missing = cellMiddle(place, "2", "0");
    pointAt(place, missing.x, missing.y);
    expect(tooltipOf(place)[0]).toBe("z-score: no value");
    pointAt(place, 1, 1);
    expect(tooltipOf(place)).toEqual([]);
  });

  it("gives the values as a CSV in number order, and replaces itself", () => {
    const place = document.createElement("div");
    const chart = heatmap(place, { rows, values, rowTitle: "cluster" });
    expect(chart.csv().split("\n").slice(0, 2)).toEqual([
      "cluster,0,1,2,10",
      "0,3,1,,0.5",
    ]);
    // a sequential scale for values without a sign
    chart.update({
      values: values.map((row) =>
        row.map((v) => (v === null ? null : Math.abs(v))),
      ),
    });
    expect(texts(place, "data-legend-tick")).toEqual(["4", "3", "2", "1", "0"]);
    // without the diagonal, the scale ends at the largest value off it
    chart.update({
      rows: ["0", "1", "2"],
      values: [
        [50, -2, 1],
        [-2, 40, 2],
        [1, 2, 30],
      ],
      ignoreDiagonal: true,
    });
    expect(texts(place, "data-legend-tick")).toEqual([
      "≥ 2",
      "1",
      "0",
      "−1",
      "−2",
    ]);
    expect(cellMiddle(place, "0", "0").fill).toBe("#7d1517");
    expect(cellMiddle(place, "1", "0").fill).toBe("#0d366b");
    heatmap(place, { rows: ["a"], values: [[1]] });
    expect(place.querySelectorAll("[data-chat-chart]").length).toBe(1);
    expect(texts(place, "data-row-label")).toEqual(["a"]);
  });

  it("says clearly what is wrong with the options", () => {
    const place = document.createElement("div");
    expect(() =>
      heatmap(place, { rows: ["0", "1"], values: [[1, 2]] }),
    ).toThrow("chat.heatmap: values must be a list of 2 rows");
    expect(() =>
      heatmap(place, { rows: ["0", "1"], values: [[1, 2], [3]] }),
    ).toThrow("chat.heatmap: values[1] has 1 numbers, but there are 2 columns");
    expect(() => heatmap("#chart", { rows: ["0"], values: [[1]] })).toThrow(
      "chat.heatmap: the first argument must be an element",
    );
    expect(place.children.length).toBe(0);
  });
});

describe("chat.bars and chat.scatter", () => {
  it("draws bars from 0 with numbers on the axis, in number order", () => {
    const place = document.createElement("div");
    const clicks: unknown[] = [];
    const chart = bars(place, {
      labels: [3, 1, 2],
      values: [30, -10, 20],
      valueTitle: "Cells",
      labelTitle: "Cluster",
      onClick: (bar: unknown) => clicks.push(bar),
    });
    expect(texts(place, "data-x-label")).toEqual(["1", "2", "3"]);
    expect(texts(place, "data-y-tick")).toContain("0");
    expect(texts(place, "data-y-tick")).toContain("−10");
    const bar = place.querySelector('[data-bar="3"]')!;
    const x = Number(bar.getAttribute("x")) + 2;
    const y = Number(bar.getAttribute("y")) + 2;
    pointAt(place, x, y);
    expect(tooltipOf(place)).toEqual(["Cells: 30", "Cluster 3"]);
    pointAt(place, x, y, "click");
    expect(clicks).toEqual([{ label: "3", value: 30, index: 0 }]);
    expect(chart.csv()).toBe("Cluster,Cells\n1,-10\n2,20\n3,30\n");
  });

  it("colors points by group with a legend, and refuses too many points", () => {
    const place = document.createElement("div");
    scatter(place, {
      x: [0, 1, 2, 3],
      y: [0, 1, 4, 9],
      groups: ["10", "2", "2", "10"],
      xTitle: "UMAP 1",
      yTitle: "UMAP 2",
      groupTitle: "Cluster",
    });
    expect(texts(place, "data-legend-item")).toEqual(["2", "10"]);
    const fills = [...place.querySelectorAll("circle")].map((c) =>
      c.getAttribute("fill"),
    );
    expect(fills[0]).toBe(fills[3]);
    expect(fills[0]).not.toBe(fills[1]);
    const point = place.querySelectorAll("circle")[2]!;
    pointAt(
      place,
      Number(point.getAttribute("cx")) + 1,
      Number(point.getAttribute("cy")),
    );
    expect(tooltipOf(place)).toEqual(["UMAP 1 2, UMAP 2 4", "Cluster 2"]);
    expect(() =>
      scatter(place, {
        x: new Array(20_001).fill(0),
        y: new Array(20_001).fill(0),
      }),
    ).toThrow("chat.scatter: 20001 points are too many");
  });
});

describe("chat.legend", () => {
  it("shows the colors on the image in number order, with Reset colors", async () => {
    const place = document.createElement("div");
    const resets: number[] = [];
    const shown = legend(place, {
      title: "Colors on the image",
      items: [
        { label: "10", color: "#e34948" },
        { label: 2, color: { r: 42, g: 120, b: 214 } },
      ],
      onReset: () => resets.push(1),
    });
    expect(
      [...place.querySelectorAll("[data-legend-item]")].map(
        (item) => item.textContent,
      ),
    ).toEqual(["2", "10"]);
    const swatch = place.querySelector<HTMLElement>(
      '[data-legend-item="2"] span',
    )!;
    expect(swatch.style.background).toBe("rgb(42, 120, 214)");
    place.querySelector("button")!.click();
    await vi.waitFor(() => expect(resets).toEqual([1]));
    shown.update({ items: { 3: "red" }, onReset: undefined });
    expect(place.querySelector("button")).toBeNull();
    expect(place.textContent).toContain("Colors on the image3");
    expect(() => legend(place, { items: [{ label: "1", color: 5 }] })).toThrow(
      'chat.legend: the color of "1" must be a CSS color',
    );
  });
});
