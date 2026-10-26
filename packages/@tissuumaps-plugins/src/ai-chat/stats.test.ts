import { describe, expect, it } from "vitest";

import {
  benjaminiHochberg,
  formatP,
  log2FoldChange,
  mannWhitney,
  normalUpperTail,
  ranks,
  spearman,
  studentTwoSided,
  summarize,
} from "./stats";

// The expected values were computed with SciPy 1.15.3 (and NumPy 2.1.3),
// unless the test says they are by hand.

/** Expects a number close to another, relative to its size */
function expectClose(actual: number, expected: number, relative = 1e-6) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
    relative * Math.abs(expected) + 1e-12,
  );
}

/** n numbers from a formula of their index */
function numbers(n: number, formula: (i: number) => number): number[] {
  return Array.from({ length: n }, (_, i) => formula(i));
}

describe("stats", () => {
  it("summarizes numbers, without the missing ones (by hand)", () => {
    expect(summarize([3, 0, NaN, 1, 8])).toEqual({
      count: 4,
      min: 0,
      max: 8,
      mean: 3,
      median: 2,
      positive: 0.75,
    });
    expect(summarize(new Float32Array([5, 1, 3])).median).toBe(3);
    const empty = summarize([NaN]);
    expect(empty.count).toBe(0);
    expect(empty.mean).toBeNaN();
  });

  it("ranks with the mean rank for ties (by hand)", () => {
    expect([...ranks([10, 20, 20, 5, 20])]).toEqual([2, 4, 4, 1, 4]);
  });

  it("gives the tail of the normal distribution, also far out", () => {
    for (const [z, tail] of [
      [0, 0.5],
      [1.96, 0.024997895148220435],
      [5, 2.866515718791933e-7],
      [10, 7.61985302416047e-24],
      [20, 2.7536241186061556e-89],
      [37, 5.7255712225239266e-300],
    ] as const) {
      expectClose(normalUpperTail(z), tail);
    }
  });

  it("tests with Wilcoxon rank-sum (Mann-Whitney U) like SciPy, with ties", () => {
    const cases: [number[], number[], number, number, number, number][] = [
      [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        0,
        0.012185780355344813,
        -2.6111648393354674,
        0,
      ],
      [
        [1, 2, 2, 3, 3, 3, 0, 0],
        [2, 3, 4, 4, 5, 0, 1],
        18,
        0.2628903364198475,
        -1.1785113019775793,
        0.32142857142857145,
      ],
      [
        numbers(60, (i) => i % 7),
        numbers(45, (i) => (i * 3) % 11),
        871.5,
        0.0018396598057363836,
        -3.1182245462405374,
        0.3227777777777778,
      ],
      // mostly zeros, like the expression of a gene
      [
        numbers(2000, (i) => (i % 3 ? (i * 7) % 13 : 0)),
        numbers(3000, (i) => (i % 2 ? (i * 5) % 11 : 0)),
        3622380.5,
        7.591263401223231e-40,
        13.210909317369678,
        0.6037300833333333,
      ],
    ];
    for (const [x, y, u, p, z, auc] of cases) {
      const test = mannWhitney(x, y);
      expect(test.u).toBe(u);
      expectClose(test.p, p);
      expectClose(test.z, z, 1e-12);
      expectClose(test.auc, auc, 1e-12);
    }
    // all equal: nothing to tell apart (by hand)
    expect(mannWhitney([2, 2], [2, 2, 2])).toMatchObject({ z: 0, p: 1 });
    expect(mannWhitney([], [1]).p).toBeNaN();
  });

  it("adjusts p-values with Benjamini-Hochberg like SciPy", () => {
    const adjusted = benjaminiHochberg([0.01, 0.04, 0.03, 0.005, 0.2, 0.5]);
    [0.03, 0.06, 0.06, 0.03, 0.24, 0.5].forEach((expected, i) => {
      expectClose(adjusted[i] as number, expected, 1e-12);
    });
  });

  it("gives the two-sided p-value of Student's t like SciPy", () => {
    for (const [t, df, p] of [
      [2, 5, 0.10193947882985828],
      [3.5, 30, 0.0014768074376442554],
      [1.2, 1000, 0.23042355232446715],
      [4, 150000, 6.337282367911805e-5],
      [0.3, 7, 0.7728900504021343],
      [50, 150000, 0],
    ] as const) {
      expectClose(studentTwoSided(t, df), p);
    }
  });

  it("correlates with Spearman like SciPy, also with ties and many pairs", () => {
    const cases: [number[], number[], number, number][] = [
      [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [2, 1, 4, 3, 6, 5, 8, 7, 10, 9],
        0.9393939393939393,
        5.484052998513666e-5,
      ],
      [
        [1, 2, 2, 3, 4, 4, 4, 5, 6, 7],
        [3, 1, 2, 2, 5, 4, 6, 6, 5, 9],
        0.8292085416561719,
        0.0030125971055099602,
      ],
      [
        numbers(200, (i) => (i * 37) % 101),
        numbers(200, (i) => ((i * 37) % 101) * 0.3 + ((i * 53) % 97)),
        0.32968523192224425,
        1.868145483300742e-6,
      ],
      [
        numbers(100000, (i) => (i * 37) % 1009),
        numbers(100000, (i) => ((i * 37) % 1009) * 0.05 + ((i * 53) % 997)),
        0.04922375573658676,
        1.0737563900107502e-54,
      ],
    ];
    for (const [x, y, rho, p] of cases) {
      const correlation = spearman(x, y);
      expectClose(correlation.rho, rho, 1e-9);
      expectClose(correlation.p, p);
    }
    // missing values are left out, constant lists have no correlation (by hand)
    expect(spearman([1, NaN, 2, 3], [1, 5, 2, 3])).toEqual({
      n: 3,
      rho: 1,
      p: 0,
    });
    expect(spearman([1, 1, 1], [1, 2, 3]).rho).toBeNaN();
  });

  it("computes the log2 fold change like scanpy, for counts and for log1p values (by hand)", () => {
    expectClose(log2FoldChange(4, 1, false), 2, 1e-9);
    expectClose(log2FoldChange(Math.log1p(4), Math.log1p(1), true), 2, 1e-9);
    // a gene not expressed outside the group: large, but finite
    expect(log2FoldChange(1, 0, false)).toBeCloseTo(29.9, 1);
  });

  it("writes p-values readably", () => {
    expect(formatP(0.04213)).toBe("0.042");
    expect(formatP(3.2e-45)).toBe("3.2e-45");
    expect(formatP(0)).toBe("< 1e-300");
    expect(formatP(1)).toBe("1");
  });
});
