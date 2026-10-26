// Statistics for the data tools of the AI chat, without external libraries.
// Each function is checked in stats.test.ts against values computed with
// SciPy, or by hand on small examples.

/** A summary of some numbers */
export type Summary = {
  /** How many numbers, without the missing ones (NaN) */
  count: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  /** The fraction of the numbers above 0, e.g. of the cells that express a gene */
  positive: number;
};

/** The result of a Wilcoxon rank-sum test (Mann-Whitney U) of x against y */
export type RankTest = {
  /** The U statistic of x */
  u: number;
  /**
   * The z-score of x, without continuity correction: above 0 when x tends to
   * be higher than y; scanpy sorts marker genes by it
   */
  z: number;
  /** The two-sided p-value */
  p: number;
  /**
   * The probability that a random value of x is higher than a random value of
   * y, ties counting half: U / (n1 · n2)
   */
  auc: number;
};

/** A Spearman correlation */
export type Correlation = {
  /** The pairs without missing values */
  n: number;
  rho: number;
  /** The two-sided p-value */
  p: number;
};

/**
 * The numbers without the missing ones (NaN), sorted, in a new array
 *
 * @param values - The numbers
 * @returns The sorted numbers
 */
export function sortedNumbers(values: ArrayLike<number>): Float64Array {
  const sorted = new Float64Array(values.length);
  let count = 0;
  for (let i = 0; i < values.length; i++) {
    const value = Number(values[i]);
    if (!Number.isNaN(value)) {
      sorted[count++] = value;
    }
  }
  return sorted.subarray(0, count).sort();
}

/**
 * The median of sorted numbers: the middle one, or the mean of the two in the
 * middle
 *
 * @param sorted - The numbers, sorted
 * @returns The median, or NaN without numbers
 */
export function median(sorted: ArrayLike<number>): number {
  const n = sorted.length;
  if (n === 0) {
    return NaN;
  }
  const half = Math.floor(n / 2);
  return n % 2 === 1
    ? (sorted[half] as number)
    : ((sorted[half - 1] as number) + (sorted[half] as number)) / 2;
}

/**
 * Summarizes some numbers, without the missing ones (NaN)
 *
 * @param values - The numbers
 * @returns Their count, range, mean, median and fraction above 0; NaN for all
 * but the count without numbers
 */
export function summarize(values: ArrayLike<number>): Summary {
  return summarizeSorted(sortedNumbers(values));
}

/**
 * Summarizes sorted numbers without missing ones, see {@link summarize}
 *
 * @param sorted - The numbers, sorted
 * @returns The summary
 */
export function summarizeSorted(sorted: Float64Array): Summary {
  const count = sorted.length;
  let sum = 0;
  let positive = 0;
  for (const value of sorted) {
    sum += value;
    if (value > 0) {
      positive++;
    }
  }
  return {
    count,
    min: count > 0 ? (sorted[0] as number) : NaN,
    max: count > 0 ? (sorted[count - 1] as number) : NaN,
    mean: count > 0 ? sum / count : NaN,
    median: median(sorted),
    positive: count > 0 ? positive / count : NaN,
  };
}

/**
 * The complementary error function, erfc(x) = 1 - erf(x), with a relative
 * error below 1.2e-7 everywhere, also for the tiny values of large x
 * (Numerical Recipes, `erfcc`)
 *
 * @param x - The value
 * @returns erfc(x)
 */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t *
                              (-1.13520398 +
                                t *
                                  (1.48851587 +
                                    t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}

/**
 * The probability that a standard normal value is above z
 *
 * @param z - The value
 * @returns The upper tail probability
 */
export function normalUpperTail(z: number): number {
  return 0.5 * erfc(z / Math.SQRT2);
}

/**
 * The sum of the ranks of x among x and y together, with the mean rank for
 * ties, and the tie term Σ(t³ − t) over the groups of t equal values
 *
 * @param xs - The values of x, sorted
 * @param ys - The values of y, sorted
 * @returns The rank sum of x, and the tie term
 */
function rankSum(
  xs: Float64Array,
  ys: Float64Array,
): { rankSum: number; ties: number } {
  let i = 0;
  let j = 0;
  let position = 0;
  let sum = 0;
  let ties = 0;
  while (i < xs.length || j < ys.length) {
    const value = Math.min(xs[i] ?? Infinity, ys[j] ?? Infinity);
    let inX = 0;
    while (i < xs.length && xs[i] === value) {
      i++;
      inX++;
    }
    let inY = 0;
    while (j < ys.length && ys[j] === value) {
      j++;
      inY++;
    }
    const t = inX + inY;
    // the equal values share the ranks position + 1 to position + t
    sum += inX * (position + (t + 1) / 2);
    ties += t * t * t - t;
    position += t;
  }
  return { rankSum: sum, ties };
}

/**
 * The Wilcoxon rank-sum test (Mann-Whitney U) of x against y, two-sided, with
 * the normal approximation, correcting the variance for ties and the p-value
 * for continuity, as `scipy.stats.mannwhitneyu(x, y, method="asymptotic")`
 *
 * @param xs - The values of x, sorted, without missing ones
 * @param ys - The values of y, sorted, without missing ones
 * @returns The test; NaN if x or y has no values
 */
export function mannWhitneySorted(
  xs: Float64Array,
  ys: Float64Array,
): RankTest {
  const n1 = xs.length;
  const n2 = ys.length;
  if (n1 === 0 || n2 === 0) {
    return { u: NaN, z: NaN, p: NaN, auc: NaN };
  }
  const n = n1 + n2;
  const { rankSum: r1, ties } = rankSum(xs, ys);
  const u = r1 - (n1 * (n1 + 1)) / 2;
  const auc = u / (n1 * n2);
  const mean = (n1 * n2) / 2;
  const variance = ((n1 * n2) / 12) * (n + 1 - ties / (n * (n - 1)));
  // all values are equal
  if (variance <= 0) {
    return { u, z: 0, p: 1, auc };
  }
  const sd = Math.sqrt(variance);
  const corrected = (Math.max(u, n1 * n2 - u) - mean - 0.5) / sd;
  return {
    u,
    z: (u - mean) / sd,
    p: Math.min(1, 2 * normalUpperTail(corrected)),
    auc,
  };
}

/**
 * The Wilcoxon rank-sum test of x against y, see {@link mannWhitneySorted}
 *
 * @param x - The values of x; missing ones (NaN) are left out
 * @param y - The values of y; missing ones (NaN) are left out
 * @returns The test
 */
export function mannWhitney(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
): RankTest {
  return mannWhitneySorted(sortedNumbers(x), sortedNumbers(y));
}

/**
 * Adjusts p-values for testing many hypotheses at once, with the
 * Benjamini-Hochberg procedure, which bounds the false discovery rate, as
 * `scipy.stats.false_discovery_control(p, method="bh")`
 *
 * @param p - The p-values
 * @returns The adjusted p-values, in the same order
 */
export function benjaminiHochberg(p: ArrayLike<number>): Float64Array {
  const m = p.length;
  const order = Array.from({ length: m }, (_, i) => i).sort(
    (a, b) => (p[a] as number) - (p[b] as number),
  );
  const adjusted = new Float64Array(m);
  let smallest = 1;
  for (let k = m - 1; k >= 0; k--) {
    const i = order[k] as number;
    smallest = Math.min(smallest, ((p[i] as number) * m) / (k + 1));
    adjusted[i] = smallest;
  }
  return adjusted;
}

/**
 * The ranks of numbers, from 1, with the mean rank for ties
 *
 * @param values - The numbers, without missing ones
 * @returns The ranks, in the same order
 */
export function ranks(values: ArrayLike<number>): Float64Array {
  const n = values.length;
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    order[i] = i;
  }
  order.sort((a, b) => (values[a] as number) - (values[b] as number));
  const result = new Float64Array(n);
  let start = 0;
  while (start < n) {
    const value = values[order[start] as number];
    let end = start + 1;
    while (end < n && values[order[end] as number] === value) {
      end++;
    }
    // the ranks start + 1 to end, shared
    const rank = (start + 1 + end) / 2;
    for (let k = start; k < end; k++) {
      result[order[k] as number] = rank;
    }
    start = end;
  }
  return result;
}

/**
 * The Pearson correlation of two lists of numbers of the same length
 *
 * @param x - The first numbers
 * @param y - The second numbers
 * @returns The correlation, or NaN if one of the lists is constant
 */
function pearson(x: Float64Array, y: Float64Array): number {
  const n = x.length;
  let meanX = 0;
  let meanY = 0;
  for (let i = 0; i < n; i++) {
    meanX += x[i] as number;
    meanY += y[i] as number;
  }
  meanX /= n;
  meanY /= n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = (x[i] as number) - meanX;
    const dy = (y[i] as number) - meanY;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  return sxx === 0 || syy === 0 ? NaN : sxy / Math.sqrt(sxx * syy);
}

/** The coefficients of the Lanczos approximation of the gamma function (g = 7) */
const lanczos = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/**
 * The natural logarithm of the gamma function, for x > 0
 *
 * @param x - The value
 * @returns ln Γ(x)
 */
function lnGamma(x: number): number {
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
  }
  const shifted = x - 1;
  let sum = lanczos[0] as number;
  for (let i = 1; i < lanczos.length; i++) {
    sum += (lanczos[i] as number) / (shifted + i);
  }
  const t = shifted + 7.5;
  return (
    0.5 * Math.log(2 * Math.PI) +
    (shifted + 0.5) * Math.log(t) -
    t +
    Math.log(sum)
  );
}

/**
 * The continued fraction of the incomplete beta function, by the modified
 * Lentz method (Numerical Recipes, `betacf`)
 */
function betaFraction(x: number, a: number, b: number): number {
  const tiny = 1e-300;
  const clamp = (value: number) => (Math.abs(value) < tiny ? tiny : value);
  let c = 1;
  let d = 1 / clamp(1 - ((a + b) * x) / (a + 1));
  let h = d;
  for (let m = 1; m <= 100_000; m++) {
    const m2 = 2 * m;
    let step = (m * (b - m) * x) / ((a - 1 + m2) * (a + m2));
    d = 1 / clamp(1 + step * d);
    c = clamp(1 + step / c);
    h *= d * c;
    step = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + 1 + m2));
    d = 1 / clamp(1 + step * d);
    c = clamp(1 + step / c);
    const change = d * c;
    h *= change;
    if (Math.abs(change - 1) < 1e-15) {
      break;
    }
  }
  return h;
}

/**
 * The regularized incomplete beta function I_x(a, b)
 *
 * @param x - The value, in [0, 1]
 * @param a - The first shape, above 0
 * @param b - The second shape, above 0
 * @returns I_x(a, b)
 */
function betaIncomplete(x: number, a: number, b: number): number {
  if (x <= 0) {
    return 0;
  }
  if (x >= 1) {
    return 1;
  }
  const front = Math.exp(
    lnGamma(a + b) -
      lnGamma(a) -
      lnGamma(b) +
      a * Math.log(x) +
      b * Math.log1p(-x),
  );
  return x < (a + 1) / (a + b + 2)
    ? (front * betaFraction(x, a, b)) / a
    : 1 - (front * betaFraction(1 - x, b, a)) / b;
}

/**
 * The two-sided p-value of a t statistic with the Student t distribution
 *
 * @param t - The statistic
 * @param df - The degrees of freedom
 * @returns The probability of a statistic at least as far from 0
 */
export function studentTwoSided(t: number, df: number): number {
  return betaIncomplete(df / (df + t * t), df / 2, 0.5);
}

/**
 * The Spearman correlation of two lists of numbers, with the p-value of the
 * t-test of Student, as `scipy.stats.spearmanr(x, y)`; the pairs with a
 * missing value (NaN) are left out
 *
 * @param x - The first numbers
 * @param y - The second numbers, as many
 * @returns The correlation; rho and p are NaN with fewer than 3 pairs or a
 * constant list
 */
export function spearman(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
): Correlation {
  const keptX: number[] = [];
  const keptY: number[] = [];
  for (let i = 0; i < x.length; i++) {
    const a = Number(x[i]);
    const b = Number(y[i]);
    if (!Number.isNaN(a) && !Number.isNaN(b)) {
      keptX.push(a);
      keptY.push(b);
    }
  }
  const n = keptX.length;
  if (n < 3) {
    return { n, rho: NaN, p: NaN };
  }
  const rho = pearson(ranks(keptX), ranks(keptY));
  if (Number.isNaN(rho)) {
    return { n, rho, p: NaN };
  }
  if (Math.abs(rho) >= 1) {
    return { n, rho: Math.sign(rho), p: 0 };
  }
  const df = n - 2;
  const t = rho * Math.sqrt(df / ((1 - rho) * (1 + rho)));
  return { n, rho, p: studentTwoSided(t, df) };
}

/**
 * The log2 fold change of the mean of a group against the mean of the others,
 * as scanpy's `rank_genes_groups` computes it: with values that were
 * log-transformed (log1p), the means are first turned back
 *
 * @param meanIn - The mean in the group
 * @param meanOut - The mean in the others
 * @param logged - Whether the values are log1p-transformed
 * @returns log2((meanIn + 1e-9) / (meanOut + 1e-9)), after expm1 if logged
 */
export function log2FoldChange(
  meanIn: number,
  meanOut: number,
  logged: boolean,
): number {
  const a = logged ? Math.expm1(meanIn) : meanIn;
  const b = logged ? Math.expm1(meanOut) : meanOut;
  return Math.log2((a + 1e-9) / (b + 1e-9));
}

/**
 * Writes a p-value readably: two significant digits, small ones as powers of
 * ten, and "< 1e-300" for those too small for the numbers of JavaScript
 *
 * @param p - The p-value
 * @returns The text, e.g. "0.042", "3.2e-45" or "< 1e-300"
 */
export function formatP(p: number): string {
  if (Number.isNaN(p)) {
    return "not available";
  }
  if (p < 1e-300) {
    return "< 1e-300";
  }
  if (p < 0.001) {
    return p.toExponential(1);
  }
  return String(Number(p.toPrecision(2)));
}
