import type { TypedArray, TypedArrayOrArray } from "../types/arrays";
import { AsyncUtils } from "./AsyncUtils";
import { RandomUtils } from "./RandomUtils";

/**
 * Utility methods for numeric clamping, remapping, alignment, rounding,
 * medians, histograms and counts
 */
export class MathUtils {
  /**
   * Clamps a value to the range `[min, max]`
   *
   * @param value - The value to clamp
   * @param min - Lower bound
   * @param max - Upper bound
   * @returns The clamped value
   */
  static clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(min, value), max);
  }

  /**
   * Linearly maps a value from one range to another
   *
   * Values outside `from` are extrapolated, not clamped; clamp at the call
   * site where the domain is known. Swapping `from` and `to` yields the
   * inverse mapping.
   *
   * @param value - The value to map
   * @param from - The source range, as `[min, max]`, with `min !== max`
   * @param to - The target range, as `[min, max]`
   * @returns The mapped value
   * @throws Error if `from` is degenerate
   */
  static remap(
    value: number,
    from: [number, number],
    to: [number, number],
  ): number {
    const [fromMin, fromMax] = from;
    const [toMin, toMax] = to;
    if (fromMin === fromMax) {
      throw new Error("from must not be degenerate");
    }
    return toMin + ((value - fromMin) / (fromMax - fromMin)) * (toMax - toMin);
  }

  /**
   * Aligns a positive number `n` to the next multiple of `m`
   *
   * If `n` is already a multiple of `m`, returns `n`.
   * Otherwise, returns the smallest multiple of `m` that is greater than `n`.
   *
   * @param n - The non-negative number to align
   * @param m - The strictly positive multiple to align to
   * @returns The aligned number
   * @throws Error if `n` is negative, or if `m` is not strictly positive
   */
  static align(n: number, m: number): number {
    if (n < 0) {
      throw new Error("n must be non-negative");
    }
    if (m <= 0) {
      throw new Error("m must be strictly positive");
    }
    if (n % m === 0) {
      return n;
    }
    return Math.ceil(n / m) * m;
  }

  /**
   * Rounds a value to the decimal places that a step needs
   *
   * Steps of `1` or more round to whole numbers, `0.1` to one decimal place,
   * `0.05` to two. Stepping through a range adds floating-point noise that this
   * rounds away, e.g. `0.30000000000000004` becomes `0.3` for a step of `0.1`.
   *
   * @param value - The value to round
   * @param step - The strictly positive step
   * @returns The rounded value
   * @throws Error if `step` is not strictly positive
   */
  static roundToStepDecimals(value: number, step: number): number {
    if (step <= 0) {
      throw new Error("step must be strictly positive");
    }
    const decimals = Math.max(0, -Math.floor(Math.log10(step)));
    return Number(value.toFixed(decimals));
  }

  /**
   * Computes the weighted median of numeric values
   *
   * Returns the smallest value whose cumulative weight, in ascending order of
   * the values, reaches half the total weight. If the total weight is zero,
   * all values are weighted equally instead.
   *
   * @param values - The non-empty values to compute the weighted median of
   * @param weights - The non-negative weight of each value
   * @returns The weighted median
   * @throws Error if `values` is empty, or if `weights` has a different length
   */
  static computeWeightedMedian(
    values: TypedArray,
    weights: TypedArray,
  ): number {
    if (values.length === 0) {
      throw new Error("values must not be empty");
    }
    if (weights.length !== values.length) {
      throw new Error("weights must have the same length as values");
    }
    const order = new Uint32Array(values.length).map((_, i) => i);
    order.sort((i, j) => values[i]! - values[j]!);
    let totalWeight = 0;
    for (let i = 0; i < weights.length; i++) {
      totalWeight += weights[i]!;
    }
    const weighted = totalWeight > 0;
    const halfWeight = 0.5 * (weighted ? totalWeight : values.length);
    let cumulativeWeight = 0;
    for (const i of order) {
      cumulativeWeight += weighted ? weights[i]! : 1;
      if (cumulativeWeight >= halfWeight) {
        return values[i]!;
      }
    }
    return values[order[order.length - 1]!]!; // unreachable but for rounding errors
  }

  /**
   * Computes the range of numeric values, as their minimum and maximum
   *
   * Non-finite values (`NaN`, infinities) are ignored. If no finite value is
   * found (e.g. for empty arrays), the empty range `[Infinity, -Infinity]` is
   * returned; callers can detect it by checking that the lower bound exceeds
   * the upper bound.
   *
   * The values are traversed on the main thread, yielding to the event loop
   * periodically so the UI stays responsive for large arrays (see
   * {@link AsyncUtils.forEach}). Aborting the signal rejects with its reason.
   *
   * @param values - The values to compute the range of
   * @param options - Optional abort signal
   * @returns A promise that resolves to the range, as `[min, max]`
   */
  static async computeRange(
    values: TypedArray,
    options?: { signal?: AbortSignal },
  ): Promise<[number, number]> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    let vmin = Infinity;
    let vmax = -Infinity;
    await AsyncUtils.forEach(
      values,
      (v) => {
        if (Number.isFinite(v)) {
          if (v < vmin) {
            vmin = v;
          }
          if (v > vmax) {
            vmax = v;
          }
        }
      },
      { signal },
    );
    return [vmin, vmax];
  }

  /**
   * Computes the histogram of numeric values over a given value range
   *
   * `hist[i]` counts the values that are closest to bin `i`, with bins spread
   * evenly over `range`: bin `0` maps to the range's lower bound, the last bin
   * to its upper bound, and each bin in between to `vmin + i / (bins - 1) *
   * (vmax - vmin)`. Values outside the range are counted in the nearest edge
   * bin, non-finite values (`NaN`, infinities) are ignored. If the range is
   * degenerate (upper bound not above lower bound), or if `bins` is `1`, all
   * values fall into bin `0`.
   *
   * If `sample` is positive and smaller than the number of values, the
   * histogram is estimated from `sample` values drawn uniformly with
   * replacement, using the seeded generator of
   * {@link RandomUtils.createUint32RNG}; the counts then sum to `sample`
   * (minus ignored non-finite values) rather than to the number of values.
   * Sampling is deterministic for a given `seed`. Otherwise, every value is
   * counted exactly once.
   *
   * The values are traversed on the main thread, yielding to the event loop
   * periodically so the UI stays responsive for large arrays (see
   * {@link AsyncUtils.forEach}). Aborting the signal rejects with its reason.
   *
   * @param values - The values to compute the histogram of
   * @param range - The value range the bins span, as `[min, max]`
   * @param options - Optional abort signal (`signal`), number of bins
   *   (`bins`, a positive integer, defaults to `1024`), number of values to
   *   sample (`sample`; omitting it, `0`, or at least the number of values
   *   disables sampling), and seed for sampling (`seed`, defaults to `0`)
   * @returns A promise that resolves to the histogram, as bin counts (`hist`)
   * and the given value range (`range`)
   */
  static async computeHistogram(
    values: TypedArray,
    range: [number, number],
    options?: {
      signal?: AbortSignal;
      bins?: number;
      sample?: number;
      seed?: number;
    },
  ): Promise<{ hist: number[]; range: [number, number] }> {
    const { signal, bins = 1024, sample, seed = 0 } = options ?? {};
    signal?.throwIfAborted();
    const [vmin, vmax] = range;
    const hist = new Array<number>(bins).fill(0);
    const scale = vmin < vmax ? (bins - 1) / (vmax - vmin) : 0;
    const rng =
      sample !== undefined && sample > 0 && sample < values.length
        ? RandomUtils.createUint32RNG(seed)
        : undefined;
    await AsyncUtils.forEach(
      { length: Math.min(sample || values.length, values.length) },
      (_, i) => {
        const v = values[rng ? rng() % values.length : i]!;
        if (Number.isFinite(v)) {
          const bin = MathUtils.clamp(
            Math.round((v - vmin) * scale),
            0,
            bins - 1,
          );
          hist[bin]! += 1;
        }
      },
      { signal },
    );
    return { hist, range };
  }

  /**
   * Redistributes the counts of a histogram over equal bins of another range
   *
   * The new bins split `range` into `bins` equal intervals, the upper bound
   * counting in the last one. Each bin of `histogram` is counted in the new bin
   * that its value falls into (see {@link computeHistogram}), or dropped if
   * its value is outside `range`. If `range` is degenerate (upper bound not
   * above lower bound), all counts fall into bin `0`.
   *
   * @param histogram - The histogram to redistribute, as bin counts (`hist`)
   * and the value range they span (`range`)
   * @param range - The value range the new bins span, as `[min, max]`
   * @param bins - The number of new bins, a positive integer
   * @returns The redistributed histogram, as the counts of the new bins
   * (`hist`) and the given value range (`range`)
   */
  static rebinHistogram(
    histogram: { hist: number[]; range: [number, number] },
    range: [number, number],
    bins: number,
  ): { hist: number[]; range: [number, number] } {
    const { hist: origHist, range: origRange } = histogram;
    const [min, max] = range;
    const lastHistogramBin = Math.max(origHist.length - 1, 1);
    const hist = new Array<number>(bins).fill(0);
    if (min >= max) {
      hist[0] = origHist.reduce((sum, count) => sum + count, 0);
      return { hist, range };
    }
    for (let i = 0; i < origHist.length; i++) {
      const value = MathUtils.remap(i, [0, lastHistogramBin], origRange);
      if (value >= min && value <= max) {
        const bin = Math.floor(MathUtils.remap(value, range, [0, bins]));
        hist[Math.min(bin, bins - 1)]! += origHist[i]!;
      }
    }
    return { hist, range };
  }

  /**
   * Counts the number of occurrences of every distinct value
   *
   * The counts are keyed by value, in the order the values first occur.
   * Distinctness follows `Map` key equality, i.e. `NaN` counts as one value
   * and `0` and `-0` as the same one.
   *
   * The values are traversed on the main thread, yielding to the event loop
   * periodically so the UI stays responsive for large arrays (see
   * {@link AsyncUtils.forEach}). Aborting the signal rejects with its reason.
   *
   * @typeParam T - Element type of the values
   * @param values - The values to count
   * @param options - Optional abort signal
   * @returns A promise that resolves to the count of every distinct value, in
   * the order the values first appear
   */
  static async computeUniqueValueCounts<T>(
    values: TypedArrayOrArray<T>,
    options?: { signal?: AbortSignal },
  ): Promise<Map<T, number>> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const counts = new Map<T, number>();
    await AsyncUtils.forEach(
      values as ArrayLike<T>,
      (v) => {
        counts.set(v, (counts.get(v) ?? 0) + 1);
      },
      { signal },
    );
    return counts;
  }
}
