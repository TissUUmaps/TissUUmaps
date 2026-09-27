import { describe, expect, it } from "vitest";

import { MathUtils } from "./MathUtils";

describe("MathUtils", () => {
  describe("clamp", () => {
    it("returns value when within range", () => {
      expect(MathUtils.clamp(5, 0, 10)).toBe(5);
    });

    it("clamps to min when value is below min", () => {
      expect(MathUtils.clamp(-3, 0, 10)).toBe(0);
    });

    it("clamps to max when value is above max", () => {
      expect(MathUtils.clamp(15, 0, 10)).toBe(10);
    });

    it("works with floats", () => {
      expect(MathUtils.clamp(3.7, 1.2, 4.5)).toBeCloseTo(3.7);
      expect(MathUtils.clamp(0.5, 1.2, 4.5)).toBeCloseTo(1.2);
    });

    it("returns the bound when min equals max", () => {
      expect(MathUtils.clamp(2, 5, 5)).toBe(5);
      expect(MathUtils.clamp(7, 5, 5)).toBe(5);
    });
  });

  describe("remap", () => {
    it("maps the range bounds onto each other", () => {
      expect(MathUtils.remap(0, [0, 1], [10, 20])).toBe(10);
      expect(MathUtils.remap(1, [0, 1], [10, 20])).toBe(20);
    });

    it("maps values linearly", () => {
      expect(MathUtils.remap(0.25, [0, 1], [10, 20])).toBeCloseTo(12.5);
      expect(MathUtils.remap(5, [0, 10], [-1, 1])).toBeCloseTo(0);
    });

    it("is the identity for equal ranges", () => {
      expect(MathUtils.remap(3.7, [1, 5], [1, 5])).toBeCloseTo(3.7);
    });

    it("supports inverted ranges", () => {
      expect(MathUtils.remap(0.25, [0, 1], [1, 0])).toBeCloseTo(0.75);
      expect(MathUtils.remap(0.75, [1, 0], [0, 1])).toBeCloseTo(0.25);
    });

    it("extrapolates outside the source range", () => {
      expect(MathUtils.remap(2, [0, 1], [10, 20])).toBeCloseTo(30);
      expect(MathUtils.remap(-1, [0, 1], [10, 20])).toBeCloseTo(0);
    });

    it("is inverted by swapping the ranges", () => {
      const mapped = MathUtils.remap(0.3, [-0.9, -0.1], [-2, 0]);
      expect(MathUtils.remap(mapped, [-2, 0], [-0.9, -0.1])).toBeCloseTo(0.3);
    });

    it("throws error when the source range is degenerate", () => {
      expect(() => MathUtils.remap(1, [2, 2], [0, 1])).toThrow(
        "from must not be degenerate",
      );
    });
  });

  describe("align", () => {
    it("returns n when n is already a multiple of m", () => {
      expect(MathUtils.align(10, 5)).toBe(10);
      expect(MathUtils.align(0, 5)).toBe(0);
      expect(MathUtils.align(100, 25)).toBe(100);
    });

    it("aligns n to the next multiple of m when not already aligned", () => {
      expect(MathUtils.align(7, 5)).toBe(10);
      expect(MathUtils.align(1, 8)).toBe(8);
      expect(MathUtils.align(33, 16)).toBe(48);
    });

    it("throws error when n is negative", () => {
      expect(() => MathUtils.align(-5, 3)).toThrow("n must be non-negative");
      expect(() => MathUtils.align(-1, 1)).toThrow("n must be non-negative");
    });

    it("throws error when m is zero or negative", () => {
      expect(() => MathUtils.align(5, 0)).toThrow(
        "m must be strictly positive",
      );
      expect(() => MathUtils.align(5, -3)).toThrow(
        "m must be strictly positive",
      );
    });

    it("works with m = 1", () => {
      expect(MathUtils.align(5, 1)).toBe(5);
      expect(MathUtils.align(0, 1)).toBe(0);
    });

    it("works with large numbers", () => {
      expect(MathUtils.align(1000000, 1024)).toBe(1000448);
      expect(MathUtils.align(1000000, 1000000)).toBe(1000000);
    });

    it("aligns to power of 2", () => {
      expect(MathUtils.align(5, 4)).toBe(8);
      expect(MathUtils.align(15, 16)).toBe(16);
      expect(MathUtils.align(256, 512)).toBe(512);
    });
  });

  describe("roundToStepDecimals", () => {
    it("rounds to whole numbers for steps of one or more", () => {
      expect(MathUtils.roundToStepDecimals(12.6, 1)).toBe(13);
      expect(MathUtils.roundToStepDecimals(12.6, 50)).toBe(13);
    });

    it("rounds to the decimal places of the step", () => {
      expect(MathUtils.roundToStepDecimals(0.1 + 0.2, 0.1)).toBe(0.3);
      expect(MathUtils.roundToStepDecimals(0.123, 0.05)).toBe(0.12);
    });

    it("throws error when the step is zero or negative", () => {
      expect(() => MathUtils.roundToStepDecimals(1, 0)).toThrow(
        "step must be strictly positive",
      );
      expect(() => MathUtils.roundToStepDecimals(1, -1)).toThrow(
        "step must be strictly positive",
      );
    });
  });

  describe("computeWeightedMedian", () => {
    it("returns the value at which the cumulative weight reaches half the total", () => {
      expect(
        MathUtils.computeWeightedMedian(
          new Float64Array([3, 1, 2]),
          new Float64Array([1, 1, 1]),
        ),
      ).toBe(2);
      expect(
        MathUtils.computeWeightedMedian(
          new Uint8Array([1, 2, 10]),
          new Uint8Array([1, 1, 40]),
        ),
      ).toBe(10);
    });

    it("returns the smaller value when the cumulative weight is exactly half", () => {
      expect(
        MathUtils.computeWeightedMedian(
          new Uint8Array([1, 2]),
          new Uint8Array([1, 1]),
        ),
      ).toBe(1);
    });

    it("ignores values without weight", () => {
      expect(
        MathUtils.computeWeightedMedian(
          new Uint8Array([0, 5, 7]),
          new Uint8Array([0, 1, 0]),
        ),
      ).toBe(5);
    });

    it("weights all values equally if the total weight is zero", () => {
      expect(
        MathUtils.computeWeightedMedian(
          new Uint8Array([3, 1, 2]),
          new Uint8Array([0, 0, 0]),
        ),
      ).toBe(2);
    });

    it("throws error when values are empty", () => {
      expect(() =>
        MathUtils.computeWeightedMedian(new Uint8Array(0), new Uint8Array(0)),
      ).toThrow("values must not be empty");
    });

    it("throws error when weights have a different length", () => {
      expect(() =>
        MathUtils.computeWeightedMedian(
          new Uint8Array([1, 2]),
          new Uint8Array([1]),
        ),
      ).toThrow("weights must have the same length as values");
    });
  });

  describe("computeRange", () => {
    it("returns the minimum and maximum of integer typed arrays", async () => {
      await expect(
        MathUtils.computeRange(new Int8Array([3, -1, 7, 2])),
      ).resolves.toEqual([-1, 7]);
    });

    it("returns the minimum and maximum of other typed arrays", async () => {
      await expect(
        MathUtils.computeRange(new Float32Array([0.5, -2.5, 1.5])),
      ).resolves.toEqual([-2.5, 1.5]);
      await expect(
        MathUtils.computeRange(new Uint8Array([200, 10, 255, 0])),
      ).resolves.toEqual([0, 255]);
    });

    it("returns a degenerate range for a single value", async () => {
      await expect(
        MathUtils.computeRange(new Uint8Array([4])),
      ).resolves.toEqual([4, 4]);
      await expect(
        MathUtils.computeRange(new Uint8Array([4, 4, 4])),
      ).resolves.toEqual([4, 4]);
    });

    it("ignores non-finite values", async () => {
      await expect(
        MathUtils.computeRange(
          new Float64Array([NaN, 2, Infinity, 4, -Infinity]),
        ),
      ).resolves.toEqual([2, 4]);
    });

    it("returns the empty range when no finite value is found", async () => {
      await expect(MathUtils.computeRange(new Uint8Array(0))).resolves.toEqual([
        Infinity,
        -Infinity,
      ]);
      await expect(
        MathUtils.computeRange(new Float32Array([NaN, Infinity])),
      ).resolves.toEqual([Infinity, -Infinity]);
    });

    it("handles large data", async () => {
      const data = new Uint16Array(100_000).map((_, i) => (i % 1000) + 5);
      await expect(MathUtils.computeRange(data)).resolves.toEqual([5, 1004]);
    });

    it("rejects with the reason of an aborted signal", async () => {
      const controller = new AbortController();
      controller.abort(new Error("aborted"));
      await expect(
        MathUtils.computeRange(new Uint8Array(10), {
          signal: controller.signal,
        }),
      ).rejects.toThrow("aborted");
    });
  });

  describe("computeHistogram", () => {
    it("assigns values to the nearest bin over the given range", async () => {
      const { hist, range } = await MathUtils.computeHistogram(
        new Uint8Array([0, 1, 2, 3, 4]),
        [0, 4],
        { bins: 5 },
      );
      expect(range).toEqual([0, 4]);
      expect(hist).toEqual([1, 1, 1, 1, 1]);
    });

    it("maps the range bounds to the first and last bins", async () => {
      const { hist } = await MathUtils.computeHistogram(
        new Float32Array([-1, -0.9, 0.49, 0.51, 1]),
        [-1, 1],
        { bins: 3 },
      );
      expect(hist).toEqual([2, 1, 2]);
    });

    it("uses 1024 bins by default", async () => {
      const data = new Uint16Array(1024).map((_, i) => i);
      const { hist } = await MathUtils.computeHistogram(data, [0, 1023]);
      expect(hist).toHaveLength(1024);
      expect(hist.every((count) => count === 1)).toBe(true);
    });

    it("counts values outside the range in the edge bins", async () => {
      const { hist } = await MathUtils.computeHistogram(
        new Int8Array([-10, 0, 5, 10, 20]),
        [0, 10],
        { bins: 3 },
      );
      expect(hist).toEqual([2, 1, 2]);
    });

    it("ignores non-finite values", async () => {
      const { hist } = await MathUtils.computeHistogram(
        new Float64Array([NaN, 2, Infinity, 4, -Infinity]),
        [2, 4],
        { bins: 2 },
      );
      expect(hist).toEqual([1, 1]);
    });

    it("puts all values into the first bin for a degenerate range", async () => {
      const { hist, range } = await MathUtils.computeHistogram(
        new Uint8Array([6, 7, 8]),
        [7, 7],
        { bins: 4 },
      );
      expect(range).toEqual([7, 7]);
      expect(hist).toEqual([3, 0, 0, 0]);
    });

    it("puts all values into a single bin", async () => {
      const { hist } = await MathUtils.computeHistogram(
        new Uint8Array([1, 5, 9]),
        [1, 9],
        {
          bins: 1,
        },
      );
      expect(hist).toEqual([3]);
    });

    it("returns zero counts for empty data", async () => {
      await expect(
        MathUtils.computeHistogram(new Uint8Array(0), [0, 255], { bins: 3 }),
      ).resolves.toEqual({ hist: [0, 0, 0], range: [0, 255] });
    });

    it("handles large data", async () => {
      const data = new Uint16Array(100_000).map((_, i) => i % 1000);
      const { hist } = await MathUtils.computeHistogram(data, [0, 999], {
        bins: 1000,
      });
      expect(hist.every((count) => count === 100)).toBe(true);
    });

    it("rejects with the reason of an aborted signal", async () => {
      const controller = new AbortController();
      controller.abort(new Error("aborted"));
      await expect(
        MathUtils.computeHistogram(new Uint8Array(10), [0, 255], {
          bins: 4,
          signal: controller.signal,
        }),
      ).rejects.toThrow("aborted");
    });

    describe("with sample", () => {
      it("counts exactly sample sampled values", async () => {
        const data = new Float32Array(10_000).map(() => Math.random());
        const { hist } = await MathUtils.computeHistogram(data, [0, 1], {
          bins: 16,
          sample: 1000,
        });
        expect(hist.reduce((a, b) => a + b, 0)).toBe(1000);
      });

      it("is deterministic for the same seed", async () => {
        const data = new Float32Array(10_000).map(() => Math.random());
        const a = await MathUtils.computeHistogram(data, [0, 1], {
          bins: 16,
          sample: 1000,
          seed: 7,
        });
        const b = await MathUtils.computeHistogram(data, [0, 1], {
          bins: 16,
          sample: 1000,
          seed: 7,
        });
        const c = await MathUtils.computeHistogram(data, [0, 1], {
          bins: 16,
          sample: 1000,
          seed: 8,
        });
        expect(a.hist).toEqual(b.hist);
        expect(c.hist).not.toEqual(a.hist);
      });

      it("defaults to seed 0", async () => {
        const data = new Float32Array(10_000).map(() => Math.random());
        const a = await MathUtils.computeHistogram(data, [0, 1], {
          bins: 16,
          sample: 1000,
        });
        const b = await MathUtils.computeHistogram(data, [0, 1], {
          bins: 16,
          sample: 1000,
          seed: 0,
        });
        expect(a.hist).toEqual(b.hist);
      });

      it("samples from the whole array", async () => {
        // 10 blocks of 1000 identical values; every block should be hit
        const data = new Uint16Array(10_000).map((_, i) =>
          Math.floor(i / 1000),
        );
        const { hist } = await MathUtils.computeHistogram(data, [0, 9], {
          bins: 10,
          sample: 1000,
        });
        expect(hist.every((count) => count > 50)).toBe(true);
      });

      it("bins fractional and negative values without truncation", async () => {
        const data = new Float64Array(1000).map((_, i) => (i % 2 ? -0.5 : 0.5));
        const { hist } = await MathUtils.computeHistogram(data, [-1, 1], {
          bins: 5,
          sample: 100,
        });
        // -0.5 -> bin 1, 0.5 -> bin 3; truncation to integers would hit bin 2
        expect(hist[0]).toBe(0);
        expect(hist[2]).toBe(0);
        expect(hist[4]).toBe(0);
        expect(hist[1]! + hist[3]!).toBe(100);
      });

      it("ignores non-finite values", async () => {
        const data = new Float64Array(1000).map((_, i) => (i % 2 ? NaN : 3));
        const { hist } = await MathUtils.computeHistogram(data, [2, 4], {
          bins: 3,
          sample: 100,
        });
        expect(hist[0]).toBe(0);
        expect(hist[2]).toBe(0);
        expect(hist[1]).toBeGreaterThan(0);
        expect(hist[1]).toBeLessThan(100);
      });

      it("counts every value once when sample is 0 or not below the length", async () => {
        const data = new Uint8Array([0, 1, 2, 3, 4]);
        const full = await MathUtils.computeHistogram(data, [0, 4], {
          bins: 5,
        });
        for (const sample of [0, 5, 6, 1000]) {
          const { hist } = await MathUtils.computeHistogram(data, [0, 4], {
            bins: 5,
            sample,
          });
          expect(hist).toEqual(full.hist);
        }
      });

      it("returns zero counts for empty data", async () => {
        await expect(
          MathUtils.computeHistogram(new Uint8Array(0), [0, 255], {
            bins: 3,
            sample: 10,
          }),
        ).resolves.toEqual({ hist: [0, 0, 0], range: [0, 255] });
      });

      it("rejects with the reason of an aborted signal", async () => {
        const controller = new AbortController();
        controller.abort(new Error("aborted"));
        await expect(
          MathUtils.computeHistogram(new Uint8Array(10), [0, 255], {
            bins: 4,
            signal: controller.signal,
            sample: 5,
          }),
        ).rejects.toThrow("aborted");
      });
    });
  });

  describe("rebinHistogram", () => {
    it("sums the bins that fall into each new bin", () => {
      expect(
        MathUtils.rebinHistogram(
          { hist: [1, 2, 3, 4], range: [0, 3] },
          [0, 4],
          2,
        ),
      ).toEqual({ hist: [3, 7], range: [0, 4] });
    });

    it("counts the upper bound in the last bin", () => {
      expect(
        MathUtils.rebinHistogram({ hist: [1, 2, 3], range: [0, 2] }, [0, 2], 2),
      ).toEqual({ hist: [1, 5], range: [0, 2] });
    });

    it("drops bins outside the range", () => {
      expect(
        MathUtils.rebinHistogram(
          { hist: [1, 2, 3, 4], range: [0, 3] },
          [1, 2],
          2,
        ),
      ).toEqual({ hist: [2, 3], range: [1, 2] });
    });

    it("puts all counts into the first bin for a degenerate range", () => {
      expect(
        MathUtils.rebinHistogram({ hist: [1, 2, 3], range: [0, 2] }, [5, 5], 3),
      ).toEqual({ hist: [6, 0, 0], range: [5, 5] });
    });
  });

  describe("computeUniqueValueCounts", () => {
    it("counts the occurrences of every distinct value", async () => {
      const counts = await MathUtils.computeUniqueValueCounts(["b", "a", "b"]);
      expect([...counts]).toEqual([
        ["b", 2],
        ["a", 1],
      ]);
    });

    it("counts the values of typed arrays", async () => {
      const counts = await MathUtils.computeUniqueValueCounts(
        new Uint8Array([1, 2, 1]),
      );
      expect([...counts]).toEqual([
        [1, 2],
        [2, 1],
      ]);
    });

    it("returns no counts for empty data", async () => {
      await expect(
        MathUtils.computeUniqueValueCounts(new Uint8Array(0)),
      ).resolves.toEqual(new Map());
    });

    it("handles large data", async () => {
      const values = new Uint16Array(100_000).map((_, i) => i % 1000);
      const counts = await MathUtils.computeUniqueValueCounts(values);
      expect(counts.size).toBe(1000);
      expect([...counts.values()].every((count) => count === 100)).toBe(true);
    });

    it("rejects with the reason of an aborted signal", async () => {
      const controller = new AbortController();
      controller.abort(new Error("aborted"));
      await expect(
        MathUtils.computeUniqueValueCounts(new Uint8Array(10), {
          signal: controller.signal,
        }),
      ).rejects.toThrow("aborted");
    });
  });
});
