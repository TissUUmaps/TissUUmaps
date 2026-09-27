import { afterEach, describe, expect, it, vi } from "vitest";

import { AsyncUtils } from "./AsyncUtils";

describe("AsyncUtils.yield", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("delegates to the native scheduler when available", async () => {
    const schedulerYield = vi.fn(() => Promise.resolve());
    vi.stubGlobal("scheduler", { yield: schedulerYield });
    await expect(AsyncUtils.yield()).resolves.toBeUndefined();
    expect(schedulerYield).toHaveBeenCalledOnce();
  });

  it("falls back to the message channel when the scheduler has no yield", async () => {
    vi.stubGlobal("scheduler", {});
    await expect(AsyncUtils.yield()).resolves.toBeUndefined();
  });

  it("resolves", async () => {
    await expect(AsyncUtils.yield()).resolves.toBeUndefined();
  });

  it("resolves concurrent waiters in order", async () => {
    const order: number[] = [];
    await Promise.all([
      AsyncUtils.yield().then(() => order.push(0)),
      AsyncUtils.yield().then(() => order.push(1)),
      AsyncUtils.yield().then(() => order.push(2)),
    ]);
    expect(order).toEqual([0, 1, 2]);
  });
});

describe("AsyncUtils.forEach", () => {
  it("invokes body with each item and index in order", async () => {
    const seen: [string, number][] = [];
    await AsyncUtils.forEach(["a", "b", "c"], (item, i) => {
      seen.push([item, i]);
    });
    expect(seen).toEqual([
      ["a", 0],
      ["b", 1],
      ["c", 2],
    ]);
  });

  it("does nothing for an empty collection", async () => {
    const body = vi.fn();
    await AsyncUtils.forEach([], body);
    expect(body).not.toHaveBeenCalled();
  });

  it("supports typed arrays", async () => {
    const seen: number[] = [];
    await AsyncUtils.forEach(new Uint32Array([10, 20, 30]), (item) => {
      seen.push(item);
    });
    expect(seen).toEqual([10, 20, 30]);
  });

  it("covers items spanning multiple chunks", async () => {
    const items = Array.from({ length: 11 }, (_, i) => i);
    let count = 0;
    let last = -1;
    await AsyncUtils.forEach(
      items,
      (item, i) => {
        expect(i).toBe(last + 1);
        expect(item).toBe(i);
        last = i;
        count++;
      },
      { yieldMs: -1, checkEvery: 4 },
    );
    expect(count).toBe(items.length);
    expect(last).toBe(items.length - 1);
  });

  it("throws immediately when the signal is already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    const body = vi.fn();
    await expect(
      AsyncUtils.forEach([1, 2, 3], body, { signal: controller.signal }),
    ).rejects.toThrow();
    expect(body).not.toHaveBeenCalled();
  });

  it("stops and rejects when aborted mid-iteration", async () => {
    const controller = new AbortController();
    const seen: number[] = [];
    const items = Array.from({ length: 16 }, (_, i) => i);
    const promise = AsyncUtils.forEach(
      items,
      (item) => {
        seen.push(item);
        if (item === 3) {
          controller.abort();
        }
      },
      { signal: controller.signal, yieldMs: -1, checkEvery: 4 },
    );
    await expect(promise).rejects.toThrow();
    // The first chunk (4 items) runs fully, then the post-yield abort check stops it.
    expect(seen.length).toBe(4);
  });
});

describe("AsyncUtils.createYielder", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("yields only past the default 5 ms budget when no options are given", async () => {
    const yieldSpy = vi.spyOn(AsyncUtils, "yield");
    const now = vi.spyOn(performance, "now");

    now.mockReturnValueOnce(0);
    const yielder = AsyncUtils.createYielder();

    now.mockReturnValueOnce(4);
    await yielder();
    expect(yieldSpy).not.toHaveBeenCalled();

    now.mockReturnValueOnce(6);
    await yielder();
    expect(yieldSpy).toHaveBeenCalledOnce();
  });

  it("yields when the time budget is exceeded", async () => {
    const yielder = AsyncUtils.createYielder({ yieldMs: -1 });
    // yieldMs < 0 means every call exceeds the budget and yields.
    await expect(yielder()).resolves.toBeUndefined();
  });

  it("is a no-op while within the time budget", async () => {
    const yielder = AsyncUtils.createYielder({ yieldMs: 60_000 });
    await expect(yielder()).resolves.toBeUndefined();
  });

  it("throws on the abort check after yielding", async () => {
    const controller = new AbortController();
    // abort while yielding to reach the abort check after the yield
    vi.spyOn(AsyncUtils, "yield").mockImplementation(() => {
      controller.abort();
      return Promise.resolve();
    });
    const yielder = AsyncUtils.createYielder({ yieldMs: -1 });
    await expect(yielder({ signal: controller.signal })).rejects.toThrow();
  });

  it("throws on the abort check even while within budget", async () => {
    const controller = new AbortController();
    const yielder = AsyncUtils.createYielder({ yieldMs: 60_000 });
    controller.abort();
    await expect(yielder({ signal: controller.signal })).rejects.toThrow();
  });
});

describe("AsyncUtils.raceSignal", () => {
  it("resolves with the promise's value when no signal is given", async () => {
    await expect(AsyncUtils.raceSignal(Promise.resolve("value"))).resolves.toBe(
      "value",
    );
  });

  it("rejects with the promise's error when no signal is given", async () => {
    const error = new Error("boom");
    await expect(AsyncUtils.raceSignal(Promise.reject(error))).rejects.toBe(
      error,
    );
  });

  it("resolves with the promise's value when the signal is not aborted", async () => {
    const controller = new AbortController();
    await expect(
      AsyncUtils.raceSignal(Promise.resolve("value"), {
        signal: controller.signal,
      }),
    ).resolves.toBe("value");
  });

  it("rejects with the promise's error when the signal is not aborted", async () => {
    const controller = new AbortController();
    const error = new Error("boom");
    await expect(
      AsyncUtils.raceSignal(Promise.reject(error), {
        signal: controller.signal,
      }),
    ).rejects.toBe(error);
  });

  it("passes a non-Error rejection reason through unchanged", async () => {
    const controller = new AbortController();
    // rejecting with a non-Error is exactly what this test exercises
    // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
    const rejected = Promise.reject("boom");
    await expect(
      AsyncUtils.raceSignal(rejected, { signal: controller.signal }),
    ).rejects.toBe("boom");
  });

  it("throws immediately when the signal is already aborted", async () => {
    const controller = new AbortController();
    const error = new Error("cancelled");
    controller.abort(error);
    await expect(
      AsyncUtils.raceSignal(new Promise<never>(() => {}), {
        signal: controller.signal,
      }),
    ).rejects.toBe(error);
  });

  it("rejects with the abort reason when aborted while pending", async () => {
    const controller = new AbortController();
    const error = new Error("cancelled");
    const promise = AsyncUtils.raceSignal(new Promise<never>(() => {}), {
      signal: controller.signal,
    });
    controller.abort(error);
    await expect(promise).rejects.toBe(error);
  });

  it("rejects with an AbortError when aborted without a reason", async () => {
    const controller = new AbortController();
    const promise = AsyncUtils.raceSignal(new Promise<never>(() => {}), {
      signal: controller.signal,
    });
    controller.abort();
    await expect(promise).rejects.toMatchObject({ name: "AbortError" });
  });

  it("ignores an abort that happens after the promise settled", async () => {
    const controller = new AbortController();
    const promise = AsyncUtils.raceSignal(Promise.resolve("value"), {
      signal: controller.signal,
    });
    await expect(promise).resolves.toBe("value");
    controller.abort();
    await expect(promise).resolves.toBe("value");
  });

  it("ignores an abort that happens after the promise rejected", async () => {
    const controller = new AbortController();
    const error = new Error("boom");
    const promise = AsyncUtils.raceSignal(Promise.reject(error), {
      signal: controller.signal,
    });
    await expect(promise).rejects.toBe(error);
    controller.abort(new Error("cancelled"));
    await expect(promise).rejects.toBe(error);
  });

  it("removes the abort listener once the promise resolved", async () => {
    const controller = new AbortController();
    const removeEventListener = vi.spyOn(
      controller.signal,
      "removeEventListener",
    );
    await AsyncUtils.raceSignal(Promise.resolve("value"), {
      signal: controller.signal,
    });
    expect(removeEventListener).toHaveBeenCalledWith(
      "abort",
      expect.any(Function),
    );
  });

  it("removes the abort listener once the promise rejected", async () => {
    const controller = new AbortController();
    const removeEventListener = vi.spyOn(
      controller.signal,
      "removeEventListener",
    );
    await expect(
      AsyncUtils.raceSignal(Promise.reject(new Error("boom")), {
        signal: controller.signal,
      }),
    ).rejects.toThrow("boom");
    expect(removeEventListener).toHaveBeenCalledWith(
      "abort",
      expect.any(Function),
    );
  });
});

describe("AsyncUtils.withResolvers", () => {
  it("resolves the promise with the value passed to resolve", async () => {
    const { promise, resolve } = AsyncUtils.withResolvers<string>();
    resolve("value");
    await expect(promise).resolves.toBe("value");
  });

  it("rejects the promise with the reason passed to reject", async () => {
    const { promise, reject } = AsyncUtils.withResolvers<string>();
    reject(new Error("boom"));
    await expect(promise).rejects.toThrow("boom");
  });
});
