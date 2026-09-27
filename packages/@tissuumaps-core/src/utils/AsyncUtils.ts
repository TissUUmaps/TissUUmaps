/**
 * Utility methods for running large CPU workloads on the main thread without
 * blocking the UI, by cooperatively yielding to the event loop
 */
export class AsyncUtils {
  // Shared MessageChannel used as a macrotask scheduler when `scheduler.yield()`
  // is unavailable. A single channel with a FIFO queue of resolvers avoids
  // allocating a channel per yield. MessageChannel delivers one message per
  // `postMessage`, in order, so the n-th message resolves the n-th waiter.
  private static _messageChannel: MessageChannel | undefined;
  private static _pendingResolvers: (() => void)[] = [];

  /**
   * Yields control back to the event loop, allowing the browser to paint and
   * process input before resuming.
   *
   * Uses the native `scheduler.yield()` when available, otherwise falls back to
   * a shared `MessageChannel` (a macrotask without `setTimeout`'s clamping).
   *
   * @returns A promise that resolves once the event loop has been yielded to
   */
  static yield(): Promise<void> {
    const scheduler = (
      globalThis as { scheduler?: { yield?: () => Promise<void> } }
    ).scheduler;
    if (typeof scheduler?.yield === "function") {
      return scheduler.yield();
    }
    if (AsyncUtils._messageChannel === undefined) {
      AsyncUtils._messageChannel = new MessageChannel();
      AsyncUtils._messageChannel.port1.onmessage = () => {
        AsyncUtils._pendingResolvers.shift()?.();
      };
    }
    return new Promise<void>((resolve) => {
      AsyncUtils._pendingResolvers.push(resolve);
      AsyncUtils._messageChannel!.port2.postMessage(undefined);
    });
  }

  /**
   * Creates a time-sliced yielder for loops whose iterations are too coarse or
   * heterogeneous for {@link AsyncUtils.forEach} (e.g. nested loops).
   *
   * Call the returned function at safe points inside a loop: it checks the
   * abort signal on every call for fail-fast cancellation, and additionally
   * yields to the event loop once the time budget since the last yield is
   * exceeded. When within budget and not aborted, it is a cheap no-op.
   *
   * @param options - Optional time (ms) to run before yielding
   * @returns The yielder, which takes an optional abort signal
   */
  static createYielder(options?: {
    yieldMs?: number;
  }): (opts?: { signal?: AbortSignal }) => Promise<void> {
    const { yieldMs = 5 } = options ?? {};
    let start = performance.now();
    return async (opts) => {
      const { signal } = opts ?? {};
      signal?.throwIfAborted();
      if (performance.now() - start > yieldMs) {
        await AsyncUtils.yield();
        signal?.throwIfAborted(); // yield() does not throw on abort
        start = performance.now();
      }
    };
  }

  /**
   * Invokes `callback(item, index)` for every element of `items`, yielding to
   * the event loop whenever the time budget is exceeded so the UI stays
   * responsive.
   *
   * The abort signal is checked up front and after every yield, preserving
   * fail-fast cancellation: an abort rejects with the signal's reason and no
   * further iterations run.
   *
   * @param items - The array-like collection to iterate over
   * @param callback - Synchronous work to perform for each `item` and its `index`
   * @param options - Optional abort signal, time (ms) to run before yielding
   *   (`yieldMs`), and the number of iterations between budget checks
   *   (`checkEvery`)
   * @returns A promise that resolves once every item has been visited, or
   *   rejects with the signal's reason
   */
  static async forEach<T>(
    items: ArrayLike<T>,
    callback: (item: T, index: number) => void,
    options?: {
      signal?: AbortSignal;
      yieldMs?: number;
      checkEvery?: number;
    },
  ): Promise<void> {
    const { signal, yieldMs, checkEvery = 1024 } = options ?? {};
    signal?.throwIfAborted();
    const maybeYield = AsyncUtils.createYielder({ yieldMs });
    for (let i = 0; i < items.length; i++) {
      callback(items[i]!, i);
      if ((i + 1) % checkEvery === 0 && i + 1 < items.length) {
        await maybeYield({ signal });
      }
    }
  }

  /**
   * Returns a promise that resolves or rejects with the given `promise`, but
   * rejects early if the given `signal` is aborted.
   *
   * @param promise - The promise to race against the abort signal
   * @param options - Optional abort signal to race against the promise
   * @returns A promise that resolves or rejects based on the race condition
   */
  static async raceSignal<T>(
    promise: Promise<T>,
    options?: { signal?: AbortSignal },
  ): Promise<T> {
    const { signal } = options ?? {};
    if (signal === undefined) {
      return promise;
    }
    signal.throwIfAborted(); // fail-fast if already aborted
    return new Promise((resolve, reject) => {
      const onAbort = () => reject(signal.reason as DOMException);
      signal.addEventListener("abort", onAbort, { once: true });
      promise.then(
        (value) => {
          signal.removeEventListener("abort", onAbort);
          resolve(value);
        },
        (error) => {
          signal.removeEventListener("abort", onAbort);
          reject(error as Error);
        },
      );
    });
  }

  /**
   * Creates a promise together with the functions that resolve and reject it,
   * like `Promise.withResolvers()`, which is not yet widely available.
   *
   * @returns The promise and the functions that resolve and reject it
   */
  static withResolvers<T>(): {
    promise: Promise<T>;
    resolve: (value: T | PromiseLike<T>) => void;
    reject: (reason?: unknown) => void;
  } {
    let resolve!: (value: T | PromiseLike<T>) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((resolvePromise, rejectPromise) => {
      resolve = resolvePromise;
      reject = rejectPromise;
    });
    return { promise, resolve, reject };
  }
}
