import { useCallback, useLayoutEffect, useRef } from "react";

/**
 * Returns a function that keeps its identity and calls the latest callback
 *
 * For callbacks that are created on every render but only called later, e.g.
 * on an edit, so that they do not invalidate the values memoized with them.
 *
 * @param callback - The callback, as of the current render
 * @returns A function that calls the callback of the latest render
 */
export function useLatestCallback<TArgs extends unknown[], TResult>(
  callback: (...args: TArgs) => TResult,
): (...args: TArgs) => TResult {
  const callbackRef = useRef(callback);
  useLayoutEffect(() => {
    callbackRef.current = callback;
  });
  return useCallback((...args: TArgs) => callbackRef.current(...args), []);
}
