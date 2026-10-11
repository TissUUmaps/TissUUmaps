import { deepEqual } from "fast-equals";
import { type Dispatch, type SetStateAction, useEffect, useState } from "react";

/**
 * Backs an editor with local draft state that follows the value it edits
 *
 * The state holds what the editor shows, including drafts that do not make a
 * complete value yet. Whenever the state implies a value that differs from the
 * current one, it is written back. A value changed elsewhere resets the state,
 * whereas one written by the editor itself leaves it alone, so that drafts not
 * reflected in the value are kept.
 *
 * @param value - The value the editor edits
 * @param onValueChange - Called with the value the state implies
 * @param valueToState - Returns the state that shows a value
 * @param stateToValue - Returns the value a state implies, updating the given
 * one, or `null` if the state is incomplete
 * @returns The state and its setter
 */
export function useDraftState<TValue, TState>(
  value: TValue,
  onValueChange: (value: TValue) => void,
  valueToState: (value: TValue) => TState,
  stateToValue: (state: TState, value: TValue) => TValue | null,
): [TState, Dispatch<SetStateAction<TState>>] {
  const [state, setState] = useState(() => valueToState(value));

  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevValue, setPrevValue] = useState(value);
  if (value !== prevValue) {
    setPrevValue(value);
    if (!deepEqual(value, stateToValue(state, prevValue))) {
      setState(valueToState(value));
    }
  }

  useEffect(() => {
    const nextValue = stateToValue(state, value);
    // compared in the shape the editor writes, as a value may omit fields
    // that the editor fills in (e.g. optional or defaulted ones)
    const currentValue = stateToValue(valueToState(value), value);
    if (nextValue !== null && !deepEqual(nextValue, currentValue)) {
      onValueChange(nextValue);
    }
  }, [state, value, onValueChange, valueToState, stateToValue]);

  return [state, setState];
}
