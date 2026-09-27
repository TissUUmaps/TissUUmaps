import { deepEqual } from "fast-equals";
import { type Dispatch, type SetStateAction, useEffect, useState } from "react";

/**
 * Backs a config widget with local state that follows its configuration
 *
 * The state holds what the widget shows, including drafts that do not make a
 * complete configuration yet. Whenever the state implies a configuration that
 * differs from the current one, it is written back. A configuration changed
 * elsewhere resets the state, whereas one written by the widget itself leaves
 * it alone, so that the drafts of the other sources are kept.
 *
 * @param config - The configuration the widget edits
 * @param onConfigChange - Called with the configuration the state implies
 * @param configToState - Returns the state that shows a configuration
 * @param stateToConfig - Returns the configuration a state implies, updating
 * the given one, or `null` if the state is incomplete
 * @returns The state and its setter
 */
export function useConfigWidgetState<TConfig, TState>(
  config: TConfig,
  onConfigChange: (config: TConfig) => void,
  configToState: (config: TConfig) => TState,
  stateToConfig: (state: TState, config: TConfig) => TConfig | null,
): [TState, Dispatch<SetStateAction<TState>>] {
  const [state, setState] = useState(() => configToState(config));

  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevConfig, setPrevConfig] = useState(config);
  if (config !== prevConfig) {
    setPrevConfig(config);
    if (!deepEqual(config, stateToConfig(state, prevConfig))) {
      setState(configToState(config));
    }
  }

  useEffect(() => {
    const nextConfig = stateToConfig(state, config);
    // compared in the shape the widget writes, as a configuration may omit
    // `source` and optional fields
    const currentConfig = stateToConfig(configToState(config), config);
    if (nextConfig !== null && !deepEqual(nextConfig, currentConfig)) {
      onConfigChange(nextConfig);
    }
  }, [state, config, onConfigChange, configToState, stateToConfig]);

  return [state, setState];
}
