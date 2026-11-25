import { useDialogContext } from "../DialogContext";

/** Returns the `ask` function of the nearest `DialogProvider`. */
export function useAskDialog() {
  return useDialogContext().ask;
}
