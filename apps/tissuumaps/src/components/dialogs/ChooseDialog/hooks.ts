import { useDialogContext } from "../DialogContext";

/** Returns the `choose` function of the nearest `DialogProvider`. */
export function useChooseDialog() {
  return useDialogContext().choose;
}
