import { createContext, useContext } from "react";

import type { DataSource } from "@tissuumaps/core";

import type { AddDataObjectDialogParams } from "./AddDataObjectDialog";
import type { AlertDialogParams } from "./AlertDialog";
import type { ConfirmDialogParams } from "./ConfirmDialog";
import type { PromptDialogParams } from "./PromptDialog";

export type DialogContextValue = {
  /** Opens an acknowledgement dialog. Resolves once dismissed. */
  alert: (params: AlertDialogParams) => Promise<void>;
  /** Opens a yes/no dialog. Resolves to `true` when confirmed, `false` otherwise. */
  confirm: (params: ConfirmDialogParams) => Promise<boolean>;
  /** Opens a text-input dialog. Resolves to the entered string, or `null`. */
  prompt: (params: PromptDialogParams) => Promise<string | null>;
  /**
   * Queues add data object dialogs: one per given source, pre-filled with it,
   * so none for an empty array, or a single empty one if `sources` is omitted.
   */
  addDataObject: <TDataSource extends DataSource>(
    params: AddDataObjectDialogParams<TDataSource>,
    sources?: string[],
  ) => void;
};

export const DialogContext = createContext<DialogContextValue | null>(null);

export function useDialogContext() {
  const context = useContext(DialogContext);
  if (context === null) {
    throw new Error("useDialogContext must be used within DialogProvider");
  }
  return context;
}
