import {
  type Dispatch,
  Fragment,
  type ReactElement,
  type ReactNode,
  type SetStateAction,
  useState,
} from "react";

import type { DataSource } from "@tissuumaps/core";

import {
  AddDataObjectDialog,
  type AddDataObjectDialogParams,
} from "./AddDataObjectDialog";
import { AlertDialog, type AlertDialogParams } from "./AlertDialog";
import { ConfirmDialog, type ConfirmDialogParams } from "./ConfirmDialog";
import { DialogContext, type DialogContextValue } from "./DialogContext";
import { PromptDialog, type PromptDialogParams } from "./PromptDialog";

type DialogProviderProps = {
  children: ReactNode;
};

type ActiveDialog = {
  id: number;
  render: (open: boolean) => ReactElement;
};

type QueuedDialog = {
  id: number;
  render: (
    open: boolean,
    onClose: () => void,
    onClosed: () => void,
  ) => ReactElement;
};

/**
 * Builds the imperative dialog API on top of the provider's state setters.
 * Called once per provider: the setters are stable, so the value never needs
 * to change.
 */
function createDialogContextValue(
  setDialog: Dispatch<SetStateAction<ActiveDialog | null>>,
  setOpen: Dispatch<SetStateAction<boolean>>,
  setQueue: Dispatch<SetStateAction<QueuedDialog[]>>,
): DialogContextValue {
  // Settles the dialog still pending (if any) with its dismissal value.
  let dismiss: (() => void) | null = null;

  function show<T>(
    dismissValue: T,
    render: (settle: (value: T) => void) => (open: boolean) => ReactElement,
  ) {
    dismiss?.();
    return new Promise<T>((resolve) => {
      const settle = (value: T) => {
        dismiss = null;
        setOpen(false);
        resolve(value);
      };
      dismiss = () => settle(dismissValue);
      // A new id on every open so uncontrolled content (e.g. the prompt
      // input) remounts rather than retaining the previous dialog's value.
      setDialog((previous) => ({
        id: (previous?.id ?? 0) + 1,
        render: render(settle),
      }));
      setOpen(true);
    });
  }

  // Sequential dialogs are queued, and shown one after another.
  let nextQueuedId = 0;
  function enqueue(...renders: QueuedDialog["render"][]) {
    const dialogs = renders.map((render) => ({ id: nextQueuedId++, render }));
    setQueue((queue) => [...queue, ...dialogs]);
  }

  return {
    alert: (params: AlertDialogParams) =>
      show<void>(undefined, (settle) => (open) => (
        <AlertDialog {...params} open={open} onDismiss={() => settle()} />
      )),
    confirm: (params: ConfirmDialogParams) =>
      show(false, (settle) => (open) => (
        <ConfirmDialog
          {...params}
          open={open}
          onCancel={() => settle(false)}
          onConfirm={() => settle(true)}
        />
      )),
    prompt: (params: PromptDialogParams) =>
      show<string | null>(null, (settle) => (open) => (
        <PromptDialog
          {...params}
          open={open}
          onCancel={() => settle(null)}
          onConfirm={settle}
        />
      )),
    addDataObject: <TDataSource extends DataSource>(
      params: AddDataObjectDialogParams<TDataSource>,
      sources?: string[],
    ) =>
      enqueue(
        ...(sources ?? [undefined]).map(
          (initialSource) =>
            (open: boolean, onClose: () => void, onClosed: () => void) => (
              <AddDataObjectDialog
                {...params}
                open={open}
                initialSource={initialSource}
                onClose={onClose}
                onClosed={onClosed}
              />
            ),
        ),
      ),
  };
}

/**
 * Orchestrates the imperative dialog API: it owns the open state and the
 * pending promise, and renders the currently active dialog. Because dialogs
 * open imperatively, `AlertDialogTrigger` is never used.
 *
 * Sequential dialogs (add data object dialogs) are queued instead, and shown
 * one after another, beneath the active dialog: an alert they open shows on
 * top of them, without closing them.
 */
export function DialogProvider({ children }: DialogProviderProps) {
  const [dialog, setDialog] = useState<ActiveDialog | null>(null);
  const [open, setOpen] = useState(false);
  const [queue, setQueue] = useState<QueuedDialog[]>([]);
  const [isQueueHeadClosing, setQueueHeadClosing] = useState(false);
  const [value] = useState(() =>
    createDialogContextValue(setDialog, setOpen, setQueue),
  );

  const queueHead = queue[0];
  return (
    <DialogContext.Provider value={value}>
      {children}
      {/* The next queued dialog is shown once the head finished closing. */}
      {queueHead && (
        <Fragment key={queueHead.id}>
          {queueHead.render(
            !isQueueHeadClosing,
            () => setQueueHeadClosing(true),
            () => {
              setQueue((queue) => queue.slice(1));
              setQueueHeadClosing(false);
            },
          )}
        </Fragment>
      )}
      {/* The dialog stays mounted with `open=false` so the close animation plays. */}
      {dialog && <Fragment key={dialog.id}>{dialog.render(open)}</Fragment>}
    </DialogContext.Provider>
  );
}
