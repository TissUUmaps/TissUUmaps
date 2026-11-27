import { BracesIcon, CircleAlertIcon } from "lucide-react";
import { useRef, useState } from "react";
import { flushSync } from "react-dom";

import { IconButton } from "@/components/common/icon-button";
import { TableColumnInput } from "@/components/controls/TableColumnField/TableColumnInput";
import { cn } from "@/lib/utils";

export type ColumnNameBoxProps = {
  /** The ID of the object's own table */
  tableId: string;

  /** The followed column, if any */
  column: string | null;

  /** Whether the followed column is not in the table, if there is one */
  isMissing: boolean;

  /** Picks the column to follow, or none; resolves to whether it applied */
  onColumnChange: (column: string | null) => Promise<boolean>;

  className?: string;
};

/**
 * The query of the followed column, which turns into a query input for any
 * column of the table when clicked, like the name box of a spreadsheet
 *
 * Esc, leaving the input or a pick turns it back. An emptied text clears the
 * column, as in the rows; any other text typed but not picked is dropped.
 */
export function ColumnNameBox({
  tableId,
  column,
  isMissing,
  onColumnChange,
  className,
}: ColumnNameBoxProps) {
  const [isEditing, setEditing] = useState(false);
  // counts the times the input opened and closed, so that a pick that applies
  // after the input was closed and opened again leaves the new one open
  const editingSessionRef = useRef(0);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const stopEditing = () => {
    editingSessionRef.current += 1;
    const hadFocus =
      wrapperRef.current?.contains(document.activeElement) ?? false;
    // the button has to be in the DOM before it can take the focus back
    // https://react.dev/reference/react-dom/flushSync
    flushSync(() => setEditing(false));
    if (hadFocus) {
      buttonRef.current?.focus();
    }
  };

  if (isEditing) {
    return (
      <div
        ref={wrapperRef}
        className={cn("min-w-0 flex-1", className)}
        // the suggestions close on the same Esc
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            stopEditing();
          }
        }}
        onBlur={(event) => {
          // the input stays open while another window has the focus
          if (
            document.hasFocus() &&
            !event.currentTarget.contains(event.relatedTarget)
          ) {
            stopEditing();
          }
        }}
      >
        <TableColumnInput
          tableId={tableId}
          value={column}
          onValueChange={async (newColumn) => {
            const editingSession = editingSessionRef.current;
            const applied = await onColumnChange(newColumn);
            if (applied && editingSessionRef.current === editingSession) {
              stopEditing();
            }
            return applied;
          }}
          placeholder="Column query"
          autoFocus
          className="h-7 font-mono"
        />
      </div>
    );
  }

  return (
    <IconButton
      ref={buttonRef}
      label={
        column === null
          ? "Pick any column by its query"
          : isMissing
            ? `Column not in the table: ${column}`
            : `Column query: ${column}`
      }
      size="xs"
      onClick={() => {
        editingSessionRef.current += 1;
        setEditing(true);
      }}
      className={cn(
        "min-w-0 max-w-full border-transparent font-mono font-normal hover:border-input",
        className,
      )}
    >
      {isMissing ? (
        <CircleAlertIcon className="text-destructive" />
      ) : (
        <BracesIcon />
      )}
      {/* right-to-left truncates long queries at the start; the inner span
      keeps their characters in left-to-right order */}
      <span
        className={cn(
          "truncate [direction:rtl] text-left",
          column !== null && "text-foreground",
          isMissing && "line-through",
        )}
      >
        <span dir="ltr">{column ?? "Any column"}</span>
      </span>
    </IconButton>
  );
}
