import type { Autocomplete as AutocompletePrimitive } from "@base-ui/react/autocomplete";
import { FolderIcon } from "lucide-react";
import {
  type Ref,
  useCallback,
  useEffect,
  useEffectEvent,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
  useTransition,
} from "react";

import type { TableColumnQuerySuggestion } from "@tissuumaps/core";

import {
  Autocomplete,
  AutocompleteClear,
  AutocompleteInput,
  AutocompleteInputGroup,
  AutocompleteItem,
  AutocompleteList,
  AutocompletePopup,
  AutocompleteStatus,
  AutocompleteTrigger,
} from "@/components/common/autocomplete";
import { InputGroupAddon } from "@/components/ui/input-group";
import { useCompressedRowVirtualizer } from "@/hooks/useCompressedRowVirtualizer";
import { useTableDataLoader } from "@/hooks/useDataLoader";
import { useProjectStore } from "@/stores/project";

export type TableColumnInputProps = {
  tableId: string | null;
  value: string | null;
  onValueChange: (value: string | null) => void;
  className?: string;
};

/** The height of a suggestion, in pixels */
const rowHeight = 32;

/** The largest height of the suggestion list (`max-h-80`), in pixels */
const maxListHeight = 320;

/**
 * How many suggestions are rendered beyond each end of the visible range, so
 * that the highlight moves onto a rendered one
 */
const overscan = 5;

function findQuery(suggestion: string, query: string): number {
  return suggestion.toLowerCase().indexOf(query.toLowerCase());
}

type SuggestionTextProps = {
  suggestion: string;
  query: string;
};

function SuggestionText({ suggestion, query }: SuggestionTextProps) {
  const index = query !== "" ? findQuery(suggestion, query) : -1;
  // right-to-left truncates long queries at the start; the inner span keeps
  // their characters in left-to-right order
  return (
    <span className="truncate [direction:rtl] text-left">
      {index === -1 ? (
        <span dir="ltr">{suggestion}</span>
      ) : (
        <span dir="ltr">
          {suggestion.slice(0, index)}
          <span className="font-semibold">
            {suggestion.slice(index, index + query.length)}
          </span>
          {suggestion.slice(index + query.length)}
        </span>
      )}
    </span>
  );
}

/**
 * Loads the data of a table on demand
 *
 * @param tableId - The ID of the table
 * @returns A callback yielding the table data, or `null` if the table is not
 * part of the current project
 */
function useLoadTableData(tableId: string | null) {
  const tables = useProjectStore((state) => state.tables);
  const loadTable = useTableDataLoader();
  return useCallback(
    async (options?: { signal?: AbortSignal }) => {
      const table = tables.find((table) => table.id === tableId);
      return table !== undefined ? await loadTable(table, options) : null;
    },
    [tables, tableId, loadTable],
  );
}

type SuggestionListHandle = {
  scrollRowIntoView: (index: number) => void;
};

type SuggestionListProps = {
  suggestions: TableColumnQuerySuggestion[];
  query: string;
  ref: Ref<SuggestionListHandle>;
};

/**
 * Renders only the suggestions within the visible range of the list
 *
 * Mounted together with its scroll container, which the virtualizer subscribes
 * to on mount.
 */
function SuggestionList({ suggestions, query, ref }: SuggestionListProps) {
  const {
    containerRef,
    firstIndex,
    lastIndex,
    layoutRowsHeight,
    rowShift,
    scrollRowIntoView,
  } = useCompressedRowVirtualizer(
    suggestions.length,
    rowHeight,
    Math.min(suggestions.length * rowHeight, maxListHeight),
    overscan,
  );
  useImperativeHandle(ref, () => ({ scrollRowIntoView }));

  // base-ui leaves the scrolling of a virtualized list to its owner
  useLayoutEffect(() => {
    containerRef.current?.scrollTo({ top: 0 });
  }, [containerRef, suggestions]);

  return (
    <div
      ref={containerRef}
      role="presentation"
      className="max-h-80 min-h-0 overflow-y-auto"
      style={{ height: `${suggestions.length * rowHeight}px` }}
    >
      <div
        role="presentation"
        className="relative"
        style={{ height: `${layoutRowsHeight}px` }}
      >
        {suggestions.slice(firstIndex, lastIndex).map((suggestion, i) => {
          const index = firstIndex + i;
          return (
            <AutocompleteItem
              key={suggestion.query}
              index={index}
              value={suggestion}
              aria-setsize={suggestions.length}
              aria-posinset={index + 1}
              className="absolute inset-x-0 top-0"
              style={{
                height: `${rowHeight}px`,
                transform: `translateY(${index * rowHeight - rowShift}px)`,
              }}
            >
              {suggestion.group && (
                <FolderIcon className="size-3.5 shrink-0 text-muted-foreground" />
              )}
              <SuggestionText suggestion={suggestion.query} query={query} />
            </AutocompleteItem>
          );
        })}
      </div>
    </div>
  );
}

export function TableColumnInput({
  tableId,
  value,
  onValueChange,
  className,
}: TableColumnInputProps) {
  const loadTableData = useLoadTableData(tableId);

  const query = value ?? "";

  const [text, setText] = useState(query);
  const [invalid, setInvalid] = useState(false);
  const [suggestions, setSuggestions] = useState<
    TableColumnQuerySuggestion[] | null
  >(null);
  const [open, setOpen] = useState(false);
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);

  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevQuery, setPrevQuery] = useState(query);
  const [prevTableId, setPrevTableId] = useState(tableId);
  if (query !== prevQuery) {
    setPrevQuery(query);
    setPrevTableId(tableId);
    setText(query);
    setInvalid(false);
    setPendingQuery(null);
  } else if (tableId !== prevTableId) {
    // the column is resolved again in the newly chosen table
    setPrevTableId(tableId);
    setPendingQuery(text.trim() !== "" ? text : null);
  }

  const [isSuggestPending, startSuggestTransition] = useTransition();
  useEffect(() => {
    if (!open) {
      return;
    }
    const abortController = new AbortController();
    startSuggestTransition(async () => {
      try {
        const tableData = await loadTableData({
          signal: abortController.signal,
        });
        const newSuggestions =
          (await tableData?.suggestColumnQueries(text, {
            signal: abortController.signal,
          })) ?? [];
        if (!abortController.signal.aborted) {
          startSuggestTransition(() => setSuggestions(newSuggestions));
        }
      } catch (error) {
        if (!abortController.signal.aborted) {
          console.error("Failed to suggest column queries", error);
        }
      }
    });
    return () => abortController.abort();
  }, [open, text, loadTableData, startSuggestTransition]);

  const handleCommitResolved = useEffectEvent((column: string | null) => {
    setPendingQuery(null);
    if (column !== null) {
      setText(column);
      setInvalid(false);
      onValueChange(column);
    } else {
      setInvalid(true);
    }
  });
  useEffect(() => {
    if (pendingQuery === null) {
      return;
    }
    const abortController = new AbortController();
    const { signal } = abortController;
    loadTableData({ signal })
      .then(async (tableData) => {
        const column =
          (await tableData?.resolveColumnQuery(pendingQuery, { signal })) ??
          null;
        if (!signal.aborted) {
          handleCommitResolved(column);
        }
      })
      .catch((error) => {
        if (!signal.aborted) {
          console.error("Failed to resolve column query", error);
          setPendingQuery(null);
        }
      });
    return () => abortController.abort();
  }, [pendingQuery, loadTableData]);

  function commit(newQuery: string) {
    if (newQuery === query && !invalid) {
      setPendingQuery(null);
      setInvalid(false);
      return;
    }
    if (newQuery.trim() === "") {
      setPendingQuery(null);
      setText("");
      setInvalid(false);
      onValueChange(null);
      return;
    }
    setPendingQuery(newQuery);
  }

  const highlightedSuggestionRef = useRef<
    TableColumnQuerySuggestion | undefined
  >(undefined);
  const suggestionListRef = useRef<SuggestionListHandle>(null);
  // base-ui closes the popup after any item press and only resets the
  // highlighted index on unmount, so cancelling the close would leave a stale
  // highlight on the children; a pressed group suggestion reopens the
  // popup instead
  const reopenRef = useRef(false);

  function handleTextChange(
    newText: string,
    details: AutocompletePrimitive.Root.ChangeEventDetails,
  ) {
    // base-ui clears the input on Escape when the popup is closed
    if (details.reason === "escape-key") {
      details.cancel();
      return;
    }
    setPendingQuery(null);
    setText(newText);
    setInvalid(false);
    if (details.reason === "item-press") {
      const pressed = suggestions?.find((s) => s.query === newText);
      if (pressed?.group) {
        reopenRef.current = true;
      } else {
        commit(newText);
      }
      return;
    }
    if (details.reason === "clear-press") {
      onValueChange(null);
    }
  }

  function handleOpenChange(newOpen: boolean) {
    setOpen(newOpen);
    if (!newOpen) {
      setSuggestions(null);
      highlightedSuggestionRef.current = undefined;
    }
  }

  function handleOpenChangeComplete(newOpen: boolean) {
    if (!newOpen && reopenRef.current) {
      reopenRef.current = false;
      setOpen(true);
    }
  }

  function getStatusMessage(): string | null {
    if (suggestions === null) {
      return isSuggestPending ? "Loading table..." : null;
    }
    if (suggestions.length === 0) {
      return text === "" ? "No columns" : `No matches for "${text}"`;
    }
    return null;
  }

  return (
    <Autocomplete
      value={text}
      onValueChange={handleTextChange}
      mode="none"
      items={suggestions ?? []}
      virtualized
      itemToStringValue={(suggestion) => suggestion.query}
      openOnInputClick
      open={open}
      onOpenChange={handleOpenChange}
      onOpenChangeComplete={handleOpenChangeComplete}
      onItemHighlighted={(suggestion, { reason, index }) => {
        highlightedSuggestionRef.current = suggestion;
        // base-ui cannot scroll to a suggestion that is not rendered, e.g.
        // when the highlight wraps around the list
        if (suggestion !== undefined && reason === "keyboard") {
          suggestionListRef.current?.scrollRowIntoView(index);
        }
      }}
    >
      <AutocompleteInputGroup className={className}>
        <AutocompleteInput
          aria-invalid={invalid || undefined}
          aria-busy={pendingQuery !== null || undefined}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              highlightedSuggestionRef.current === undefined
            ) {
              commit(text);
            }
          }}
          onBlur={() => commit(text)}
        />
        <InputGroupAddon align="inline-end">
          <AutocompleteClear />
          <AutocompleteTrigger aria-label="Show columns" title="Show columns" />
        </InputGroupAddon>
      </AutocompleteInputGroup>
      <AutocompletePopup className="flex flex-col overflow-hidden">
        <AutocompleteStatus>{getStatusMessage()}</AutocompleteStatus>
        <AutocompleteList className="flex min-h-0 flex-col">
          {suggestions !== null && suggestions.length > 0 && (
            <SuggestionList
              ref={suggestionListRef}
              suggestions={suggestions}
              query={text}
            />
          )}
        </AutocompleteList>
      </AutocompletePopup>
    </Autocomplete>
  );
}
