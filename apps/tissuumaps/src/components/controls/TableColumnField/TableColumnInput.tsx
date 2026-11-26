import type { Autocomplete as AutocompletePrimitive } from "@base-ui/react/autocomplete";
import { FolderIcon } from "lucide-react";
import {
  type Ref,
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
import { useLazyTableData } from "@/data/hooks/useLazyData";
import { useCompressedRowVirtualizer } from "@/hooks/useCompressedRowVirtualizer";

import { findColumnByName } from "./findColumnByName";

export type TableColumnInputProps = {
  tableId: string | null;
  value: string | null;
  /**
   * Applies a column, or none; may resolve to whether it applied, and a refused
   * column shows the value again
   */
  onValueChange: (value: string | null) => void | Promise<boolean>;
  /**
   * The `id` of the leaf section to pick a column of (see
   * `TableData.getColumnSections`); the input then shows and searches the leaf's
   * columns by name, and a typed name has to match a column's name (see
   * {@link findColumnByName})
   */
  sectionId?: string;
  /** The name to show for the value, if it is not the query itself */
  valueName?: string;
  /**
   * Called with a text that matches the name of no column of the section when
   * it is committed (e.g. to search other sections for it)
   */
  onNoMatch?: (text: string) => void;
  placeholder?: string;
  className?: string;
};

/**
 * @param suggestion - A suggestion
 * @returns The text to show for the suggestion
 */
function getSuggestionText(suggestion: TableColumnQuerySuggestion): string {
  return suggestion.name ?? suggestion.query;
}

/**
 * @param suggestion - A suggestion
 * @param text - A searched text
 * @returns Whether the suggestion's text contains the text, ignoring case
 */
function matchesText(
  suggestion: TableColumnQuerySuggestion,
  text: string,
): boolean {
  return getSuggestionText(suggestion)
    .toLowerCase()
    .includes(text.toLowerCase());
}

/** Milliseconds of no typing after which a text matching nothing is reported */
const noMatchDelay = 600;

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
              <SuggestionText
                suggestion={getSuggestionText(suggestion)}
                query={query}
              />
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
  sectionId,
  valueName,
  onNoMatch,
  placeholder,
  className,
}: TableColumnInputProps) {
  const loadTableData = useLazyTableData(tableId);

  // within a section, the input shows and searches names, not queries
  const valueText = (value !== null ? valueName : undefined) ?? value ?? "";

  const [text, setText] = useState(valueText);
  const [invalid, setInvalid] = useState(false);
  const [suggestions, setSuggestions] = useState<
    TableColumnQuerySuggestion[] | null
  >(null);
  // the text the suggestions were made for, which lags behind while typing
  const [suggestionsText, setSuggestionsText] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);

  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevValueText, setPrevValueText] = useState(valueText);
  const [prevTableId, setPrevTableId] = useState(tableId);
  if (valueText !== prevValueText) {
    setPrevValueText(valueText);
    setPrevTableId(tableId);
    setText(valueText);
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
            sectionId,
            signal: abortController.signal,
          })) ?? [];
        if (!abortController.signal.aborted) {
          startSuggestTransition(() => {
            setSuggestions(newSuggestions);
            setSuggestionsText(text);
          });
        }
      } catch (error) {
        if (!abortController.signal.aborted) {
          console.error("Failed to suggest column queries", error);
        }
      }
    });
    return () => abortController.abort();
  }, [open, text, sectionId, loadTableData, startSuggestTransition]);

  // every suggestion is listed, matches first, as the table ranks them; within
  // a section, a text that no suggestion matches is reported (see below)
  const hasNoMatch =
    sectionId !== undefined &&
    text.trim() !== "" &&
    text !== valueText &&
    suggestionsText === text &&
    suggestions !== null &&
    !suggestions.some((suggestion) => matchesText(suggestion, text));

  // a text matching nothing in the section is reported once typing pauses,
  // and never before its suggestions have loaded; a committed one right away
  const reportNoMatch = useEffectEvent((noMatchText: string) => {
    onNoMatch?.(noMatchText);
  });
  useEffect(() => {
    if (!hasNoMatch) {
      return;
    }
    const timeout = setTimeout(() => reportNoMatch(text), noMatchDelay);
    return () => clearTimeout(timeout);
  }, [hasNoMatch, text]);

  const handleCommitResolved = useEffectEvent((column: string | null) => {
    setPendingQuery(null);
    if (column !== null) {
      // within a section, the input keeps showing the picked name
      const committedText = sectionId === undefined ? column : text;
      setText(committedText);
      setInvalid(false);
      void Promise.resolve(onValueChange(column)).then((applied) => {
        // unless typed over since
        if (applied === false) {
          setText((currentText) =>
            currentText === committedText ? valueText : currentText,
          );
        }
      });
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
        // within a section, the text is a name, not a query
        let column: string | null = null;
        if (tableData !== null && sectionId !== undefined) {
          const found = await findColumnByName(
            tableData,
            sectionId,
            pendingQuery,
            { signal },
          );
          column = found?.column ?? null;
        } else if (tableData !== null) {
          column = await tableData.resolveColumnQuery(pendingQuery, { signal });
        }
        if (!signal.aborted) {
          if (sectionId !== undefined && column === null) {
            reportNoMatch(pendingQuery);
          }
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
  }, [pendingQuery, sectionId, loadTableData]);

  function commit(newQuery: string) {
    if (newQuery === valueText && !invalid) {
      setPendingQuery(null);
      setInvalid(false);
      return;
    }
    if (newQuery.trim() === "") {
      setPendingQuery(null);
      setText("");
      setInvalid(false);
      void onValueChange(null);
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
      const pressed = suggestions?.find(
        (s) => getSuggestionText(s) === newText,
      );
      if (pressed?.group) {
        reopenRef.current = true;
      } else {
        commit(newText);
      }
      return;
    }
    if (details.reason === "clear-press") {
      void onValueChange(null);
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
    return hasNoMatch ? `No matches for "${text}"` : null;
  }

  return (
    <Autocomplete
      value={text}
      onValueChange={handleTextChange}
      mode="none"
      items={suggestions ?? []}
      virtualized
      itemToStringValue={getSuggestionText}
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
          placeholder={placeholder}
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
          <AutocompleteTrigger label="Show columns" />
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
