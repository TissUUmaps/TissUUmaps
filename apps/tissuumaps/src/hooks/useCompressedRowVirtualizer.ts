import { type RefObject, useLayoutEffect, useRef, useState } from "react";

/**
 * Maps a scroll range that is too tall for the browser onto one it can lay out
 *
 * Browsers cap the height of an element at a few ten million pixels, and
 * silently clamp anything taller, so a list of millions of rows cannot be
 * scrolled past the cap. The list is therefore laid out at a compressed
 * height, and every scroll position is scaled between the two: the content
 * position is `compression` times the layout position, so that the end of the
 * layout still shows the end of the content.
 *
 * @param contentHeight - The height of the content, in pixels
 * @param viewportHeight - The height of the visible part of the scroll
 * container, in pixels
 * @param maxLayoutHeight - The largest height to lay out, in pixels
 * @returns The height to lay out, and the factor content positions exceed
 * layout positions by, which is `1` while the content fits
 */
function compressScrollRange(
  contentHeight: number,
  viewportHeight: number,
  maxLayoutHeight: number,
): { layoutHeight: number; compression: number } {
  const range = contentHeight - viewportHeight;
  const maxRange = maxLayoutHeight - viewportHeight;
  if (range <= maxRange || maxRange <= 0) {
    return { layoutHeight: contentHeight, compression: 1 };
  }
  return { layoutHeight: maxLayoutHeight, compression: range / maxRange };
}

/**
 * The tallest scroll content that is laid out, in pixels
 *
 * Browsers cap the height of an element at a few ten million pixels, Chrome
 * at about 33 million divided by the page zoom, and clamp anything taller, so
 * that a longer list could not be scrolled past the cap. Content beyond this
 * height is compressed instead, see {@link compressScrollRange}.
 */
const maxLayoutHeight = 10_000_000;

/**
 * The layout of a compressed virtualized list
 */
export type CompressedRowVirtualizer<THeader extends HTMLElement> = {
  /** Attached to the scroll container */
  containerRef: RefObject<HTMLDivElement | null>;
  /**
   * Attached to the header that sticks to the top of the scroll container, if
   * there is one
   */
  headerRef: RefObject<THeader | null>;
  /** The first row within the visible range */
  firstIndex: number;
  /** The row past the last one within the visible range */
  lastIndex: number;
  /** The height to lay the rows out at, in pixels */
  layoutRowsHeight: number;
  /** How far the rows have run ahead of the layout, in pixels */
  rowShift: number;
  /** Scrolls the least distance that brings a row fully into view */
  scrollRowIntoView: (index: number) => void;
};

/**
 * Virtualizes a list of rows that is taller than the browser can lay out
 *
 * Rows have a fixed height, so that the visible range follows from the scroll
 * offset alone. Past the height a browser lays out, the rows are compressed
 * into the height it does, and every row is shifted by how far the list has
 * run ahead of the layout at the current scroll offset.
 *
 * @param rowCount - The number of rows in the list
 * @param rowHeight - The height of every row, in pixels
 * @param containerHeight - The height of the scroll container, in pixels
 * @param overscan - How many rows to render beyond each end of the visible
 * range
 * @returns The refs to attach, the visible range of rows and their layout
 */
export function useCompressedRowVirtualizer<
  THeader extends HTMLElement = HTMLElement,
>(
  rowCount: number,
  rowHeight: number,
  containerHeight: number,
  overscan: number,
): CompressedRowVirtualizer<THeader> {
  const containerRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<THeader>(null);
  const [headerHeight, setHeaderHeight] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(containerHeight);
  const compressionRef = useRef(1);
  const [scrollOffset, setScrollOffset] = useState(0);
  // the offset is also kept in a ref, so that restoring it after a resize does
  // not have to re-subscribe on every scroll
  const scrollOffsetRef = useRef(0);

  useLayoutEffect(() => {
    const container = containerRef.current;
    const header = headerRef.current;
    if (container === null) {
      return;
    }
    const onScroll = () => {
      // the element scrolls in layout positions, the rows are laid out in
      // content positions, which differ once the content is compressed
      const offset = container.scrollTop * compressionRef.current;
      scrollOffsetRef.current = offset;
      setScrollOffset(offset);
    };
    // a hidden panel loses its scroll offset without raising a scroll event,
    // which would leave the rendered rows outside of the visible range
    const resizeObserver = new ResizeObserver(() => {
      setHeaderHeight(header?.offsetHeight ?? 0);
      if (container.clientHeight === 0) {
        return;
      }
      setViewportHeight(container.clientHeight);
      const layoutOffset = Math.round(
        scrollOffsetRef.current / compressionRef.current,
      );
      if (container.scrollTop !== layoutOffset) {
        container.scrollTop = layoutOffset;
      }
    });
    container.addEventListener("scroll", onScroll, { passive: true });
    resizeObserver.observe(container);
    if (header !== null) {
      resizeObserver.observe(header);
    }
    return () => {
      container.removeEventListener("scroll", onScroll);
      resizeObserver.disconnect();
    };
  }, []);

  const { layoutHeight, compression } = compressScrollRange(
    headerHeight + rowCount * rowHeight,
    viewportHeight,
    maxLayoutHeight,
  );
  // laid out before the browser can raise a scroll event against a stale factor
  useLayoutEffect(() => {
    compressionRef.current = compression;
    const container = containerRef.current;
    if (container === null) {
      return;
    }
    // the same content offset lies at a different layout offset under the new
    // factor, and no scroll event announces the change
    const layoutOffset = Math.round(scrollOffsetRef.current / compression);
    if (container.scrollTop !== layoutOffset) {
      container.scrollTop = layoutOffset;
    }
  }, [compression]);

  // the rows follow the header in the scroll content, so the offset into them
  // is the scroll offset less the header
  const rowsScrollOffset = Math.max(0, scrollOffset - headerHeight);

  function scrollRowIntoView(index: number) {
    const container = containerRef.current;
    if (container === null) {
      return;
    }
    const rowTop = headerHeight + index * rowHeight;
    const offset = scrollOffsetRef.current;
    let newOffset: number;
    if (rowTop < offset) {
      newOffset = rowTop;
    } else if (rowTop + rowHeight > offset + viewportHeight) {
      newOffset = rowTop + rowHeight - viewportHeight;
    } else {
      return;
    }
    container.scrollTop = Math.round(newOffset / compressionRef.current);
  }

  return {
    containerRef,
    headerRef,
    firstIndex: Math.max(
      0,
      Math.floor(rowsScrollOffset / rowHeight) - overscan,
    ),
    lastIndex: Math.min(
      rowCount,
      Math.ceil((rowsScrollOffset + viewportHeight) / rowHeight) + overscan,
    ),
    layoutRowsHeight: layoutHeight - headerHeight,
    rowShift: scrollOffset * (1 - 1 / compression),
    scrollRowIntoView,
  };
}
