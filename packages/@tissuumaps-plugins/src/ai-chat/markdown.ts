// The model's text is never used as HTML: the elements are created here and
// the text is put in them as plain text, so that a tag like <script> in an
// answer is shown as text and never run.

/** `code`, **bold** or *italic*; the * must touch the word, so "2 * 3" is not italic */
const inlineMarkdown = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/;

/**
 * Creates an element that contains the given text as plain text
 *
 * @param tag - The tag name
 * @param text - The text
 * @returns The element
 */
function styled(tag: string, text: string): HTMLElement {
  const element = document.createElement(tag);
  element.textContent = text;
  return element;
}

/**
 * Splits a line into plain text and `code`, **bold** and *italic* pieces;
 * underscores are left alone, so that IDs like cell_labels are not made italic
 *
 * @param line - The line
 * @returns The pieces, as text or elements
 */
function renderInline(line: string): (string | HTMLElement)[] {
  // split() with a group in the pattern keeps the matches, at odd positions
  return line.split(inlineMarkdown).map((part, index) => {
    if (index % 2 === 0) {
      return part;
    }
    if (part.startsWith("`")) {
      const code = styled("code", part.slice(1, -1));
      code.style.cssText =
        "font-family: monospace; background: var(--muted); padding: 0 4px; border-radius: 4px;";
      return code;
    }
    if (part.startsWith("**")) {
      return styled("b", part.slice(2, -2));
    }
    return styled("i", part.slice(1, -1));
  });
}

/**
 * Turns an answer of the model into elements: "- " or "* " lines become a
 * bulleted list (indented lines further to the right), "1. " lines a numbered
 * list, "#" lines a bold heading, and every other line a paragraph
 *
 * @param text - The answer
 * @returns The element holding the answer
 */
export function renderMarkdown(text: string): HTMLElement {
  const box = document.createElement("div");
  let currentList: HTMLUListElement | HTMLOListElement | null = null;
  // models sometimes write "sample\_1" for "sample_1"
  for (const line of text.replaceAll("\\_", "_").split("\n")) {
    const bullet = /^(\s*)[-*]\s+(.*)/.exec(line);
    const numbered = /^\s*(\d+)[.)]\s+(.*)/.exec(line);
    const heading = /^#{1,6}\s+(.*)/.exec(line);
    if (heading !== null) {
      currentList = null;
      const title = styled("b", heading[1] ?? "");
      title.style.cssText = "display: block; margin-top: 8px;";
      box.append(title);
    } else if (bullet !== null || numbered !== null) {
      const tag = bullet !== null ? "ul" : "ol";
      if (currentList === null || currentList.tagName.toLowerCase() !== tag) {
        currentList = document.createElement(tag);
        // TissUUmaps' styles remove the list bullets
        currentList.style.cssText = `margin: 0; padding-left: 20px; list-style: ${tag === "ul" ? "disc" : "decimal"};`;
        if (currentList instanceof HTMLOListElement) {
          currentList.start = Number(numbered?.[1] ?? 1);
        }
        box.append(currentList);
      }
      const item = document.createElement("li");
      item.append(
        ...renderInline((bullet !== null ? bullet[2] : numbered?.[2]) ?? ""),
      );
      if (bullet !== null) {
        item.style.marginLeft = `${(bullet[1] ?? "").length * 8}px`;
      }
      currentList.append(item);
    } else {
      currentList = null;
      const paragraph = document.createElement("div");
      // keeps empty lines visible
      paragraph.style.minHeight = "1em";
      paragraph.append(...renderInline(line));
      box.append(paragraph);
    }
  }
  return box;
}
