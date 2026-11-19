// The model's text is never used as HTML: the elements are created here and
// the text is put in them as plain text, so that a tag like <script> in an
// answer is shown as text and never run.

/**
 * `code`, a [link](https://…), **bold** or *italic*; the * must touch the
 * word, so "2 * 3" is not italic. Only links to http and https addresses.
 */
const inlineMarkdown =
  /(`[^`]+`|\[[^\]\n]+\]\(https?:\/\/[^\s()]+(?:\([^\s()]*\)[^\s()]*)*\)|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/;

/** The parts of a link: [text](address) */
const linkParts = /^\[([^\]\n]+)\]\((\S+)\)$/;

/**
 * The line under the header of a table, e.g. "|---|:--:|---:|"; it must hold
 * a "|", or it would be a horizontal line
 */
const tableSeparator = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

/** A cell that holds only a number, e.g. "12", "-0.5", "1,234", "3.2e-5" or "45%" */
const numberCell = /^[-+−]?(\d[\d,]*(\.\d+)?|\.\d+)([eE][-+−]?\d+)?\s*%?$/;

/** How the columns of a table are aligned */
type Alignment = "left" | "center" | "right" | undefined;

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
 * Creates a link that opens in a new tab and tells the site nothing about
 * the page it comes from
 *
 * @param text - The text of the link
 * @param address - Where it goes; only http and https addresses
 * @returns The link
 */
export function makeLink(text: string, address: string): HTMLAnchorElement {
  const link = document.createElement("a");
  link.textContent = text;
  link.href = address;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.referrerPolicy = "no-referrer";
  link.title = address;
  link.style.cssText = "color: var(--primary); text-decoration: underline;";
  return link;
}

/**
 * Splits a line into plain text and `code`, links, **bold** and *italic*
 * pieces; underscores are left alone, so that IDs like cell_labels are not
 * made italic
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
    if (part.startsWith("[")) {
      const [, text = "", address = ""] = linkParts.exec(part) ?? [];
      return makeLink(text, address);
    }
    if (part.startsWith("**")) {
      return styled("b", part.slice(2, -2));
    }
    return styled("i", part.slice(1, -1));
  });
}

/**
 * Splits a row of a table into its cells: the "|" at the start and at the
 * end are optional, "\|" is a "|" inside a cell, and so is a "|" in `code`
 *
 * @param line - The row
 * @returns The text of each cell
 */
function tableCells(line: string): string[] {
  let text = line.trim();
  if (text.startsWith("|")) {
    text = text.slice(1);
  }
  if (text.endsWith("|") && !text.endsWith("\\|")) {
    text = text.slice(0, -1);
  }
  const cells: string[] = [];
  let cell = "";
  let inCode = false;
  for (let index = 0; index < text.length; index++) {
    const character = text[index];
    if (character === "\\" && text[index + 1] === "|") {
      cell += "|";
      index++;
      continue;
    }
    if (character === "`") {
      inCode = !inCode;
    }
    if (character === "|" && !inCode) {
      cells.push(cell.trim());
      cell = "";
      continue;
    }
    cell += character;
  }
  cells.push(cell.trim());
  return cells;
}

/**
 * Whether two lines start a table: a header row, then the line of dashes
 * with as many columns
 *
 * @param header - The first line
 * @param separator - The line after it
 * @returns Whether they do
 */
function startsTable(header: string, separator: string | undefined): boolean {
  return (
    separator !== undefined &&
    header.includes("|") &&
    separator.includes("|") &&
    tableSeparator.test(separator) &&
    tableCells(header).length === tableCells(separator).length
  );
}

/**
 * Makes a table: the header, then the rows, each with as many cells as the
 * header; columns aligned as the line of dashes says, and columns of
 * numbers to the right, so that their digits line up
 *
 * @param header - The header row
 * @param separator - The line of dashes, e.g. "|---|:--:|"
 * @param rows - The other rows
 * @returns The element holding the table, which scrolls sideways if it is
 * wider than the chat
 */
function renderTable(
  header: string,
  separator: string,
  rows: string[],
): HTMLElement {
  const names = tableCells(header);
  const body = rows.map((row) => {
    const cells = tableCells(row).slice(0, names.length);
    while (cells.length < names.length) {
      cells.push("");
    }
    return cells;
  });
  const alignments: Alignment[] = tableCells(separator).map(
    (dashes, column) => {
      const left = dashes.startsWith(":");
      const right = dashes.endsWith(":");
      if (left && right) {
        return "center";
      }
      if (right) {
        return "right";
      }
      if (left) {
        return "left";
      }
      const values = body.map((cells) => cells[column] ?? "").filter(Boolean);
      return values.length > 0 &&
        values.every((value) => numberCell.test(value.replaceAll("*", "")))
        ? "right"
        : undefined;
    },
  );
  const cellStyle =
    "padding: 3px 8px; border-bottom: 1px solid var(--border); vertical-align: top;";
  const makeCell = (tag: "th" | "td", text: string, column: number) => {
    const cell = document.createElement(tag);
    cell.style.cssText = `${cellStyle}text-align: ${alignments[column] ?? "left"};`;
    cell.append(...renderInline(text));
    return cell;
  };
  const table = document.createElement("table");
  table.style.cssText =
    "border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums;" +
    "border: 1px solid var(--border);";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.style.cssText = "background: var(--muted); color: var(--foreground);";
  headRow.append(...names.map((name, column) => makeCell("th", name, column)));
  for (const cell of headRow.children) {
    (cell as HTMLElement).style.fontWeight = "600";
  }
  head.append(headRow);
  const rowsElement = document.createElement("tbody");
  rowsElement.append(
    ...body.map((cells) => {
      const row = document.createElement("tr");
      row.append(...cells.map((text, column) => makeCell("td", text, column)));
      return row;
    }),
  );
  table.append(head, rowsElement);
  const box = document.createElement("div");
  box.style.cssText =
    "overflow-x: auto; scrollbar-width: thin; margin: 4px 0; white-space: normal;";
  box.append(table);
  return box;
}

/**
 * Turns an answer of the model into elements: "- " or "* " lines become a
 * bulleted list (indented lines further to the right), "1. " lines a numbered
 * list, "#" lines a bold heading, rows with "|" under a header and a line of
 * dashes a table, and every other line a paragraph
 *
 * @param text - The answer
 * @returns The element holding the answer
 */
export function renderMarkdown(text: string): HTMLElement {
  const box = document.createElement("div");
  let currentList: HTMLUListElement | HTMLOListElement | null = null;
  // models sometimes write "sample\_1" for "sample_1"
  const lines = text.replaceAll("\\_", "_").split("\n");
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index] ?? "";
    const separator = lines[index + 1];
    if (startsTable(line, separator)) {
      currentList = null;
      const rows: string[] = [];
      index += 2;
      while (index < lines.length && (lines[index] ?? "").includes("|")) {
        rows.push(lines[index] ?? "");
        index++;
      }
      // the line after the table is read again by the loop
      index--;
      box.append(renderTable(line, separator ?? "", rows));
      continue;
    }
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
