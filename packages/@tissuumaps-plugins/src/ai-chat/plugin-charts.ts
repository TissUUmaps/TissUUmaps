import { categoricalColorPalettes } from "@tissuumaps/core";

// Ready-made charts for the plugins made by the chat: chat.heatmap,
// chat.bars, chat.scatter and chat.legend. A model rarely draws a good
// chart from nothing (numbers missing on the axes, colors that ignore the
// theme, scales not centered on 0), so the plugins get these pieces, made
// and tested once: numbers on the axes, in number order for groups named by
// numbers; a color bar with its numbers, centered on 0 for values with a
// sign; the colors of the light and dark theme, drawn again when it
// changes; the values under the mouse; a function for a click; and buttons
// that save the chart as a PNG and its values as a CSV. They are SVG, built
// with textContent only: names and values from the data never become HTML.

/** The colors of the theme that plugins get, by their CSS variables */
const themeVariables = {
  background: "--background",
  foreground: "--foreground",
  muted: "--muted",
  mutedForeground: "--muted-foreground",
  primary: "--primary",
  primaryForeground: "--primary-foreground",
  accent: "--accent",
  border: "--border",
  card: "--card",
  destructive: "--destructive",
} as const;

/** The colors of the theme, resolved, and whether it is dark */
export type ThemeColors = Record<keyof typeof themeVariables, string> & {
  dark: boolean;
};

/**
 * The colors of the theme of TissUUmaps as they are now, as colors that a
 * canvas understands (it does not understand var(--…)), and whether the
 * theme is dark
 *
 * @returns The colors, by name, and dark
 */
export function themeColors(): ThemeColors {
  const root = document.documentElement;
  const style = getComputedStyle(root);
  const colors = Object.fromEntries(
    Object.entries(themeVariables).map(([name, variable]) => [
      name,
      style.getPropertyValue(variable).trim(),
    ]),
  ) as Record<keyof typeof themeVariables, string>;
  return { ...colors, dark: root.classList.contains("dark") };
}

/**
 * Calls a function each time the theme of TissUUmaps has changed, once the
 * page shows the new colors
 *
 * @param listener - The function, with the new colors
 * @returns A function that stops calling it
 */
export function watchTheme(
  listener: (colors: ThemeColors) => void,
): () => void {
  let dark = document.documentElement.classList.contains("dark");
  const observer = new MutationObserver(() => {
    const now = document.documentElement.classList.contains("dark");
    if (now !== dark) {
      dark = now;
      listener(themeColors());
    }
  });
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
  return () => observer.disconnect();
}

// ---------------------------------------------------------------- colors

/** A color as red, green and blue from 0 to 255 */
type Rgb = [number, number, number];

/**
 * Reads "#rrggbb"
 *
 * @param hex - The color
 * @returns Its red, green and blue
 */
function fromHex(hex: string): Rgb {
  const number = parseInt(hex.slice(1), 16);
  return [(number >> 16) & 255, (number >> 8) & 255, number & 255];
}

/**
 * Writes "#rrggbb"
 *
 * @param rgb - Red, green and blue, from 0 to 255
 * @returns The color
 */
function toHex(rgb: Rgb): string {
  return `#${rgb
    .map((part) =>
      Math.round(Math.min(255, Math.max(0, part)))
        .toString(16)
        .padStart(2, "0"),
    )
    .join("")}`;
}

/**
 * Turns a color into OKLab, where mixing colors looks even to the eye
 *
 * @param rgb - The color
 * @returns Its lightness and its a and b
 */
function toOklab(rgb: Rgb): Rgb {
  const [r, g, b] = rgb.map((part) => {
    const c = part / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }) as Rgb;
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

/**
 * Turns an OKLab color back into red, green and blue
 *
 * @param lab - The color in OKLab
 * @returns Its red, green and blue
 */
function fromOklab([L, a, b]: Rgb): Rgb {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ].map((c) => {
    const clamped = Math.min(1, Math.max(0, c));
    return (
      255 *
      (clamped <= 0.0031308
        ? 12.92 * clamped
        : 1.055 * clamped ** (1 / 2.4) - 0.055)
    );
  }) as Rgb;
}

/**
 * A color between stops, mixed in OKLab
 *
 * @param stops - The colors, as "#rrggbb", evenly spread from 0 to 1
 * @param t - Where, from 0 to 1
 * @returns The color, as "#rrggbb"
 */
function between(stops: string[], t: number): string {
  const share = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  const place = share * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(place));
  const from = toOklab(fromHex(stops[index]!));
  const to = toOklab(fromHex(stops[index + 1]!));
  const part = place - index;
  if (part === 0 || part === 1) {
    return stops[index + part]!;
  }
  return toHex(
    fromOklab(from.map((value, i) => value + (to[i]! - value) * part) as Rgb),
  );
}

/**
 * The scales of the charts, light and dark: blue for minus, a gray that is
 * nothing at 0, red for plus; and one blue from little to much
 */
const scales = {
  light: {
    minus: ["#f0efec", "#86b6ef", "#256abf", "#0d366b"],
    plus: ["#f0efec", "#f0a29a", "#cc3d3a", "#7d1517"],
    sequential: ["#eef3f9", "#86b6ef", "#256abf", "#0d366b"],
  },
  dark: {
    minus: ["#383835", "#1f4f8c", "#3987e5", "#b7d3f6"],
    plus: ["#383835", "#7f2a28", "#e05b58", "#f6c1bd"],
    sequential: ["#262d36", "#184f95", "#3987e5", "#cde2fb"],
  },
};

/** The colors of groups and series, in a fixed order (light and dark) */
const seriesColors = {
  light: [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
  ],
  dark: [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#0ca30c",
    "#9085e9",
    "#e66767",
  ],
};

/**
 * The color of a value with a sign: blue below 0, gray at 0, red above
 *
 * @param value - The value
 * @param limit - The largest absolute value, the end of the scale
 * @param dark - Whether the theme is dark
 * @returns The color
 */
export function signedColor(
  value: number,
  limit: number,
  dark: boolean,
): string {
  const scale = scales[dark ? "dark" : "light"];
  return between(
    value < 0 ? scale.minus : scale.plus,
    Math.abs(value) / (limit || 1),
  );
}

/**
 * The color of a value without a sign, from light (little) to dark (much),
 * the other way round in the dark theme
 *
 * @param value - The value
 * @param low - The lowest value
 * @param high - The highest value
 * @param dark - Whether the theme is dark
 * @returns The color
 */
export function sequentialColor(
  value: number,
  low: number,
  high: number,
  dark: boolean,
): string {
  return between(
    scales[dark ? "dark" : "light"].sequential,
    high > low ? (value - low) / (high - low) : 0.5,
  );
}

/**
 * The color of a group or series: the eight colors of the charts, then the
 * distinct colors of TissUUmaps
 *
 * @param index - The number of the group, from 0
 * @param count - How many groups there are
 * @param dark - Whether the theme is dark
 * @returns The color
 */
export function groupColor(
  index: number,
  count: number,
  dark: boolean,
): string {
  const few = seriesColors[dark ? "dark" : "light"];
  if (count <= few.length) {
    return few[index % few.length]!;
  }
  const many = categoricalColorPalettes[0]!.colors;
  const color = many[index % many.length]!;
  return toHex([color.r, color.g, color.b]);
}

// ---------------------------------------------------------------- numbers

/**
 * The name of a group: whole numbers without ".0" ("3", not "3.0")
 *
 * @param value - The group, as read from a table
 * @returns Its name
 */
export function groupName(value: unknown): string {
  if (
    typeof value === "number" ||
    typeof value === "bigint" ||
    typeof value === "boolean"
  ) {
    return String(value);
  }
  const text =
    typeof value === "string"
      ? value
      : value === undefined || value === null
        ? ""
        : (JSON.stringify(value) ?? "");
  return /^-?\d+\.0+$/.test(text.trim())
    ? text.trim().replace(/\.0+$/, "")
    : text;
}

/**
 * The order in which to show groups: by number when every name is a number,
 * else as given
 *
 * @param names - The names
 * @returns The indices of the names, in that order
 */
export function numberOrder(names: string[]): number[] {
  const indices = names.map((_, index) => index);
  const numbers = names.map((name) =>
    name.trim() === "" ? Number.NaN : Number(name),
  );
  if (names.length === 0 || !numbers.every(Number.isFinite)) {
    return indices;
  }
  return indices.sort((a, b) => numbers[a]! - numbers[b]!);
}

/**
 * Writes a number for a label: few digits, but enough to tell it
 *
 * @param value - The number
 * @returns The text
 */
export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  const size = Math.abs(value);
  const digits =
    size === 0 || size >= 100 ? 0 : size >= 10 ? 1 : size >= 1 ? 2 : 3;
  const text = value.toFixed(digits);
  const trimmed = text.includes(".") ? text.replace(/\.?0+$/, "") : text;
  return trimmed === "-0" ? "0" : trimmed.replace(/^-/, "−");
}

/**
 * Round numbers for the ticks of an axis, e.g. 0, 20, 40, 60
 *
 * @param low - The lowest value
 * @param high - The highest value
 * @param count - About how many ticks
 * @returns The ticks, from low to high, covering both
 */
export function niceTicks(low: number, high: number, count = 5): number[] {
  if (!(high > low)) {
    const value = Number.isFinite(low) ? low : 0;
    return value === 0 ? [0, 1] : [Math.min(0, value), Math.max(0, value)];
  }
  const rough = (high - low) / count;
  const power = 10 ** Math.floor(Math.log10(rough));
  const step =
    [1, 2, 2.5, 5, 10]
      .map((factor) => factor * power)
      .find((s) => s >= rough) ?? 10 * power;
  const first = Math.floor(low / step) * step;
  const ticks: number[] = [];
  for (let tick = first; tick < high + step * 0.999; tick += step) {
    ticks.push(Number(tick.toPrecision(12)));
  }
  return ticks;
}

// ---------------------------------------------------------------- checks

/**
 * Throws a clear error for wrong options, for the model to read
 *
 * @param chart - Which piece, e.g. "chat.heatmap"
 * @param message - What is wrong
 */
function wrong(chart: string, message: string): never {
  throw new Error(`${chart}: ${message}`);
}

/**
 * Reads a list of names
 *
 * @param chart - Which piece
 * @param name - The option
 * @param value - Its value
 * @returns The names
 */
function namesOf(chart: string, name: string, value: unknown): string[] {
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
    wrong(chart, `${name} must be a list of names, e.g. ["0", "1", "2"]`);
  }
  return Array.from(value as ArrayLike<unknown>, groupName);
}

/**
 * Reads a list of numbers; null, NaN or something else is a missing value
 *
 * @param chart - Which piece
 * @param name - The option
 * @param value - Its value
 * @returns The numbers, NaN where missing
 */
function numbersOf(chart: string, name: string, value: unknown): number[] {
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
    wrong(chart, `${name} must be a list of numbers`);
  }
  return Array.from(value as ArrayLike<unknown>, (item) =>
    typeof item === "number" && Number.isFinite(item) ? item : Number.NaN,
  );
}

/**
 * Reads an optional text option
 *
 * @param value - Its value
 * @returns The text, or undefined
 */
function textOf(value: unknown): string | undefined {
  return typeof value === "string" || typeof value === "number"
    ? String(value)
    : undefined;
}

/**
 * Reads an optional function option
 *
 * @param chart - Which piece
 * @param name - The option
 * @param value - Its value
 * @returns The function, or undefined
 */
function functionOf<T extends (...args: never[]) => unknown>(
  chart: string,
  name: string,
  value: unknown,
): T | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "function") {
    wrong(chart, `${name} must be a function`);
  }
  return value as T;
}

/**
 * Turns the options into an object, or says what is wrong
 *
 * @param chart - Which piece
 * @param value - The options
 * @returns The options
 */
function optionsOf(chart: string, value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    wrong(chart, "give the options as an object, e.g. { rows, values }");
  }
  return value as Record<string, unknown>;
}

/**
 * Checks the element to draw in
 *
 * @param chart - Which piece
 * @param element - The element
 * @returns The element
 */
function elementOf(chart: string, element: unknown): HTMLElement {
  if (!(element instanceof HTMLElement)) {
    wrong(
      chart,
      "the first argument must be an element of the panel, e.g. a div made with document.createElement",
    );
  }
  return element;
}

/**
 * The lines that the tooltip function of a plugin adds; a function that
 * fails adds none
 *
 * @param tooltip - The function, if given
 * @param data - What is under the mouse
 * @returns The lines
 */
function extraLines(
  tooltip: ((data: never) => unknown) | undefined,
  data: unknown,
): string[] {
  if (tooltip === undefined) {
    return [];
  }
  try {
    const extra = tooltip(data as never);
    return (Array.isArray(extra) ? extra : [extra])
      .filter((line) => line !== undefined && line !== null && line !== "")
      .map(String);
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------- frame

const svgSpace = "http://www.w3.org/2000/svg";

/** The size of the text of the charts, in pixels */
const fontSize = 12;

/**
 * About how wide a text is, without measuring it (the charts must work
 * before the panel is in the page)
 *
 * @param text - The text
 * @param size - The size of the font
 * @returns The width, in pixels
 */
function textWidth(text: string, size = fontSize): number {
  return text.length * size * 0.6;
}

/**
 * Makes an element of SVG
 *
 * @param parent - Where it goes
 * @param tag - Its tag, e.g. "rect"
 * @param attributes - Its attributes
 * @param text - Its text
 * @returns The element
 */
function svgElement(
  parent: Element,
  tag: string,
  attributes: Record<string, string | number>,
  text?: string,
): SVGElement {
  const element = document.createElementNS(svgSpace, tag);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  if (text !== undefined) {
    element.textContent = text;
  }
  parent.append(element);
  return element;
}

/** What is under the mouse */
type Hit = {
  /** A key, the same for the same thing */
  key: string;
  /** The lines of the tooltip: the value first */
  lines: string[];
  /** Where it is in the SVG, outlined */
  box: { x: number; y: number; width: number; height: number };
  /** What a click gives the plugin */
  data: unknown;
};

/** What drawing a chart gives back */
type Drawn = {
  /** The height of the SVG, in pixels */
  height: number;
  /** The width of the SVG, in pixels */
  width: number;
  /** What is at x, y in the SVG */
  find: (x: number, y: number) => Hit | undefined;
  /** Where the part with a key is now, to find it after a new drawing */
  centerOf: (key: string) => [number, number] | undefined;
};

/** The handle that a plugin gets for a chart */
export type ChartHandle = {
  /** Changes some options and draws it again */
  update: (options: object) => void;
  /** The chart as a PNG image */
  png: () => Promise<Blob>;
  /** Saves the chart as a PNG */
  savePng: (name?: string) => Promise<void>;
  /** The values as a CSV text */
  csv: () => string;
  /** Saves the values as a CSV */
  saveCsv: (name?: string) => void;
  /** Removes the chart and stops following the theme */
  destroy: () => void;
};

/**
 * Saves a file with a link, as the browser downloads
 *
 * @param name - The name of the file
 * @param blob - Its content
 */
function saveBlob(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

/**
 * A name of a file without characters that a file name cannot have
 *
 * @param name - The name the plugin gave
 * @param ending - The ending, e.g. "png"
 * @returns The name of the file
 */
function fileNameOf(name: string, ending: string): string {
  const base =
    name
      .replace(/\.(png|csv)$/i, "")
      .replace(/[^\w.-]+/g, "_")
      .slice(0, 80) || "chart";
  return `${base}.${ending}`;
}

/**
 * Quotes a value for a CSV when needed
 *
 * @param value - The value
 * @returns The text for the CSV
 */
function csvValue(value: string | number): string {
  if (typeof value === "number") {
    return Number.isFinite(value) ? String(value) : "";
  }
  return /[",\n\r]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
}

/** What makes a chart: how it is drawn, and its values */
type ChartMaker = {
  /** Which piece, e.g. "chat.heatmap", for errors */
  kind: string;
  /** Reads the options; throws when they are wrong */
  read: (options: Record<string, unknown>) => void;
  /** Draws it into the SVG, for a width and the colors of the theme */
  draw: (svg: SVGSVGElement, width: number, theme: ThemeColors) => Drawn;
  /** The values as rows of a CSV */
  csv: () => (string | number)[][];
  /** The name of the files */
  fileName: () => string;
  /** What a click on a part does */
  onClick: () => ((data: never) => void) | undefined;
};

/** The charts in an element, so that a new one replaces the old one */
const chartsIn = new WeakMap<HTMLElement, ChartHandle>();

/**
 * Puts a chart into an element: the SVG, the tooltip, the buttons to save
 * it; draws it again when its width or the theme changes
 *
 * @param element - The element, given by the plugin
 * @param maker - What makes the chart
 * @param options - The options of the plugin
 * @param onDestroy - Called when it is removed
 * @returns The handle
 */
function makeChart(
  element: HTMLElement,
  maker: ChartMaker,
  options: Record<string, unknown>,
  onDestroy: (handle: ChartHandle) => void,
): ChartHandle {
  let current = { ...options };
  maker.read(current);
  chartsIn.get(element)?.destroy();

  const root = document.createElement("div");
  root.dataset.chatChart = maker.kind;
  root.style.cssText =
    "position: relative; display: flex; flex-direction: column; gap: 4px;" +
    "max-width: 100%; overflow-x: auto; color: var(--foreground);";
  const svg = document.createElementNS(svgSpace, "svg");
  svg.style.display = "block";
  svg.style.cursor = "default";
  const tooltip = document.createElement("div");
  tooltip.setAttribute("role", "tooltip");
  tooltip.style.cssText =
    "position: absolute; display: none; pointer-events: none; padding: 4px 6px;" +
    "border-radius: 6px; border: 1px solid var(--border); background: var(--card);" +
    "color: var(--foreground); font-size: 12px; white-space: nowrap; z-index: 2;" +
    "box-shadow: 0 2px 6px rgb(0 0 0 / 0.15);";
  const buttons = document.createElement("div");
  buttons.style.cssText = "display: flex; gap: 6px; flex-wrap: wrap;";
  const button = (text: string, action: () => void) => {
    const made = document.createElement("button");
    made.type = "button";
    made.textContent = text;
    made.style.cssText =
      "padding: 2px 8px; border: 1px solid var(--border); border-radius: 6px;" +
      "background: var(--background); color: var(--foreground); font: inherit;" +
      "font-size: 12px; cursor: pointer;";
    made.addEventListener("click", action);
    buttons.append(made);
    return made;
  };
  root.append(svg, tooltip, buttons);
  element.append(root);

  let drawn: Drawn | undefined;
  let width = 0;
  let hover: Hit | undefined;
  let selected: string | undefined;
  let wasConnected = false;
  let destroyed = false;

  const outline = (hit: Hit | undefined, kind: "hover" | "selected") => {
    svg.querySelector(`[data-outline="${kind}"]`)?.remove();
    if (hit === undefined || drawn === undefined) {
      return;
    }
    const theme = themeColors();
    svgElement(svg, "rect", {
      "data-outline": kind,
      x: hit.box.x - 1,
      y: hit.box.y - 1,
      width: hit.box.width + 2,
      height: hit.box.height + 2,
      fill: "none",
      stroke: theme.foreground || "currentColor",
      "stroke-width": kind === "selected" ? 2 : 1.5,
      "stroke-dasharray": kind === "selected" ? "none" : "3 2",
      "pointer-events": "none",
    });
  };

  const draw = () => {
    if (destroyed) {
      return;
    }
    const theme = themeColors();
    width = Math.max(240, element.clientWidth || 480);
    svg.replaceChildren();
    svg.setAttribute("xmlns", svgSpace);
    svg.setAttribute(
      "font-family",
      getComputedStyle(root).fontFamily || "sans-serif",
    );
    svg.setAttribute("font-size", String(fontSize));
    drawn = maker.draw(svg, width, theme);
    svg.setAttribute("width", String(drawn.width));
    svg.setAttribute("height", String(drawn.height));
    svg.setAttribute("viewBox", `0 0 ${drawn.width} ${drawn.height}`);
    if (selected !== undefined) {
      const center = drawn.centerOf(selected);
      outline(
        center === undefined ? undefined : drawn.find(...center),
        "selected",
      );
    }
  };

  const where = (event: MouseEvent) => {
    const bounds = svg.getBoundingClientRect();
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  };
  const onMove = (event: MouseEvent) => {
    const { x, y } = where(event);
    const hit = drawn?.find(x, y);
    if (hit?.key !== hover?.key) {
      hover = hit;
      outline(hit, "hover");
    }
    if (hit === undefined) {
      tooltip.style.display = "none";
      svg.style.cursor = "default";
      return;
    }
    svg.style.cursor = maker.onClick() === undefined ? "default" : "pointer";
    tooltip.replaceChildren(
      ...hit.lines.map((line, index) => {
        const row = document.createElement("div");
        row.textContent = line;
        if (index === 0) {
          row.style.fontWeight = "600";
        }
        return row;
      }),
    );
    tooltip.style.display = "block";
    const rootBounds = root.getBoundingClientRect();
    const left = event.clientX - rootBounds.left + root.scrollLeft + 12;
    const room = root.clientWidth + root.scrollLeft - tooltip.offsetWidth - 4;
    tooltip.style.left = `${Math.max(0, Math.min(left, room))}px`;
    tooltip.style.top = `${event.clientY - rootBounds.top + 12}px`;
  };
  const onLeave = () => {
    hover = undefined;
    outline(undefined, "hover");
    tooltip.style.display = "none";
  };
  const onClick = (event: MouseEvent) => {
    const { x, y } = where(event);
    const hit = drawn?.find(x, y);
    const action = maker.onClick();
    if (hit === undefined || action === undefined) {
      return;
    }
    selected = hit.key;
    outline(hit, "selected");
    action(hit.data as never);
  };
  svg.addEventListener("mousemove", onMove);
  svg.addEventListener("mouseleave", onLeave);
  svg.addEventListener("click", onClick);

  const png = async (): Promise<Blob> => {
    draw();
    const copy = svg.cloneNode(true) as SVGSVGElement;
    copy.querySelectorAll("[data-outline]").forEach((part) => part.remove());
    const text = new XMLSerializer().serializeToString(copy);
    const url = URL.createObjectURL(
      new Blob([text], { type: "image/svg+xml" }),
    );
    try {
      const image = new Image();
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () =>
          reject(new Error("the chart could not be drawn as an image"));
        image.src = url;
      });
      const scale = 2;
      const canvas = document.createElement("canvas");
      canvas.width = Math.round((drawn?.width ?? width) * scale);
      canvas.height = Math.round((drawn?.height ?? 1) * scale);
      const context = canvas.getContext("2d");
      if (context === null) {
        throw new Error("this browser cannot draw the chart as an image");
      }
      context.scale(scale, scale);
      context.drawImage(image, 0, 0);
      return await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob(
          (blob) =>
            blob === null
              ? reject(new Error("the PNG could not be made"))
              : resolve(blob),
          "image/png",
        ),
      );
    } finally {
      URL.revokeObjectURL(url);
    }
  };
  const csv = () =>
    maker
      .csv()
      .map((row) => row.map(csvValue).join(","))
      .join("\n") + "\n";

  const handle: ChartHandle = {
    update: (changes: object) => {
      const next = { ...current, ...optionsOf(maker.kind, changes) };
      maker.read(next);
      current = next;
      selected = undefined;
      draw();
    },
    png,
    savePng: async (name?: string) =>
      saveBlob(fileNameOf(name ?? maker.fileName(), "png"), await png()),
    csv,
    saveCsv: (name?: string) =>
      saveBlob(
        fileNameOf(name ?? maker.fileName(), "csv"),
        new Blob([csv()], { type: "text/csv" }),
      ),
    destroy: () => {
      if (destroyed) {
        return;
      }
      destroyed = true;
      stopTheme();
      resize?.disconnect();
      root.remove();
      if (chartsIn.get(element) === handle) {
        chartsIn.delete(element);
      }
      onDestroy(handle);
    },
  };

  const status = document.createElement("span");
  status.style.cssText = "color: var(--muted-foreground); font-size: 12px;";
  button("Save PNG", () => {
    status.textContent = "";
    handle.savePng().catch((error: unknown) => {
      status.textContent = `Could not save the PNG: ${error instanceof Error ? error.message : String(error)}`;
    });
  });
  button("Save CSV", () => handle.saveCsv());
  buttons.append(status);

  // drawn again when the theme or the width changes; removed once the
  // panel that holds it is gone
  const stopTheme = watchTheme(() => {
    if (wasConnected && !root.isConnected) {
      handle.destroy();
      return;
    }
    draw();
  });
  const resize =
    typeof ResizeObserver === "undefined"
      ? undefined
      : new ResizeObserver(() => {
          if (root.isConnected) {
            wasConnected = true;
          } else if (wasConnected) {
            handle.destroy();
            return;
          }
          const now = Math.max(240, element.clientWidth || 480);
          if (now !== width) {
            draw();
          }
        });
  resize?.observe(element);
  chartsIn.set(element, handle);
  draw();
  return handle;
}

/**
 * Writes the title of a chart at its top
 *
 * @param svg - The chart
 * @param title - The title
 * @param x - Its middle
 * @param theme - The colors
 */
function drawTitle(
  svg: SVGSVGElement,
  title: string | undefined,
  x: number,
  theme: ThemeColors,
): void {
  if (title !== undefined) {
    svgElement(
      svg,
      "text",
      {
        x,
        y: 16,
        "text-anchor": "middle",
        "font-weight": 600,
        "font-size": 13,
        fill: theme.foreground,
      },
      title,
    );
  }
}

/**
 * Draws the background of a chart, so that the PNG has it too
 *
 * @param svg - The chart
 * @param width - Its width
 * @param height - Its height
 * @param theme - The colors
 */
function drawBackground(
  svg: SVGSVGElement,
  width: number,
  height: number,
  theme: ThemeColors,
): void {
  const background = svgElement(svg, "rect", {
    x: 0,
    y: 0,
    width,
    height,
    fill: theme.background || (theme.dark ? "#1a1a19" : "#ffffff"),
  });
  svg.prepend(background);
}

/**
 * Draws a vertical color bar with its numbers
 *
 * @param svg - The chart
 * @param place - Where: its left and top, its height
 * @param scale - The color of a value, the lowest and highest value
 * @param title - What the colors show
 * @param theme - The colors of the theme
 */
function drawColorBar(
  svg: SVGSVGElement,
  place: { x: number; y: number; height: number },
  scale: {
    color: (value: number) => string;
    low: number;
    high: number;
    signed: boolean;
    /** Whether some values are beyond the low or the high end */
    beyond: { low: boolean; high: boolean };
  },
  title: string,
  theme: ThemeColors,
): void {
  const id = `chat-scale-${Math.random().toString(36).slice(2)}`;
  const definitions = svgElement(svg, "defs", {});
  const gradient = svgElement(definitions, "linearGradient", {
    id,
    x1: 0,
    y1: 1,
    x2: 0,
    y2: 0,
  });
  for (let step = 0; step <= 20; step++) {
    const share = step / 20;
    svgElement(gradient, "stop", {
      offset: share,
      "stop-color": scale.color(scale.low + (scale.high - scale.low) * share),
    });
  }
  svgElement(
    svg,
    "text",
    {
      x: place.x,
      y: place.y - 14,
      fill: theme.foreground,
      "data-legend-title": "true",
    },
    title,
  );
  svgElement(svg, "rect", {
    x: place.x,
    y: place.y,
    width: 14,
    height: place.height,
    fill: `url(#${id})`,
    stroke: theme.border,
  });
  // round numbers inside the scale (0 among them for values with a sign);
  // the ends too when no round number is near them, and always when values
  // go beyond them ("≥ 15")
  const span = scale.high - scale.low || 1;
  const yOf = (value: number) =>
    place.y + place.height * ((scale.high - value) / span);
  const near = 14;
  const inner = niceTicks(scale.low, scale.high, 4).filter(
    (tick) =>
      tick >= scale.low - span * 1e-9 && tick <= scale.high + span * 1e-9,
  );
  const ends: { value: number; text: string; beyond: boolean }[] = [
    {
      value: scale.high,
      text: `${scale.beyond.high ? "≥ " : ""}${formatNumber(scale.high)}`,
      beyond: scale.beyond.high,
    },
    {
      value: scale.low,
      text: `${scale.beyond.low ? "≤ " : ""}${formatNumber(scale.low)}`,
      beyond: scale.beyond.low,
    },
  ];
  const ticks: { value: number; text: string }[] = [];
  for (const end of ends) {
    const closest = Math.min(
      ...inner.map((tick) => Math.abs(yOf(tick) - yOf(end.value))),
    );
    if (end.beyond || closest > 0.2 * place.height) {
      ticks.push(end);
    }
  }
  for (const tick of inner) {
    if (
      ticks.every((other) => Math.abs(yOf(other.value) - yOf(tick)) >= near)
    ) {
      ticks.push({ value: tick, text: formatNumber(tick) });
    }
  }
  ticks.sort((a, b) => b.value - a.value);
  for (const { value: tick, text } of ticks) {
    const y = yOf(tick);
    svgElement(svg, "line", {
      x1: place.x + 14,
      x2: place.x + 18,
      y1: y,
      y2: y,
      stroke: theme.foreground,
    });
    svgElement(
      svg,
      "text",
      {
        x: place.x + 21,
        y,
        "dominant-baseline": "middle",
        fill: theme.foreground,
        "data-legend-tick": "true",
      },
      text,
    );
  }
}

// ---------------------------------------------------------------- heatmap

/** A cell of a heatmap, as a click and the tooltip get it */
export type HeatmapCell = {
  row: string;
  column: string;
  value: number | null;
  rowIndex: number;
  columnIndex: number;
};

/**
 * chat.heatmap(element, options): a heatmap with the names of the rows and
 * columns, a color bar, the values under the mouse, and Save PNG / CSV
 *
 * @param element - Where it goes, an element of the panel of the plugin
 * @param given - The options: rows, columns, values, title, rowTitle,
 * columnTitle, valueTitle, signed, limit, ignoreDiagonal, tooltip(cell),
 * onClick(cell), fileName, sort
 * @param onDestroy - Called when it is removed
 * @returns The handle
 */
export function heatmap(
  element: unknown,
  given: unknown,
  onDestroy: (handle: ChartHandle) => void = () => undefined,
): ChartHandle {
  const kind = "chat.heatmap";
  let rows: string[] = [];
  let columns: string[] = [];
  let values: number[][] = [];
  let rowOrder: number[] = [];
  let columnOrder: number[] = [];
  let options: Record<string, unknown> = {};
  let signed = false;
  let low = 0;
  let high = 1;
  let beyond = { low: false, high: false };

  const cellOf = (row: number, column: number): HeatmapCell => {
    const value = values[row]![column]!;
    return {
      row: rows[row]!,
      column: columns[column]!,
      value: Number.isFinite(value) ? value : null,
      rowIndex: row,
      columnIndex: column,
    };
  };

  const maker: ChartMaker = {
    kind,
    read: (read) => {
      options = read;
      rows = namesOf(kind, "rows", read["rows"]);
      columns =
        read["columns"] === undefined
          ? rows
          : namesOf(kind, "columns", read["columns"]);
      const matrix = read["values"];
      if (
        !Array.isArray(matrix) ||
        matrix.length !== rows.length ||
        rows.length === 0
      ) {
        wrong(
          kind,
          `values must be a list of ${rows.length || "the"} rows (one per name in rows), each a list of numbers, one per column`,
        );
      }
      values = matrix.map((row: unknown, index) => {
        const numbers = numbersOf(kind, `values[${index}]`, row);
        if (numbers.length !== columns.length) {
          wrong(
            kind,
            `values[${index}] has ${numbers.length} numbers, but there are ${columns.length} columns`,
          );
        }
        return numbers;
      });
      functionOf(kind, "onClick", read["onClick"]);
      functionOf(kind, "tooltip", read["tooltip"]);
      const sort = read["sort"] !== false;
      rowOrder = sort ? numberOrder(rows) : rows.map((_, i) => i);
      columnOrder = sort ? numberOrder(columns) : columns.map((_, i) => i);
      // with ignoreDiagonal, the scale is made without the cells of a group
      // with itself (e.g. neighborhood enrichment), which get the end colors
      const ignoreDiagonal = read["ignoreDiagonal"] === true;
      const all = values.flat().filter(Number.isFinite);
      const finite = values
        .flatMap((row, r) =>
          row.filter((_, c) => !ignoreDiagonal || rows[r] !== columns[c]),
        )
        .filter(Number.isFinite);
      const smallest = finite.length === 0 ? 0 : Math.min(...finite);
      const largest = finite.length === 0 ? 1 : Math.max(...finite);
      signed =
        typeof read["signed"] === "boolean" ? read["signed"] : smallest < 0;
      const limit = Number(read["limit"]);
      if (signed) {
        const biggest =
          Number.isFinite(limit) && limit > 0
            ? limit
            : Math.max(Math.abs(smallest), Math.abs(largest)) || 1;
        low = -biggest;
        high = biggest;
      } else {
        low = smallest;
        high = largest > smallest ? largest : smallest + 1;
      }
      beyond = {
        low: all.some((value) => value < low),
        high: all.some((value) => value > high),
      };
    },
    draw: (svg, width, theme) => {
      const title = textOf(options["title"]);
      const rowTitle = textOf(options["rowTitle"]);
      const columnTitle = textOf(options["columnTitle"]);
      const valueTitle = textOf(options["valueTitle"]) ?? "Value";
      const color = (value: number) =>
        signed
          ? signedColor(value, high, theme.dark)
          : sequentialColor(value, low, high, theme.dark);
      const pad = 8;
      const top = pad + (title === undefined ? 0 : 20) + 22;
      const rowLabels = Math.max(...rows.map((name) => textWidth(name))) + 8;
      const left = pad + (rowTitle === undefined ? 0 : 18) + rowLabels;
      const tickTexts = [`≤ ${formatNumber(low)}`, `≥ ${formatNumber(high)}`];
      const legendWidth =
        Math.max(
          21 + Math.max(...tickTexts.map((text) => textWidth(text))),
          textWidth(valueTitle),
        ) + 20;
      const room = width - left - pad - legendWidth - 16;
      const cell = Math.max(8, Math.min(44, Math.floor(room / columns.length)));
      const gap = cell >= 16 ? 2 : 1;
      const longest = Math.max(...columns.map((name) => textWidth(name)));
      const turned = longest > cell;
      const columnLabels = turned ? longest + 8 : 18;
      const gridWidth = cell * columns.length;
      const gridHeight = cell * rows.length;
      const legendX = left + gridWidth + 16;
      const legendHeight = Math.max(80, Math.min(220, gridHeight));
      const chartWidth = Math.max(width, legendX + legendWidth);
      const height =
        Math.max(top + gridHeight, top + legendHeight + 4) +
        columnLabels +
        (columnTitle === undefined ? 0 : 20) +
        pad;
      drawTitle(svg, title, left + gridWidth / 2, theme);

      // the cells, in the order of the rows and columns
      rowOrder.forEach((row, y) => {
        columnOrder.forEach((column, x) => {
          const value = values[row]![column]!;
          svgElement(svg, "rect", {
            x: left + x * cell,
            y: top + y * cell,
            width: cell - gap,
            height: cell - gap,
            fill: Number.isFinite(value) ? color(value) : theme.muted,
            "data-cell": `${rows[row]}|${columns[column]}`,
          });
        });
      });
      // the names of the rows, left, and of the columns, below
      rowOrder.forEach((row, y) => {
        svgElement(
          svg,
          "text",
          {
            x: left - 6,
            y: top + y * cell + (cell - gap) / 2,
            "text-anchor": "end",
            "dominant-baseline": "middle",
            fill: theme.foreground,
            "data-row-label": "true",
          },
          rows[row],
        );
      });
      const below = top + gridHeight + 4;
      columnOrder.forEach((column, x) => {
        const middle = left + x * cell + (cell - gap) / 2;
        svgElement(
          svg,
          "text",
          turned
            ? {
                x: middle,
                y: below,
                transform: `rotate(-90 ${middle} ${below})`,
                "text-anchor": "end",
                "dominant-baseline": "middle",
                fill: theme.foreground,
                "data-column-label": "true",
              }
            : {
                x: middle,
                y: below + 10,
                "text-anchor": "middle",
                fill: theme.foreground,
                "data-column-label": "true",
              },
          columns[column],
        );
      });
      if (rowTitle !== undefined) {
        const middle = top + gridHeight / 2;
        svgElement(
          svg,
          "text",
          {
            x: pad + 10,
            y: middle,
            transform: `rotate(-90 ${pad + 10} ${middle})`,
            "text-anchor": "middle",
            fill: theme.mutedForeground || theme.foreground,
          },
          rowTitle,
        );
      }
      if (columnTitle !== undefined) {
        svgElement(
          svg,
          "text",
          {
            x: left + gridWidth / 2,
            y: below + columnLabels + 12,
            "text-anchor": "middle",
            fill: theme.mutedForeground || theme.foreground,
          },
          columnTitle,
        );
      }
      drawColorBar(
        svg,
        { x: legendX, y: top, height: legendHeight },
        { color, low, high, signed, beyond },
        valueTitle,
        theme,
      );
      drawBackground(svg, chartWidth, height, theme);

      const tooltip = functionOf<(cell: HeatmapCell) => unknown>(
        kind,
        "tooltip",
        options["tooltip"],
      );
      return {
        width: chartWidth,
        height,
        find: (x, y) => {
          const across = Math.floor((x - left) / cell);
          const down = Math.floor((y - top) / cell);
          if (
            x < left ||
            y < top ||
            across >= columns.length ||
            down >= rows.length
          ) {
            return undefined;
          }
          const found = cellOf(rowOrder[down]!, columnOrder[across]!);
          const more = extraLines(tooltip, found);
          return {
            key: `${found.rowIndex},${found.columnIndex}`,
            lines: [
              `${valueTitle}: ${found.value === null ? "no value" : formatNumber(found.value)}`,
              `${rowTitle ?? "Row"} ${found.row}, ${(columnTitle ?? "column").toLowerCase()} ${found.column}`,
              ...more,
            ],
            box: {
              x: left + across * cell,
              y: top + down * cell,
              width: cell - gap,
              height: cell - gap,
            },
            data: found,
          };
        },
        centerOf: (key) => {
          const [row, column] = key.split(",").map(Number);
          const down = rowOrder.indexOf(row!);
          const across = columnOrder.indexOf(column!);
          return down < 0 || across < 0
            ? undefined
            : [left + (across + 0.5) * cell, top + (down + 0.5) * cell];
        },
      };
    },
    csv: () => [
      [
        textOf(options["rowTitle"]) ?? "row",
        ...columnOrder.map((c) => columns[c]!),
      ],
      ...rowOrder.map((row) => [
        rows[row]!,
        ...columnOrder.map((column) => values[row]![column]!),
      ]),
    ],
    fileName: () =>
      textOf(options["fileName"]) ?? textOf(options["title"]) ?? "heatmap",
    onClick: () => functionOf(kind, "onClick", options["onClick"]),
  };
  return makeChart(
    elementOf(kind, element),
    maker,
    optionsOf(kind, given),
    onDestroy,
  );
}

// ---------------------------------------------------------------- axes

/** A linear scale from values to pixels */
type Axis = {
  ticks: number[];
  at: (value: number) => number;
};

/**
 * A linear scale with round ticks
 *
 * @param low - The lowest value
 * @param high - The highest value
 * @param from - The pixel of the lowest tick
 * @param to - The pixel of the highest tick
 * @returns The scale
 */
function linearAxis(low: number, high: number, from: number, to: number): Axis {
  const ticks = niceTicks(low, high);
  const first = ticks[0]!;
  const last = ticks[ticks.length - 1]!;
  return {
    ticks,
    at: (value) => from + ((value - first) / (last - first || 1)) * (to - from),
  };
}

/**
 * Draws the ticks, numbers and grid of a vertical axis
 *
 * @param svg - The chart
 * @param axis - The scale
 * @param left - Where the plot starts
 * @param right - Where it ends
 * @param theme - The colors
 */
function drawValueAxis(
  svg: SVGSVGElement,
  axis: Axis,
  left: number,
  right: number,
  theme: ThemeColors,
): void {
  for (const tick of axis.ticks) {
    const y = axis.at(tick);
    svgElement(svg, "line", {
      x1: left,
      x2: right,
      y1: y,
      y2: y,
      stroke: theme.border,
      "stroke-width": tick === 0 ? 1.5 : 1,
    });
    svgElement(
      svg,
      "text",
      {
        x: left - 6,
        y,
        "text-anchor": "end",
        "dominant-baseline": "middle",
        fill: theme.mutedForeground || theme.foreground,
        "data-y-tick": "true",
      },
      formatNumber(tick),
    );
  }
}

/**
 * Writes the title of the vertical axis, turned, at the left
 *
 * @param svg - The chart
 * @param title - The title
 * @param middle - The middle of the axis
 * @param theme - The colors
 */
function drawYTitle(
  svg: SVGSVGElement,
  title: string | undefined,
  middle: number,
  theme: ThemeColors,
): void {
  if (title !== undefined) {
    svgElement(
      svg,
      "text",
      {
        x: 18,
        y: middle,
        transform: `rotate(-90 18 ${middle})`,
        "text-anchor": "middle",
        fill: theme.mutedForeground || theme.foreground,
      },
      title,
    );
  }
}

/**
 * Reads colors given by the plugin: a list, or an object by name
 *
 * @param given - The colors
 * @param names - The names of the groups
 * @returns The color of each group, if given
 */
function colorsOf(given: unknown, names: string[]): (string | undefined)[] {
  if (Array.isArray(given)) {
    return names.map((_, index) => cssColor(given[index]));
  }
  if (typeof given === "object" && given !== null) {
    const byName = new Map(
      Object.entries(given).map(([name, color]) => [groupName(name), color]),
    );
    return names.map((name) => cssColor(byName.get(name)));
  }
  const one = cssColor(given);
  return names.map(() => one);
}

/**
 * Turns a color given by a plugin into CSS: a CSS color, {r, g, b} or
 * [r, g, b]
 *
 * @param value - The color
 * @returns The CSS color, or undefined
 */
function cssColor(value: unknown): string | undefined {
  if (typeof value === "string" && value.trim() !== "") {
    return value.trim();
  }
  const parts = Array.isArray(value)
    ? value
    : typeof value === "object" && value !== null
      ? [
          (value as Record<string, unknown>)["r"],
          (value as Record<string, unknown>)["g"],
          (value as Record<string, unknown>)["b"],
        ]
      : [];
  return parts.length >= 3 &&
    parts.slice(0, 3).every((p) => typeof p === "number")
    ? `rgb(${(parts.slice(0, 3) as number[]).map(Math.round).join(", ")})`
    : undefined;
}

// ---------------------------------------------------------------- bars

/** A bar, as a click and the tooltip get it */
export type Bar = { label: string; value: number | null; index: number };

/**
 * chat.bars(element, options): a bar chart with numbers on the axis, the
 * value under the mouse, and Save PNG / CSV
 *
 * @param element - Where it goes
 * @param given - The options: labels, values, title, labelTitle,
 * valueTitle, colors, tooltip(bar), onClick(bar), fileName, sort
 * @param onDestroy - Called when it is removed
 * @returns The handle
 */
export function bars(
  element: unknown,
  given: unknown,
  onDestroy: (handle: ChartHandle) => void = () => undefined,
): ChartHandle {
  const kind = "chat.bars";
  let labels: string[] = [];
  let values: number[] = [];
  let order: number[] = [];
  let options: Record<string, unknown> = {};

  const maker: ChartMaker = {
    kind,
    read: (read) => {
      options = read;
      labels = namesOf(kind, "labels", read["labels"]);
      values = numbersOf(kind, "values", read["values"]);
      if (labels.length === 0 || values.length !== labels.length) {
        wrong(kind, "give labels and values, one number for each label");
      }
      functionOf(kind, "onClick", read["onClick"]);
      functionOf(kind, "tooltip", read["tooltip"]);
      order =
        read["sort"] === false ? labels.map((_, i) => i) : numberOrder(labels);
    },
    draw: (svg, width, theme) => {
      const title = textOf(options["title"]);
      const labelTitle = textOf(options["labelTitle"]);
      const valueTitle = textOf(options["valueTitle"]);
      const finite = values.filter(Number.isFinite);
      const axisHeight = 200;
      const top = 8 + (title === undefined ? 0 : 20) + 6;
      const value = linearAxis(
        Math.min(0, ...finite),
        Math.max(0, ...finite),
        top + axisHeight,
        top,
      );
      const tickWidth = Math.max(
        ...value.ticks.map((tick) => textWidth(formatNumber(tick))),
      );
      const left = (valueTitle === undefined ? 8 : 30) + tickWidth + 8;
      const band = Math.max(
        10,
        Math.min(60, Math.floor((width - left - 8) / labels.length)),
      );
      const plotWidth = band * labels.length;
      const longest = Math.max(...labels.map((label) => textWidth(label)));
      const turned = longest > band - 4;
      const labelHeight = turned ? longest + 8 : 18;
      const height =
        top +
        axisHeight +
        labelHeight +
        (labelTitle === undefined ? 0 : 20) +
        8;
      const chartWidth = Math.max(width, left + plotWidth + 8);
      const colors = colorsOf(options["colors"], labels);
      const series = seriesColors[theme.dark ? "dark" : "light"][0]!;
      drawTitle(svg, title, left + plotWidth / 2, theme);
      drawValueAxis(svg, value, left, left + plotWidth, theme);
      drawYTitle(svg, valueTitle, top + axisHeight / 2, theme);
      const zero = value.at(0);
      const barWidth = Math.max(2, band - Math.max(2, Math.round(band * 0.25)));
      const boxes: Hit["box"][] = [];
      order.forEach((index, place) => {
        const amount = values[index]!;
        const x = left + place * band + (band - barWidth) / 2;
        const end = Number.isFinite(amount) ? value.at(amount) : zero;
        const box = {
          x,
          y: Math.min(zero, end),
          width: barWidth,
          height: Math.max(
            Number.isFinite(amount) ? 1 : 0,
            Math.abs(end - zero),
          ),
        };
        boxes.push(box);
        svgElement(svg, "rect", {
          ...box,
          fill: colors[index] ?? series,
          "data-bar": labels[index]!,
        });
        const middle = left + place * band + band / 2;
        const below = top + axisHeight + 4;
        svgElement(
          svg,
          "text",
          turned
            ? {
                x: middle,
                y: below,
                transform: `rotate(-90 ${middle} ${below})`,
                "text-anchor": "end",
                "dominant-baseline": "middle",
                fill: theme.foreground,
                "data-x-label": "true",
              }
            : {
                x: middle,
                y: below + 10,
                "text-anchor": "middle",
                fill: theme.foreground,
                "data-x-label": "true",
              },
          labels[index],
        );
      });
      if (labelTitle !== undefined) {
        svgElement(
          svg,
          "text",
          {
            x: left + plotWidth / 2,
            y: top + axisHeight + labelHeight + 16,
            "text-anchor": "middle",
            fill: theme.mutedForeground || theme.foreground,
          },
          labelTitle,
        );
      }
      drawBackground(svg, chartWidth, height, theme);
      const tooltip = functionOf<(bar: Bar) => unknown>(
        kind,
        "tooltip",
        options["tooltip"],
      );
      return {
        width: chartWidth,
        height,
        find: (x, y) => {
          const place = Math.floor((x - left) / band);
          if (
            x < left ||
            place >= labels.length ||
            y < top ||
            y > top + axisHeight
          ) {
            return undefined;
          }
          const index = order[place]!;
          const amount = values[index]!;
          const bar: Bar = {
            label: labels[index]!,
            value: Number.isFinite(amount) ? amount : null,
            index,
          };
          return {
            key: String(index),
            lines: [
              `${valueTitle ?? "Value"}: ${bar.value === null ? "no value" : formatNumber(bar.value)}`,
              `${labelTitle ?? "Label"} ${bar.label}`,
              ...extraLines(tooltip, bar),
            ],
            box: {
              x: left + place * band,
              y: top,
              width: band,
              height: axisHeight,
            },
            data: bar,
          };
        },
        centerOf: (key) => {
          const place = order.indexOf(Number(key));
          return place < 0
            ? undefined
            : [left + (place + 0.5) * band, top + axisHeight / 2];
        },
      };
    },
    csv: () => [
      [
        textOf(options["labelTitle"]) ?? "label",
        textOf(options["valueTitle"]) ?? "value",
      ],
      ...order.map((index) => [labels[index]!, values[index]!]),
    ],
    fileName: () =>
      textOf(options["fileName"]) ?? textOf(options["title"]) ?? "bars",
    onClick: () => functionOf(kind, "onClick", options["onClick"]),
  };
  return makeChart(
    elementOf(kind, element),
    maker,
    optionsOf(kind, given),
    onDestroy,
  );
}

// ---------------------------------------------------------------- scatter

/** The most points that chat.scatter draws */
export const maxScatterPoints = 20_000;

/** A point, as a click and the tooltip get it */
export type ScatterPoint = {
  index: number;
  x: number;
  y: number;
  group?: string;
  label?: string;
};

/**
 * chat.scatter(element, options): points with numbers on both axes, colors
 * by group with a legend, the point under the mouse, and Save PNG / CSV
 *
 * @param element - Where it goes
 * @param given - The options: x, y, groups, labels, title, xTitle, yTitle,
 * groupTitle, colors, tooltip(point), onClick(point), fileName
 * @param onDestroy - Called when it is removed
 * @returns The handle
 */
export function scatter(
  element: unknown,
  given: unknown,
  onDestroy: (handle: ChartHandle) => void = () => undefined,
): ChartHandle {
  const kind = "chat.scatter";
  let xs: number[] = [];
  let ys: number[] = [];
  let groups: string[] | undefined;
  let names: string[] | undefined;
  let groupList: string[] = [];
  let options: Record<string, unknown> = {};

  const maker: ChartMaker = {
    kind,
    read: (read) => {
      options = read;
      xs = numbersOf(kind, "x", read["x"]);
      ys = numbersOf(kind, "y", read["y"]);
      if (xs.length === 0 || xs.length !== ys.length) {
        wrong(kind, "give x and y, two lists of numbers of the same length");
      }
      if (xs.length > maxScatterPoints) {
        wrong(
          kind,
          `${xs.length} points are too many: give at most ${maxScatterPoints}, e.g. a random sample of them`,
        );
      }
      groups =
        read["groups"] === undefined
          ? undefined
          : namesOf(kind, "groups", read["groups"]);
      if (groups !== undefined && groups.length !== xs.length) {
        wrong(kind, "groups must have one name for each point");
      }
      names =
        read["labels"] === undefined
          ? undefined
          : namesOf(kind, "labels", read["labels"]);
      functionOf(kind, "onClick", read["onClick"]);
      functionOf(kind, "tooltip", read["tooltip"]);
      const unique = [...new Set(groups ?? [])];
      groupList = numberOrder(unique).map((index) => unique[index]!);
    },
    draw: (svg, width, theme) => {
      const title = textOf(options["title"]);
      const xTitle = textOf(options["xTitle"]);
      const yTitle = textOf(options["yTitle"]);
      const groupTitle = textOf(options["groupTitle"]);
      const finiteX = xs.filter(Number.isFinite);
      const finiteY = ys.filter(Number.isFinite);
      const legendWidth =
        groupList.length === 0
          ? 0
          : Math.max(
              textWidth(groupTitle ?? ""),
              ...groupList.slice(0, 30).map((group) => textWidth(group) + 18),
            ) + 16;
      const top = 8 + (title === undefined ? 0 : 20) + 6;
      const plotHeight = 260;
      const yAxis = linearAxis(
        Math.min(...finiteY),
        Math.max(...finiteY),
        top + plotHeight,
        top,
      );
      const tickWidth = Math.max(
        ...yAxis.ticks.map((tick) => textWidth(formatNumber(tick))),
      );
      const left = (yTitle === undefined ? 8 : 30) + tickWidth + 8;
      const plotWidth = Math.max(160, width - left - 12 - legendWidth);
      const xAxis = linearAxis(
        Math.min(...finiteX),
        Math.max(...finiteX),
        left,
        left + plotWidth,
      );
      const height =
        top + plotHeight + 22 + (xTitle === undefined ? 0 : 18) + 8;
      const chartWidth = Math.max(width, left + plotWidth + 12 + legendWidth);
      drawTitle(svg, title, left + plotWidth / 2, theme);
      drawValueAxis(svg, yAxis, left, left + plotWidth, theme);
      drawYTitle(svg, yTitle, top + plotHeight / 2, theme);
      for (const tick of xAxis.ticks) {
        const x = xAxis.at(tick);
        svgElement(svg, "line", {
          x1: x,
          x2: x,
          y1: top,
          y2: top + plotHeight,
          stroke: theme.border,
        });
        svgElement(
          svg,
          "text",
          {
            x,
            y: top + plotHeight + 16,
            "text-anchor": "middle",
            fill: theme.mutedForeground || theme.foreground,
            "data-x-tick": "true",
          },
          formatNumber(tick),
        );
      }
      if (xTitle !== undefined) {
        svgElement(
          svg,
          "text",
          {
            x: left + plotWidth / 2,
            y: top + plotHeight + 36,
            "text-anchor": "middle",
            fill: theme.mutedForeground || theme.foreground,
          },
          xTitle,
        );
      }
      const given = colorsOf(options["colors"], groupList);
      const colorOfGroup = new Map(
        groupList.map((group, index) => [
          group,
          given[index] ?? groupColor(index, groupList.length, theme.dark),
        ]),
      );
      const single = seriesColors[theme.dark ? "dark" : "light"][0]!;
      const radius = xs.length > 5000 ? 2 : 3;
      const points = svgElement(svg, "g", {
        stroke: theme.background,
        "stroke-width": xs.length > 5000 ? 0 : 1,
      });
      const placed: [number, number][] = xs.map((x, index) => [
        xAxis.at(x),
        yAxis.at(ys[index]!),
      ]);
      placed.forEach(([x, y], index) => {
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          return;
        }
        svgElement(points, "circle", {
          cx: x,
          cy: y,
          r: radius,
          fill:
            groups === undefined
              ? single
              : (colorOfGroup.get(groups[index]!) ?? single),
        });
      });
      if (groupList.length > 0) {
        const legendX = left + plotWidth + 16;
        let y = top + 4;
        if (groupTitle !== undefined) {
          svgElement(
            svg,
            "text",
            {
              x: legendX,
              y: y + 4,
              fill: theme.foreground,
              "font-weight": 600,
            },
            groupTitle,
          );
          y += 18;
        }
        for (const group of groupList.slice(0, 30)) {
          svgElement(svg, "rect", {
            x: legendX,
            y: y - 5,
            width: 10,
            height: 10,
            rx: 2,
            fill: colorOfGroup.get(group)!,
          });
          svgElement(
            svg,
            "text",
            {
              x: legendX + 16,
              y,
              "dominant-baseline": "middle",
              fill: theme.foreground,
              "data-legend-item": "true",
            },
            group,
          );
          y += 16;
        }
        if (groupList.length > 30) {
          svgElement(
            svg,
            "text",
            { x: legendX, y: y + 4, fill: theme.mutedForeground },
            `… ${groupList.length - 30} more`,
          );
        }
      }
      drawBackground(
        svg,
        chartWidth,
        Math.max(height, top + 18 * Math.min(31, groupList.length + 1)),
        theme,
      );
      const tooltip = functionOf<(point: ScatterPoint) => unknown>(
        kind,
        "tooltip",
        options["tooltip"],
      );
      return {
        width: chartWidth,
        height: Math.max(height, top + 18 * Math.min(31, groupList.length + 1)),
        find: (x, y) => {
          let best = -1;
          let distance = 64;
          placed.forEach(([px, py], index) => {
            const d = (px - x) ** 2 + (py - y) ** 2;
            if (d < distance) {
              distance = d;
              best = index;
            }
          });
          if (best < 0) {
            return undefined;
          }
          const point: ScatterPoint = {
            index: best,
            x: xs[best]!,
            y: ys[best]!,
            ...(groups === undefined ? {} : { group: groups[best]! }),
            ...(names?.[best] === undefined ? {} : { label: names[best] }),
          };
          const [px, py] = placed[best]!;
          return {
            key: String(best),
            lines: [
              `${xTitle ?? "x"} ${formatNumber(point.x)}, ${yTitle ?? "y"} ${formatNumber(point.y)}`,
              ...(point.label === undefined ? [] : [point.label]),
              ...(point.group === undefined
                ? []
                : [`${groupTitle ?? "Group"} ${point.group}`]),
              ...extraLines(tooltip, point),
            ],
            box: {
              x: px - radius,
              y: py - radius,
              width: radius * 2,
              height: radius * 2,
            },
            data: point,
          };
        },
        centerOf: (key) => placed[Number(key)],
      };
    },
    csv: () => [
      [
        textOf(options["xTitle"]) ?? "x",
        textOf(options["yTitle"]) ?? "y",
        ...(groups === undefined
          ? []
          : [textOf(options["groupTitle"]) ?? "group"]),
        ...(names === undefined ? [] : ["label"]),
      ],
      ...xs.map((x, index) => [
        x,
        ys[index]!,
        ...(groups === undefined ? [] : [groups[index]!]),
        ...(names === undefined ? [] : [names[index] ?? ""]),
      ]),
    ],
    fileName: () =>
      textOf(options["fileName"]) ?? textOf(options["title"]) ?? "scatter",
    onClick: () => functionOf(kind, "onClick", options["onClick"]),
  };
  return makeChart(
    elementOf(kind, element),
    maker,
    optionsOf(kind, given),
    onDestroy,
  );
}

// ---------------------------------------------------------------- legend

/** The handle that a plugin gets for a legend */
export type LegendHandle = {
  /** Changes some options and shows it again */
  update: (options: object) => void;
  /** Removes it */
  destroy: () => void;
};

/**
 * chat.legend(element, options): the colors that a plugin gave to the
 * cells or points on the image, with their groups, and a button that puts
 * the colors from before back
 *
 * @param element - Where it goes
 * @param given - The options: title, items ([{ label, color }] or
 * { label: color }), onReset, resetText, note, sort
 * @returns The handle
 */
export function legend(element: unknown, given: unknown): LegendHandle {
  const kind = "chat.legend";
  const parent = elementOf(kind, element);
  const root = document.createElement("div");
  root.dataset.chatLegend = "true";
  root.style.cssText =
    "display: flex; flex-direction: column; gap: 4px; padding: 6px 8px;" +
    "border: 1px solid var(--border); border-radius: 6px; color: var(--foreground);" +
    "font-size: 12px;";
  let options: Record<string, unknown> = {};

  const show = (read: Record<string, unknown>) => {
    const raw = read["items"];
    const entries: [string, unknown][] = Array.isArray(raw)
      ? raw.map((item: unknown) => {
          const record = (
            typeof item === "object" && item !== null ? item : {}
          ) as Record<string, unknown>;
          return [groupName(record["label"]), record["color"]];
        })
      : typeof raw === "object" && raw !== null
        ? Object.entries(raw).map(([label, color]) => [groupName(label), color])
        : wrong(
            kind,
            'items must be a list of { label, color }, e.g. [{ label: "3", color: "#e34948" }]',
          );
    const items = entries.map(([label, color]) => {
      const css = cssColor(color);
      if (css === undefined) {
        wrong(
          kind,
          `the color of "${label}" must be a CSS color, e.g. "#e34948"`,
        );
      }
      return { label, color: css };
    });
    const onReset = functionOf<() => unknown>(kind, "onReset", read["onReset"]);
    const order =
      read["sort"] === false
        ? items.map((_, i) => i)
        : numberOrder(items.map(({ label }) => label));
    options = read;

    const parts: HTMLElement[] = [];
    const title = textOf(read["title"]);
    if (title !== undefined) {
      const heading = document.createElement("div");
      heading.style.fontWeight = "600";
      heading.textContent = title;
      parts.push(heading);
    }
    const list = document.createElement("div");
    list.style.cssText = "display: flex; flex-wrap: wrap; gap: 4px 12px;";
    for (const index of order) {
      const { label, color } = items[index]!;
      const item = document.createElement("span");
      item.dataset.legendItem = label;
      item.style.cssText =
        "display: inline-flex; align-items: center; gap: 6px;";
      const swatch = document.createElement("span");
      swatch.style.cssText =
        "width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0;" +
        "border: 1px solid var(--border);";
      swatch.style.background = color;
      const text = document.createElement("span");
      text.textContent = label;
      item.append(swatch, text);
      list.append(item);
    }
    parts.push(list);
    const note = textOf(read["note"]);
    if (note !== undefined) {
      const line = document.createElement("div");
      line.style.color = "var(--muted-foreground)";
      line.textContent = note;
      parts.push(line);
    }
    if (onReset !== undefined) {
      const reset = document.createElement("button");
      reset.type = "button";
      reset.textContent = textOf(read["resetText"]) ?? "Reset colors";
      reset.style.cssText =
        "align-self: flex-start; padding: 2px 8px; border: 1px solid var(--border);" +
        "border-radius: 6px; background: var(--background); color: var(--foreground);" +
        "font: inherit; cursor: pointer;";
      reset.addEventListener("click", () => {
        void Promise.resolve()
          .then(() => onReset())
          .catch(() => undefined);
      });
      parts.push(reset);
    }
    root.replaceChildren(...parts);
  };

  show(optionsOf(kind, given));
  parent.append(root);
  return {
    update: (changes: object) =>
      show({ ...options, ...optionsOf(kind, changes) }),
    destroy: () => root.remove(),
  };
}
