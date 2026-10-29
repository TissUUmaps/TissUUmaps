import { type Persona, thinkingWords } from "./persona";
import type { Provider } from "./providers";
import type { ToolArguments } from "./types";

// All the colors are TissUUmaps' theme variables, which change with the light
// and the dark theme, so that the panel stays readable with both.

/** The shared look of the menus, the text fields and the buttons */
export const controlStyle =
  "padding: 4px 8px; border: 1px solid var(--input); border-radius: 6px;" +
  "background: var(--background); color: var(--foreground); font: inherit;";

/** The look of the small buttons, e.g. "Undo" and the suggestions */
export const smallButtonStyle = `${controlStyle}padding: 2px 8px; font-size: 12px;`;

/** The look of small, quieter text, e.g. the thinking word and "Made by" */
export const quietTextStyle =
  "color: var(--muted-foreground); font-size: 12px;";

/** The kinds of messages in the chat, each with its own look */
export type MessageKind =
  "user" | "assistant" | "tool" | "note" | "error" | "info";

/** The look of each kind of message */
export const messageStyles: Record<MessageKind, string> = {
  user: "background: var(--secondary); color: var(--secondary-foreground); border-color: var(--border);",
  assistant: "background: transparent; border-color: var(--border);",
  tool:
    "background: transparent; color: var(--muted-foreground); border-color: var(--border);" +
    "border-style: dashed; font-size: 12px;",
  note: "background: var(--accent); color: var(--accent-foreground); border-color: var(--ring);",
  error:
    "background: color-mix(in oklch, var(--destructive) 12%, transparent);" +
    "border-color: var(--destructive);",
  info: "background: transparent; border-color: var(--border);",
};

/** How long each word is shown while the model thinks, in milliseconds */
const thinkingWordTime = 2000;

/** The line at the bottom of the panel */
export const credits = "Made by Alessia Giustolisi";

/**
 * Creates a button
 *
 * @param text - The button text
 * @param onClick - The click action
 * @param style - The look of the button
 * @returns The button
 */
export function makeButton(
  text: string,
  onClick: () => void,
  style = controlStyle,
): HTMLButtonElement {
  const button = document.createElement("button");
  button.style.cssText = style;
  button.textContent = text;
  button.addEventListener("click", onClick);
  return button;
}

/**
 * Creates the avatar shown next to the assistant's name: the image of the
 * personality, or an empty round placeholder until there is one
 *
 * @param persona - The personality
 * @returns The element
 */
export function makeAvatar(persona: Persona): HTMLElement {
  const style =
    "width: 20px; height: 20px; border-radius: 50%; flex: none; object-fit: cover;";
  if (persona.avatar !== undefined) {
    const image = document.createElement("img");
    image.src = persona.avatar;
    // the name is next to it
    image.alt = "";
    image.style.cssText = style;
    return image;
  }
  const placeholder = document.createElement("span");
  placeholder.setAttribute("aria-hidden", "true");
  placeholder.style.cssText = `${style}background: var(--muted); border: 1px solid var(--border);`;
  return placeholder;
}

/**
 * Picks a word to show while the model thinks, at random
 *
 * @param previous - The word shown until now, which is not picked again
 * @returns The word with its translation
 */
function randomThinkingWord(
  previous?: (typeof thinkingWords)[number],
): (typeof thinkingWords)[number] {
  const others = thinkingWords.filter((entry) => entry !== previous);
  return others[Math.floor(Math.random() * others.length)] as {
    word: string;
    translation: string;
  };
}

/**
 * Makes a dot blink, unless the user asked the system for less motion or the
 * browser cannot animate
 *
 * @param dot - The dot
 * @param index - Its position, so that the dots blink one after the other
 */
function blink(dot: HTMLElement, index: number): void {
  const lessMotion =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (lessMotion || typeof dot.animate !== "function") {
    return;
  }
  dot.animate([{ opacity: 0.25 }, { opacity: 1 }, { opacity: 0.25 }], {
    duration: 1200,
    delay: index * 200,
    iterations: Infinity,
  });
}

/**
 * Creates the indicator shown while the model thinks: three blinking dots and
 * an Italian word that changes every two seconds, in random order. With the
 * mouse over the word, or the keyboard focus on it, its English translation
 * is shown and the word stays.
 *
 * @returns The element, and a function that stops the changes
 */
export function makeThinkingIndicator(): {
  element: HTMLElement;
  stop: () => void;
} {
  const element = document.createElement("div");
  element.style.cssText = `display: flex; align-items: center; gap: 6px; ${quietTextStyle}`;
  const dots = document.createElement("span");
  dots.setAttribute("aria-hidden", "true");
  for (let index = 0; index < 3; index++) {
    const dot = document.createElement("span");
    dot.textContent = "•";
    dot.style.margin = "0 1px";
    dots.append(dot);
    blink(dot, index);
  }
  const word = document.createElement("span");
  word.tabIndex = 0;
  word.style.cssText =
    "font-style: italic; cursor: help; border-bottom: 1px dotted var(--muted-foreground);";
  const translation = document.createElement("span");
  translation.setAttribute("role", "tooltip");
  translation.style.cssText =
    "position: absolute; left: calc(100% + 6px); top: 50%; transform: translateY(-50%);" +
    "display: none; z-index: 1; white-space: nowrap; padding: 2px 6px; font-size: 12px;" +
    "border: 1px solid var(--border); border-radius: 6px; background: var(--popover);" +
    "color: var(--popover-foreground); box-shadow: 0 1px 3px rgb(0 0 0 / 20%);";
  const wordBox = document.createElement("span");
  wordBox.style.position = "relative";
  wordBox.append(word, translation);
  element.append(dots, wordBox);

  let current = randomThinkingWord();
  let paused = false;
  function showWord(): void {
    word.textContent = current.word;
    translation.textContent = current.translation;
  }
  function pause(on: boolean): void {
    paused = on;
    translation.style.display = on ? "" : "none";
  }
  showWord();
  word.addEventListener("mouseenter", () => pause(true));
  word.addEventListener("mouseleave", () => pause(false));
  word.addEventListener("focus", () => pause(true));
  word.addEventListener("blur", () => pause(false));
  const timer = setInterval(() => {
    if (!paused) {
      current = randomThinkingWord(current);
      showWord();
    }
  }, thinkingWordTime);
  return { element, stop: () => clearInterval(timer) };
}

/** The icon of each tool, at the start of its folded row */
export const toolIcons: Record<string, string> = {
  set_visibility: "👁",
  show_only: "👁",
  set_opacity: "🌫",
  set_point_size: "⚫",
  color_by_column: "🎨",
  set_group_colors: "🎨",
  set_color: "🎨",
  show_only_groups: "🔍",
  show_all_groups: "🔍",
  call_action: "⚙️",
};

/**
 * The first line of a text, cut at 120 characters, for a folded row
 *
 * @param text - The text
 * @returns The line
 */
export function firstLine(text: string): string {
  const line = text.split("\n")[0] ?? "";
  return line.length > 120 ? `${line.slice(0, 119)}…` : line;
}

/**
 * Formats a tool call for the chat, e.g. "set_visibility(category: points,
 * visible: false)" or 'set_group_colors(colors: {"3":"red"})'
 *
 * @param name - The tool name
 * @param args - The arguments
 * @returns The text
 */
export function describeCall(name: string, args: ToolArguments): string {
  const values = Object.entries(args).map(([key, value]) => {
    if (Array.isArray(value)) {
      // objects in full, so that two different calls never read the same
      const items = value.map((item: unknown) =>
        typeof item === "string" ? item : JSON.stringify(item),
      );
      return `${key}: [${items.join(", ")}]`;
    }
    if (typeof value === "object" && value !== null) {
      return `${key}: ${JSON.stringify(value)}`;
    }
    return `${key}: ${String(value)}`;
  });
  return `${name}(${values.join(", ")})`;
}

/**
 * Writes an amount in dollars, with enough digits for fractions of a cent
 *
 * @param amount - The amount
 * @returns E.g. "$0.0004" or "$0.012"
 */
export function dollars(amount: number): string {
  return `$${amount.toFixed(amount < 0.01 ? 4 : 3)}`;
}

/**
 * Whether a service is reached through Ollama on the user's computer, which
 * needs no key but has to be running
 *
 * @param provider - The service
 * @returns Whether it is one of the Ollama services
 */
export function usesOllama(provider: Provider): boolean {
  return !provider.needsKey;
}

/**
 * The message when OpenAI or Anthropic cannot be reached
 *
 * @param provider - The service
 * @returns The message
 */
export function notReachable(provider: Provider): string {
  return `${provider.name} does not answer. Check the internet connection, then click "Check again".`;
}
