/** The start of the localStorage keys under which the panel remembers settings */
const storagePrefix = "tissuumaps.ai-chat.";

/**
 * Reads a setting remembered in an earlier session, e.g. the chosen service
 *
 * @param name - The name of the setting, e.g. "provider" or "model.openai"
 * @returns The value, or `null` if none was remembered or the browser does
 * not allow storage (e.g. in a private window)
 */
export function loadSetting(name: string): string | null {
  try {
    return window.localStorage.getItem(storagePrefix + name);
  } catch {
    return null;
  }
}

/**
 * Remembers a setting for the next session, or forgets it, if the browser
 * allows it
 *
 * @param name - The name of the setting
 * @param value - The value, or `null` to forget it
 */
export function saveSetting(name: string, value: string | null): void {
  try {
    if (value === null) {
      window.localStorage.removeItem(storagePrefix + name);
    } else {
      window.localStorage.setItem(storagePrefix + name, value);
    }
  } catch {
    // not remembered: the default is used next time
  }
}
