import { runRequest } from "./browser-storage";
import type { ChatPluginEntry, PluginStorage } from "./chat-plugins";

// The plugins made by the chat that the user keeps ("Remember") are kept in
// IndexedDB, by id, next to the skills (see browser-storage.ts). Only their
// code and what the chat knows about them are kept: their code runs again
// only once the user says yes.

/**
 * Reads the plugins the browser remembers
 *
 * @returns The plugins; none if the browser has no IndexedDB, since then it
 * could never remember one
 * @throws If the database cannot be read
 */
function loadStoredPlugins(): Promise<ChatPluginEntry[]> {
  if (typeof indexedDB === "undefined") {
    return Promise.resolve([]);
  }
  return runRequest(
    "plugins",
    "readonly",
    (store) => store.getAll() as IDBRequest<ChatPluginEntry[]>,
  );
}

/** Where the browser remembers the plugins made by the chat */
export const pluginStorage: PluginStorage = {
  load: loadStoredPlugins,
  save: async (entry) => {
    await runRequest("plugins", "readwrite", (store) => store.put(entry));
  },
  remove: async (id) => {
    await runRequest("plugins", "readwrite", (store) => store.delete(id));
  },
};
