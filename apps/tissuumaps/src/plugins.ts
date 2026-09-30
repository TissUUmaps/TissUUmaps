import { castDraft } from "immer";

import type { Plugin, PluginRegistry, PluginStores } from "@tissuumaps/core";

import { appStore } from "./stores/app";
import { dataStore } from "./stores/data";
import { projectStore } from "./stores/project";
import { settingsStore } from "./stores/settings";

declare global {
  interface Window {
    /** The plugin registry, available once the application has started up */
    tissuumaps?: PluginRegistry;
  }
}

/**
 * The stores handed to a plugin, both when it is set up and when it is mounted
 */
const pluginStores: PluginStores = {
  appStore,
  dataStore,
  projectStore,
  settingsStore,
};

/**
 * The registered plugins, by plugin ID, each with the teardown callback
 * returned by its `setup` and, while it is mounted, the unmount callback
 * returned by its `mount`
 *
 * Kept outside of the app store, whose state holds the plugins' names and
 * containers, since the plugin objects and callbacks are the registry's
 * business alone.
 */
const pluginRegistrations = new Map<
  string,
  {
    plugin: Plugin;
    teardown: (() => void) | void;
    mounted: boolean;
    unmount: (() => void) | void;
  }
>();

/**
 * The plugin registry, which owns the plugin lifecycle
 *
 * Registering a plugin calls its `setup` and adds the plugin to the app store
 * once that has succeeded, as its name and whether it can be mounted. Mounting
 * it calls its `mount` with a fresh element, which is added to the app store
 * once `mount` has succeeded, so that the plugin's panel is shown; unmounting
 * it removes the element from the app store first and then calls the unmount
 * callback returned by `mount`. Unregistering a plugin removes it from the app
 * store, unmounts it if it is mounted, and then calls the teardown callback
 * returned by `setup`. The registry is the only writer of the app store's
 * `plugins`.
 *
 * The plugin object itself is not put into the app store: the app store only
 * holds what the user interface renders, so that Immer does not freeze
 * anything the plugin owns.
 *
 * Errors thrown by a plugin are caught and logged, so that a failing plugin
 * does not take the application down with it. A plugin whose `setup` throws is
 * not registered, and one whose `mount` throws is not mounted, but stays
 * registered.
 */
export const pluginRegistry: PluginRegistry = {
  registerPlugin: (plugin) => {
    pluginRegistry.unregisterPlugin(plugin.id);
    let teardown: (() => void) | void = undefined;
    if (plugin.setup !== undefined) {
      try {
        teardown = plugin.setup(pluginStores);
      } catch (setupError) {
        console.error(`Error during setup of plugin ${plugin.id}:`, setupError);
        return;
      }
    }
    pluginRegistrations.set(plugin.id, {
      plugin,
      teardown,
      mounted: false,
      unmount: undefined,
    });
    appStore.setState((draft) => {
      draft.plugins.set(plugin.id, {
        name: plugin.name,
        mountable: plugin.mount !== undefined,
      });
    });
  },
  mountPlugin: (pluginId) => {
    const registration = pluginRegistrations.get(pluginId);
    if (
      registration === undefined ||
      registration.plugin.mount === undefined ||
      registration.mounted
    ) {
      return;
    }
    const container = document.createElement("div");
    try {
      registration.unmount = registration.plugin.mount(container, pluginStores);
    } catch (mountError) {
      console.error(`Error during mount of plugin ${pluginId}:`, mountError);
      return;
    }
    registration.mounted = true;
    appStore.setState((draft) => {
      const draftPlugin = draft.plugins.get(pluginId);
      if (draftPlugin !== undefined) {
        // the element is opaque to Immer, which its draft type cannot express
        draftPlugin.container = castDraft(container);
      }
    });
  },
  unmountPlugin: (pluginId) => {
    const registration = pluginRegistrations.get(pluginId);
    if (registration === undefined || !registration.mounted) {
      return;
    }
    const unmount = registration.unmount;
    registration.mounted = false;
    registration.unmount = undefined;
    appStore.setState((draft) => {
      const draftPlugin = draft.plugins.get(pluginId);
      if (draftPlugin !== undefined) {
        delete draftPlugin.container;
      }
    });
    try {
      unmount?.();
    } catch (unmountError) {
      console.error(
        `Error during unmount of plugin ${pluginId}:`,
        unmountError,
      );
    }
  },
  unregisterPlugin: (pluginId) => {
    const registration = pluginRegistrations.get(pluginId);
    if (registration !== undefined) {
      pluginRegistry.unmountPlugin(pluginId);
      pluginRegistrations.delete(pluginId);
      appStore.setState((draft) => {
        draft.plugins.delete(pluginId);
      });
      try {
        registration.teardown?.();
      } catch (teardownError) {
        console.error(
          `Error during teardown of plugin ${pluginId}:`,
          teardownError,
        );
      }
    }
  },
};

/**
 * The plugins shipped with TissUUmaps in `@tissuumaps/plugins`, registered by
 * {@link enableBuiltInPlugins}
 */
const builtInPlugins: Plugin[] = [];

/**
 * Exposes the {@link pluginRegistry} to plugins as `window.tissuumaps`
 *
 * @returns A callback that removes the registry from `window` again, unless it
 * has been replaced in the meantime, and unregisters all plugins that are still
 * registered
 */
export function startPluginRegistry(): () => void {
  window.tissuumaps = pluginRegistry;
  return () => {
    if (window.tissuumaps === pluginRegistry) {
      delete window.tissuumaps;
    }
    for (const pluginId of [...pluginRegistrations.keys()]) {
      pluginRegistry.unregisterPlugin(pluginId);
    }
  };
}

/**
 * Registers the plugins shipped with TissUUmaps with the {@link pluginRegistry}
 *
 * Called once during application startup, after the plugin registry has been
 * started. The plugins are set up, but not mounted.
 */
export function enableBuiltInPlugins(): void {
  for (const plugin of builtInPlugins) {
    pluginRegistry.registerPlugin(plugin);
  }
}

/**
 * Loads a third-party plugin from a local ES module file, and registers it
 * with the {@link pluginRegistry}, see {@link loadPluginFromURL}
 *
 * The module is loaded from a `blob:` URL, so it has to be a single file: its
 * relative imports cannot be resolved.
 *
 * @param file - The module file
 * @returns The ID of the registered plugin
 * @throws Error if the module cannot be loaded or run, if it has no default
 * export or its default export is not a plugin, or if the plugin's `setup`
 * throws
 */
export async function loadPluginFromFile(file: File): Promise<string> {
  // a module is only run with a JavaScript MIME type, which a file may lack
  const blob = new Blob([file], { type: "text/javascript" });
  const url = URL.createObjectURL(blob);
  try {
    return await loadPluginFromURL(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Loads a third-party plugin from an ES module, and registers it with the
 * {@link pluginRegistry}
 *
 * The module's default export has to be a plugin, which is registered, and
 * thereby set up, but not mounted. A plugin that is already registered as that
 * same object, because the module registered it itself when it ran, or because
 * the same URL has been loaded before, is not registered again.
 *
 * A module on another origin has to be served with CORS. Modules are cached by
 * URL: loading the same URL again does not run the module again.
 *
 * @param pluginUrl - The URL of the module, absolute or relative to the
 * document base URL
 * @returns The ID of the registered plugin
 * @throws Error if `pluginUrl` is not a valid URL, if the module cannot be
 * loaded or run, if it has no default export or its default export is not a
 * plugin, or if the plugin's `setup` throws
 */
export async function loadPluginFromURL(pluginUrl: string): Promise<string> {
  // import() resolves relative URLs against the importing module, not the page
  const absolutePluginUrl = new URL(pluginUrl, document.baseURI).href;
  const exports: unknown = await import(/* @vite-ignore */ absolutePluginUrl);
  if (
    exports === null ||
    typeof exports !== "object" ||
    !("default" in exports)
  ) {
    throw new Error(`${pluginUrl} has no default export`);
  }
  const plugin = exports.default;
  if (!isPlugin(plugin)) {
    throw new Error(`The default export of ${pluginUrl} is not a plugin`);
  }
  // not set up again if the module has registered it itself, or on reloading
  if (pluginRegistrations.get(plugin.id)?.plugin !== plugin) {
    pluginRegistry.registerPlugin(plugin);
  }
  if (!pluginRegistrations.has(plugin.id)) {
    throw new Error(
      `Error during setup of plugin ${plugin.id}, see the browser console for details`,
    );
  }
  return plugin.id;
}

/**
 * Checks whether a value has the shape of a plugin
 *
 * Only the types of the properties are checked, not the signatures of the
 * functions, which cannot be inspected at runtime.
 *
 * @param value - The value to check, such as a module's default export
 * @returns Whether the value is a plugin
 */
function isPlugin(value: unknown): value is Plugin {
  return (
    typeof value === "object" &&
    value !== null &&
    "id" in value &&
    typeof value.id === "string" &&
    "name" in value &&
    typeof value.name === "string" &&
    (!("setup" in value) ||
      value.setup === undefined ||
      typeof value.setup === "function") &&
    (!("mount" in value) ||
      value.mount === undefined ||
      typeof value.mount === "function")
  );
}
