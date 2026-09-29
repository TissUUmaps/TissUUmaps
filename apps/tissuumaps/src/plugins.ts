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
