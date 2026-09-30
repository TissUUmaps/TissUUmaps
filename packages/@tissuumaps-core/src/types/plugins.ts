import type { AppStoreApi } from "./stores/app";
import type { DataStoreApi } from "./stores/data";
import type { ProjectStoreApi } from "./stores/project";
import type { SettingsStoreApi } from "./stores/settings";

/**
 * The stores provided by the application for a plugin to interact with
 */
export type PluginStores = {
  appStore: AppStoreApi;
  dataStore: DataStoreApi;
  projectStore: ProjectStoreApi;
  settingsStore: SettingsStoreApi;
};

/**
 * A plugin that can be registered with the application
 *
 * A plugin's lifecycle is `setup`, `mount`, `unmount`, `teardown`: registering
 * the plugin calls {@link setup}; mounting it, which happens when the user opens
 * it, calls {@link mount}; unmounting it, which happens when the user closes its
 * panel, calls the callback returned by `mount`; and unregistering it unmounts
 * it if it is mounted and then calls the callback returned by `setup`. A plugin
 * can be mounted and unmounted any number of times while it is registered. All
 * four steps are optional.
 */
export type Plugin = {
  /** The unique ID of the plugin */
  id: string;

  /** The human-readable name of the plugin */
  name: string;

  /**
   * Sets up the plugin and registers any necessary event listeners or other resources
   *
   * Called once when the plugin is registered. A plugin whose `setup` throws is
   * not registered, and the teardown callback is not called: releasing whatever
   * `setup` had already acquired before it throws is `setup`'s own
   * responsibility.
   *
   * A plugin that only contributes a user interface does not need a `setup`:
   * its {@link mount} receives the same stores.
   *
   * @param stores - The stores provided by the application for the plugin to interact with
   * @returns A teardown callback that unregisters the event listeners and
   * releases the other resources again, if any. It is called last when the
   * plugin is unregistered, after the user interface has been unmounted, and
   * only for a plugin that was registered successfully.
   */
  setup?: (stores: PluginStores) => (() => void) | void;

  /**
   * Mounts the plugin's user interface, if any
   *
   * A plugin with a `mount` can be opened by the user, which mounts it and shows
   * it as a panel titled with the plugin's {@link name}. Closing the panel
   * unmounts the plugin again, but keeps it registered, so that it can be opened
   * again later.
   *
   * Called each time the plugin is mounted, with a fresh, empty element owned by
   * the application. The element is not part of the document yet: the
   * application attaches it to the panel once that is shown, and keeps the user
   * interface mounted while the panel is hidden behind other tabs or moved.
   * A plugin whose `mount` throws is not mounted, but stays registered.
   *
   * The container is discarded when the plugin is unmounted, so the returned
   * callback only has to release what is not plain DOM, such as store
   * subscriptions, listeners on `window`, or a React root.
   *
   * @param container - The element to mount the user interface into
   * @param stores - The stores provided by the application for the plugin to interact with
   * @returns An unmount callback that unmounts the user interface again, if
   * any. It is called when the plugin is unmounted, which also happens first
   * when a mounted plugin is unregistered, before the teardown callback
   * returned by {@link setup}.
   */
  mount?: (container: HTMLElement, stores: PluginStores) => (() => void) | void;
};

/**
 * The registry through which the application registers, mounts, unmounts and
 * unregisters plugins
 */
export type PluginRegistry = {
  /**
   * Registers a plugin, calling its `setup`, if it has one
   *
   * The plugin is not mounted: that is up to the user, or to
   * {@link mountPlugin}. Registering a plugin whose ID is already in use
   * unregisters the previously registered plugin first.
   *
   * @param plugin - The plugin to register
   */
  registerPlugin: (plugin: Plugin) => void;

  /**
   * Mounts a registered plugin, calling its `mount` and showing its panel
   *
   * Does nothing if the plugin is not registered, has no `mount`, or is already
   * mounted.
   *
   * @param pluginId - The ID of the plugin to mount
   */
  mountPlugin: (pluginId: string) => void;

  /**
   * Unmounts a mounted plugin, removing its panel and calling the unmount
   * callback returned by its `mount`
   *
   * The plugin stays registered. Does nothing if the plugin is not mounted.
   *
   * @param pluginId - The ID of the plugin to unmount
   */
  unmountPlugin: (pluginId: string) => void;

  /**
   * Unregisters a registered plugin, unmounting it if it is mounted and then
   * calling the teardown callback returned by its `setup`
   *
   * @param pluginId - The ID of the plugin to unregister
   */
  unregisterPlugin: (pluginId: string) => void;
};
