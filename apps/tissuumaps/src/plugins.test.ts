import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
// the store types only resolve with the Immer middleware's type augmentation,
// which the declarations of ./stores/app do not pull in
import type {} from "zustand/middleware/immer";

import type { Plugin } from "@tissuumaps/core";

import {
  loadPluginFromURL,
  pluginRegistry,
  startPluginRegistry,
} from "./plugins";
import { appStore } from "./stores/app";

/**
 * Creates a `data:` URL for an ES module with the given source, which vitest
 * imports like any other module
 */
function makeModuleUrl(source: string): string {
  return `data:text/javascript,${encodeURIComponent(source)}`;
}

describe("pluginRegistry", () => {
  let stopPluginRegistry: () => void;

  beforeEach(() => {
    stopPluginRegistry = startPluginRegistry();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    stopPluginRegistry();
    vi.restoreAllMocks();
  });

  it("sets a plugin up and adds it to the app store when registered", () => {
    const setup = vi.fn();
    pluginRegistry.registerPlugin({ id: "p", name: "P", setup });
    expect(setup).toHaveBeenCalledOnce();
    expect(appStore.getState().plugins.get("p")).toEqual({
      name: "P",
      mountable: false,
    });
  });

  it("does not register a plugin whose setup throws", () => {
    pluginRegistry.registerPlugin({
      id: "p",
      name: "P",
      setup: () => {
        throw new Error("setup failed");
      },
    });
    expect(appStore.getState().plugins.has("p")).toBe(false);
  });

  it("adds the container to the app store once mounted, and removes it when unmounted", () => {
    const unmount = vi.fn();
    const mount = vi.fn(() => unmount);
    pluginRegistry.registerPlugin({ id: "p", name: "P", mount });
    pluginRegistry.mountPlugin("p");
    const [container] = mount.mock.calls[0] as unknown as [HTMLElement];
    expect(appStore.getState().plugins.get("p")?.container).toBe(container);
    pluginRegistry.unmountPlugin("p");
    expect(unmount).toHaveBeenCalledOnce();
    expect(appStore.getState().plugins.get("p")?.container).toBeUndefined();
  });

  it("does not mount a mounted plugin again", () => {
    const mount = vi.fn();
    pluginRegistry.registerPlugin({ id: "p", name: "P", mount });
    pluginRegistry.mountPlugin("p");
    pluginRegistry.mountPlugin("p");
    expect(mount).toHaveBeenCalledOnce();
  });

  it("keeps a plugin whose mount throws registered, but unmounted", () => {
    pluginRegistry.registerPlugin({
      id: "p",
      name: "P",
      mount: () => {
        throw new Error("mount failed");
      },
    });
    pluginRegistry.mountPlugin("p");
    expect(appStore.getState().plugins.get("p")).toEqual({
      name: "P",
      mountable: true,
    });
  });

  it("unmounts a mounted plugin before tearing it down when unregistered", () => {
    const calls: string[] = [];
    pluginRegistry.registerPlugin({
      id: "p",
      name: "P",
      setup: () => () => calls.push("teardown"),
      mount: () => () => calls.push("unmount"),
    });
    pluginRegistry.mountPlugin("p");
    pluginRegistry.unregisterPlugin("p");
    expect(calls).toEqual(["unmount", "teardown"]);
    expect(appStore.getState().plugins.has("p")).toBe(false);
  });

  it("replaces a plugin registered again under the same ID", () => {
    const teardown = vi.fn();
    const plugin: Plugin = { id: "p", name: "P", setup: () => teardown };
    pluginRegistry.registerPlugin(plugin);
    pluginRegistry.registerPlugin({ ...plugin, name: "Q" });
    expect(teardown).toHaveBeenCalledOnce();
    expect(appStore.getState().plugins.get("p")?.name).toBe("Q");
  });
});

describe("loadPluginFromURL", () => {
  let stopPluginRegistry: () => void;

  beforeEach(() => {
    stopPluginRegistry = startPluginRegistry();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    stopPluginRegistry();
    vi.restoreAllMocks();
  });

  it("registers the default export, without mounting it", async () => {
    const url = makeModuleUrl(
      'export default { id: "loaded", name: "Loaded", mount: () => {} };',
    );
    await expect(loadPluginFromURL(url)).resolves.toBe("loaded");
    expect(appStore.getState().plugins.get("loaded")).toEqual({
      name: "Loaded",
      mountable: true,
    });
  });

  it("does not set up a plugin again that the module has registered itself", async () => {
    const url = makeModuleUrl(
      'const plugin = { id: "self", name: "Self", setup: () => { window.selfSetups = (window.selfSetups ?? 0) + 1; } }; export default plugin; window.tissuumaps.registerPlugin(plugin);',
    );
    await expect(loadPluginFromURL(url)).resolves.toBe("self");
    expect(window).toHaveProperty("selfSetups", 1);
  });

  it("sets a plugin up again only once it has been unregistered, when loading the same URL again", async () => {
    const url = makeModuleUrl(
      'export default { id: "again", name: "Again", setup: () => { window.againSetups = (window.againSetups ?? 0) + 1; } };',
    );
    await loadPluginFromURL(url);
    await loadPluginFromURL(url);
    expect(window).toHaveProperty("againSetups", 1);
    pluginRegistry.unregisterPlugin("again");
    await loadPluginFromURL(url);
    expect(window).toHaveProperty("againSetups", 2);
  });

  it("throws for a module without a default export", async () => {
    const url = makeModuleUrl("export const plugin = {};");
    await expect(loadPluginFromURL(url)).rejects.toThrow("no default export");
  });

  it("throws for a default export that is not a plugin", async () => {
    const url = makeModuleUrl('export default { id: "no-name" };');
    await expect(loadPluginFromURL(url)).rejects.toThrow("is not a plugin");
    expect(appStore.getState().plugins.has("no-name")).toBe(false);
  });

  it("throws for a plugin whose setup throws", async () => {
    const url = makeModuleUrl(
      'export default { id: "failing", name: "Failing", setup: () => { throw new Error("setup failed"); } };',
    );
    await expect(loadPluginFromURL(url)).rejects.toThrow(
      "Error during setup of plugin failing",
    );
    expect(appStore.getState().plugins.has("failing")).toBe(false);
  });

  it("throws for a module that cannot be run", async () => {
    const url = makeModuleUrl('throw new Error("module failed");');
    await expect(loadPluginFromURL(url)).rejects.toThrow("module failed");
  });
});
