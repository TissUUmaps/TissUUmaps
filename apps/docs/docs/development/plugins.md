---
sidebar_position: 8
---

# Plugins

A plugin is an ES module that exports the plugin as its default export _and_
registers it through the plugin registry, which TissUUmaps exposes as
`window.tissuumaps` once the application has started up:

```javascript
const plugin = {
  id: "my-plugin",
  name: "My plugin",
  setup: ({ appStore, dataStore, projectStore, settingsStore }) => {
    return () => {}; // teardown
  },
};

export default plugin;

if (window.tissuumaps !== undefined) {
  window.tissuumaps.registerPlugin(plugin); // startup already finished
} else {
  window.addEventListener(
    "tissuumaps-loaded",
    () => window.tissuumaps.registerPlugin(plugin),
    { once: true },
  );
}
```

Written this way, the same file works however it gets into TissUUmaps: the user
can load it through the plugins menu, which uses the default export (see
[Loading plugins from the plugins menu](#loading-plugins-from-the-plugins-menu)),
and a deployment can include it in the page, where it registers itself:

```html
<script type="module" src="my-plugin.js"></script>
```

Because of the `export`, a plugin included in the page has to be loaded with
`type="module"`; a classic `<script>` fails with a syntax error.

A plugin can be registered at any time, but `window.tissuumaps` only exists once
the application has started up. TissUUmaps signals this by dispatching a
`tissuumaps-loaded` event on `window` at the end of its startup — a plain event
that is not replayed, so a listener added afterwards never fires. Whether a
plugin runs before or after startup usually cannot be guaranteed: module scripts
run in document order, so it depends on where the plugin's `<script>` is placed
relative to the application's, and an `async` or dynamically imported module
races it. Checking `window.tissuumaps` first and listening for
`tissuumaps-loaded` otherwise, as above, is correct either way.

Note that the project is not yet loaded when `tissuumaps-loaded` fires — loading
is only started during startup. A plugin that depends on project contents should
subscribe to `projectStore` in `setup` instead of reading it once.

A plugin is unregistered again — unmounting it if it is mounted, and tearing it
down — using `window.tissuumaps.unregisterPlugin(pluginId)`.

For TypeScript, the `Plugin`, `PluginRegistry` and `PluginStores` types are
exported from `@tissuumaps/core`, and `window.tissuumaps` is typed as
`PluginRegistry | undefined`.

The plugins shipped with TissUUmaps in `@tissuumaps/plugins` are the exception:
the application registers them itself on startup, so they only need to export the
plugin.

## Loading plugins from the plugins menu

The user can load a plugin through the plugins menu in the tab bar: _Load plugin
from file…_ picks a local file, and _Load plugin from URL…_ asks for a URL,
absolute or relative to the page. TissUUmaps runs the module, registers its
default export, which sets the plugin up, and opens the plugin's panel right away
if it has a `mount`. Loading a module without a default export fails, even if
the module registers a plugin itself when it runs — that plugin then stays
registered, but is not opened.

- A module loaded from a URL on another origin has to be served with
  [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS) headers,
  as common public hosts such as GitHub, jsDelivr or unpkg do.
- A module loaded from a local file has to be a single file, bundled if need be:
  its relative imports cannot be resolved, and neither can paths relative to
  `import.meta.url`. Imports of absolute URLs work.
- A plugin that registered itself when its module ran is not registered again,
  so its `setup` is only called once.
- Loading the same URL again does not run the module again, since the browser
  caches modules by URL. If the plugin is still registered, it is only opened
  again; if it has been unregistered in the meantime, it is registered again.
- Loaded plugins are not remembered: after reloading the page, they have to be
  loaded again.

:::warning

A plugin runs with full access to TissUUmaps and the data it has opened.
TissUUmaps asks for confirmation before loading one, but cannot check what it
does: only load plugins from sources you trust.

:::

## Plugin properties

- `id` (required): a unique identifier for the plugin. Registering a plugin whose
  `id` is already in use unregisters the previous plugin first.
- `name` (required): a human-readable name for the plugin.
- `setup` (optional): called once, immediately upon registration, with a reference
  to each of the application's Zustand stores (see below). It may return a
  teardown callback, see [Plugin lifecycle](#plugin-lifecycle). Errors thrown by
  `setup` are caught and logged; they do not abort application startup. The
  plugin is then not registered, and no teardown happens - a teardown never has
  to cope with a half-initialized plugin. A `setup` that can fail part-way
  through is responsible for releasing what it had already set up before it
  rethrows. A plugin that only adds a user interface does not need a `setup`,
  because its `mount` receives the same stores.
- `mount` (optional): mounts a user interface panel when the user opens the
  plugin, and may return an unmount callback, see
  [User interface plugins](#user-interface-plugins).

## Plugin lifecycle

A plugin goes through four steps, all of them optional:

1. `setup` is called once, when the plugin is registered.
2. `mount` is called when the plugin is mounted, i.e. when the user opens it
   from the plugins menu, or through `window.tissuumaps.mountPlugin(pluginId)`.
   Registering a plugin does not mount it.
3. The unmount callback returned by `mount` is called when the plugin is
   unmounted, i.e. when the user closes its panel, or through
   `window.tissuumaps.unmountPlugin(pluginId)`. The plugin stays registered and
   set up, and can be mounted again later, which calls `mount` again.
4. The teardown callback returned by `setup` is called last, when the plugin is
   unregistered, once a mounted user interface has been unmounted.

Unregistering happens through `window.tissuumaps.unregisterPlugin(pluginId)`, by
registering another plugin with the same `id`, and when the plugin registry is
stopped (currently only on hot module replacement during development, not on
page unload). Closing a plugin's panel does not unregister it. The unmount and
teardown callbacks are only ever called for a plugin whose `setup` succeeded;
errors they throw are caught and logged. If `mount` throws, the error is logged
and the plugin is not mounted, but it stays registered.

## Stores

`setup` and `mount` receive the application's four Zustand stores:

| Store           | Contents                                                                                              |
| --------------- | ----------------------------------------------------------------------------------------------------- |
| `appStore`      | Application state: workspace, whether a project is open, interaction mode, data providers, plugins    |
| `dataStore`     | Data references (`DataRef`) for the loaded data of each project object                                |
| `projectStore`  | The currently loaded project (name, layers, images, labels, points, shapes, tables, maps) and its URL |
| `settingsStore` | User settings that are persisted across sessions                                                      |

Each store is a Zustand store API. Using `appStore` as an example, a plugin can
read the current value of `myProperty` using `appStore.getState().myProperty`,
call the action `myAction` using `appStore.getState().myAction(...)`, and observe
changes using `appStore.subscribe((state, prevState) => {})`.

The stores use the [Immer](https://immerjs.github.io/immer/) middleware, so
`setState` takes a recipe that mutates a draft:

```javascript
appStore.setState((draft) => {
  draft.myProperty = myValue;
});
```

:::caution

The recipe must not return a value. Writing `appStore.setState((draft) => draft.myMap.set(k, v))`
returns the map and makes Immer reject the update — always use a block body.

:::

`dataStore` is derived state: its contents are reconciled from `appStore` and
`projectStore` by the application. Plugins should treat it as read-only and drive
data loading by changing the project instead, for example
`projectStore.getState().updateTable(tableId, { dataSource })`. Likewise,
`appStore`'s `plugins` is written by the registry alone.

A plugin shows the user the settings of a project object, e.g. one it created,
by calling `appStore.getState().focusObject({ kind: "images", id })`: the panel
of the object's collection is brought to the front and the object's settings are
expanded, but not scrolled into view. `kind` is one of `images`, `labels`,
`points` and `shapes`.

## User interface plugins

A plugin adds a panel to the TissUUmaps user interface by declaring a `mount`.
Every such plugin is listed by its `name` in the plugins menu, the kebab button
in the tab bar next to the dark mode toggle. Picking it there mounts the plugin,
and its panel appears as a tab titled with the plugin's `name`, next to the
built-in Project, Images, Labels, Points, Shapes and Tables panels:

```javascript
const plugin = {
  id: "my-plugin",
  name: "My plugin",
  mount: (container, { projectStore }) => {
    const paragraph = document.createElement("p");
    const update = (state) => {
      paragraph.textContent = `${state.images.length} images`;
    };
    update(projectStore.getState());
    container.append(paragraph);
    return projectStore.subscribe(update); // unmount
  },
};
// exported and registered as shown above
```

Because `mount` receives the stores itself, a plugin that only adds a user
interface does not need a `setup`.

`mount` is called with an empty `HTMLElement` owned by TissUUmaps, into which the
plugin renders its user interface, and with the same four stores that `setup`
receives. It may return a callback that unmounts that user interface again.
The container is discarded when the plugin is unmounted, so the callback only has
to release what is not plain DOM, such as store subscriptions, listeners on `window`,
or a React root.

### Panel lifetime

The panel is shown for exactly as long as the plugin is mounted:

- `mount` is called when the user picks the plugin in the plugins menu, or when
  `window.tissuumaps.mountPlugin(pluginId)` is called, and the panel is made
  active. Picking a plugin that is already mounted only activates its panel.
- The container is not part of the document when `mount` is called: TissUUmaps
  attaches it to the panel once the panel is shown. Measure the container's size
  with a `ResizeObserver` rather than once in `mount`.
- The user interface stays mounted while other tabs of its group are active, and
  when the panel is dragged elsewhere — neither remounts it.
- Closing the panel unmounts the plugin, calling the callback returned by
  `mount`, but keeps it registered: whatever `setup` set up stays in place, and
  the plugin can be opened from the plugins menu again, which calls `mount` again
  with a fresh container.
- Unregistering a mounted plugin calls the callback returned by `mount` first and
  the callback returned by `setup` afterwards, so an unmount callback may still
  rely on whatever `setup` set up.

### Bringing your own framework

`mount` is a plain DOM contract, so a plugin can use whichever framework it
likes — or none at all. With React, for example:

```javascript
mount: (container, stores) => {
  const root = ReactDOM.createRoot(container);
  root.render(React.createElement(MyPanel, { stores }));
  return () => root.unmount();
};
```
