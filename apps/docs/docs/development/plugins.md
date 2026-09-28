---
sidebar_position: 8
---

# Plugins

Once the application has started up, TissUUmaps exposes its plugin registry as
`window.tissuumaps`. Plugins register themselves through the registry:

```javascript
window.tissuumaps.registerPlugin({
  id: "my-plugin",
  name: "My plugin",
  setup: ({ appStore, dataStore, projectStore, settingsStore }) => {
    return () => {}; // teardown
  },
});
```

A plugin can be registered at any time, but `window.tissuumaps` only exists once
the application has started up. TissUUmaps signals this by dispatching a
`tissuumaps-loaded` event on `window` at the end of its startup — a plain event
that is not replayed, so a listener added afterwards never fires.

Whether a plugin script runs before or after startup usually cannot be
guaranteed: a classic `<script>` in `index.html` runs before the application's
deferred `<script type="module">`, but an `async` or dynamically imported script
races it. Use the combined pattern, which is correct either way:

```javascript
function registerMyPlugin() {
  window.tissuumaps.registerPlugin({
    id: "my-plugin",
    name: "My plugin",
    setup: ({ appStore, dataStore, projectStore, settingsStore }) => {
      return () => {}; // teardown
    },
  });
}

if (window.tissuumaps !== undefined) {
  registerMyPlugin(); // startup already finished
} else {
  window.addEventListener("tissuumaps-loaded", registerMyPlugin, {
    once: true,
  });
}
```

Note that the project is not yet loaded when `tissuumaps-loaded` fires — loading
is only started during startup. A plugin that depends on project contents should
subscribe to `projectStore` in `setup` instead of reading it once.

A plugin is unregistered again — unmounting and tearing it down — using
`window.tissuumaps.unregisterPlugin(pluginId)`.

For TypeScript, the `Plugin`, `PluginRegistry` and `PluginStores` types are
exported from `@tissuumaps/core`, and `window.tissuumaps` is typed as
`PluginRegistry | undefined`.

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
- `mount` (optional): mounts a user interface panel shown for as long as the
  plugin is registered, and may return an unmount callback, see
  [User interface plugins](#user-interface-plugins).

## Plugin lifecycle

A plugin goes through four steps, all of them optional:

1. `setup` is called when the plugin is registered.
2. `mount` is called once `setup` has returned successfully, so anything `setup`
   prepares is available to it.
3. The unmount callback returned by `mount` is called when the plugin is
   unregistered.
4. The teardown callback returned by `setup` is called last, once the user
   interface has been unmounted.

Unregistering happens through `window.tissuumaps.unregisterPlugin(pluginId)`, by
closing the plugin's panel, by registering another plugin with the same `id`, and
when the plugin registry is stopped (currently only on hot module replacement
during development, not on page unload). The unmount and teardown callbacks are
only ever called for a plugin whose `setup` succeeded; errors they throw are
caught and logged. If `mount` throws, the plugin is not registered either, but
since `setup` did succeed, its teardown callback _is_ called.

## Stores

`setup` and `mount` receive the application's four Zustand stores:

| Store           | Contents                                                                                              |
| --------------- | ----------------------------------------------------------------------------------------------------- |
| `appStore`      | Application state: workspace, start page, interaction mode, data providers, plugins                   |
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

## User interface plugins

A plugin adds a panel to the TissUUmaps user interface by declaring a `mount`.
The panel appears as a tab titled with the plugin's `name`, next to the built-in
Project, Images, Labels, Points, Shapes and Tables panels:

```javascript
window.tissuumaps.registerPlugin({
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
});
```

Because `mount` receives the stores itself, a plugin that only adds a user
interface does not need a `setup`.

`mount` is called with an empty `HTMLElement` owned by TissUUmaps, into which the
plugin renders its user interface, and with the same four stores that `setup`
receives. It may return a callback that unmounts that user interface again.
The container is discarded together with the plugin, so the callback only has to
release what is not plain DOM, such as store subscriptions, listeners on `window`,
or a React root.

### Panel lifetime

The panel is shown for exactly as long as the plugin is registered:

- `mount` is called immediately upon registration, right after `setup`. The
  container is not part of the document at that point: TissUUmaps attaches it to
  the panel once the panel is shown. Measure the container's size with a
  `ResizeObserver` rather than once in `mount`.
- The user interface stays mounted while other tabs of its group are active, and
  when the panel is dragged elsewhere — neither remounts it.
- Unregistering the plugin calls the callback returned by `mount` first and the
  callback returned by `setup` afterwards, so an unmount callback may still rely
  on whatever `setup` set up. The container is discarded afterwards; registering
  the plugin again gives it a fresh one.

:::caution

Closing the panel unregisters the plugin. There is no user interface for adding a
plugin back, so a closed plugin has to be registered again — by reloading the
page, or by calling `registerPlugin` again.

:::

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
