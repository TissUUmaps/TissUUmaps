# @tissuumaps/core

## 0.1.0-beta.1

### Minor Changes

- [#259](https://github.com/TissUUmaps/TissUUmaps/pull/259) [`62b79c2`](https://github.com/TissUUmaps/TissUUmaps/commit/62b79c2218ea5419972553b43cc217084d9a9d23) Thanks [@cavenel](https://github.com/cavenel)! - Plugins can show the settings of an image, labels, points or shapes: `appStore.getState().showImageSettings(imageId)` brings the Images panel to the front and expands the image's settings, and `showLabelsSettings`, `showPointsSettings` and `showShapesSettings` do the same for the other panels. Adds these actions and the `imageSettingsRequest`, `labelsSettingsRequest`, `pointsSettingsRequest` and `shapesSettingsRequest` app store state.

- [#253](https://github.com/TissUUmaps/TissUUmaps/pull/253) [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62) Thanks [@jwindhager](https://github.com/jwindhager)! - `TableData.loadUniqueValueCounts` and `TableData.loadValueRange` are now optional: table data only implements them if it can determine the counts or the range more cheaply than by loading the column (e.g. from file metadata). Callers use the new `TableUtils.loadUniqueValueCounts` and `TableUtils.loadValueRange`, which fall back to the loaded column values (see `TableUtils.computeValueRange`). CSV tables no longer implement either method, and Parquet and hierarchical (AnnData) tables no longer implement `loadUniqueValueCounts`. Unless the table data determines them itself, the app now computes unique value counts and value ranges from the column values it has already loaded, instead of loading the column again.

- [#253](https://github.com/TissUUmaps/TissUUmaps/pull/253) [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62) Thanks [@jwindhager](https://github.com/jwindhager)! - Group-by palettes (colors and markers without a map) now assign their values by group position, in the order the groups first appear in the table column, instead of by hashing the group names; random colors are still hashed. Projects that group by a palette show different colors or markers than before. `ConfigUtils.createGroupValueGetter` takes the ordered groups alongside the palette, and the new `TableUtils.loadGroupCounts` lists them.

- [#252](https://github.com/TissUUmaps/TissUUmaps/pull/252) [`cb49f75`](https://github.com/TissUUmaps/TissUUmaps/commit/cb49f75bb87c17d06198562d53e4ab17cd89aaf5) Thanks [@jwindhager](https://github.com/jwindhager)! - Plugins are no longer mounted when registered: a menu in the tab bar lists the registered plugins and mounts the one picked, and closing a plugin's panel only unmounts it. The menu can also load third-party plugins, ES modules whose default export is the plugin, from a local file or a URL, absolute or relative to the page. Third-party plugins should both default-export the plugin and register it through `window.tissuumaps`, so that the same file can be loaded through the menu or included in the page. Adds `PluginRegistry.mountPlugin`/`unmountPlugin`.

## 0.1.0-beta.0

### Minor Changes

- [#230](https://github.com/TissUUmaps/TissUUmaps/pull/230) [`3e2feac`](https://github.com/TissUUmaps/TissUUmaps/commit/3e2feaccf033d6ebf785e78e129671cd7683bd2e) Thanks [@jwindhager](https://github.com/jwindhager)! - Initial beta release.
