# @tissuumaps/core

## 0.1.0-beta.4

### Minor Changes

- [#394](https://github.com/TissUUmaps/TissUUmaps/pull/394) [`3503273`](https://github.com/TissUUmaps/TissUUmaps/commit/3503273f101dda019c8c2f170d3297f93e474b86) Thanks [@cavenel](https://github.com/cavenel)! - Either end of a color range can be `null`, which stands for the minimum or maximum value of the table column.

- [#314](https://github.com/TissUUmaps/TissUUmaps/pull/314) [`e369291`](https://github.com/TissUUmaps/TissUUmaps/commit/e36929107a23b1881a99d80536004daaf88b3bd3) Thanks [@cavenel](https://github.com/cavenel)! - Projects get `colorbarsVisibility`, set with the `setColorbarsVisibility` store action.

- [#418](https://github.com/TissUUmaps/TissUUmaps/pull/418) [`b774531`](https://github.com/TissUUmaps/TissUUmaps/commit/b7745316a23f68c84d57d8cf51f2b914258f74ec) Thanks [@cavenel](https://github.com/cavenel)! - `ProjectUtils` gets `mayBeVisible`, which returns whether an object and its layer are visible.

- [#402](https://github.com/TissUUmaps/TissUUmaps/pull/402) [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5) Thanks [@cavenel](https://github.com/cavenel)! - Opacity configs that read from a table column take an optional `range`, like color configs.

- [#420](https://github.com/TissUUmaps/TissUUmaps/pull/420) [`cd2e922`](https://github.com/TissUUmaps/TissUUmaps/commit/cd2e922bdf98a4713c6a28940dbc31cf6d53fb6b) Thanks [@cavenel](https://github.com/cavenel)! - Add `RawPoints.pointOrder`, which draws points in ascending order of a table column's values, and `MathUtils.computeSortOrder` and `MathUtils.toSortKey`, which compute the indices that visit values in ascending order.

- [#403](https://github.com/TissUUmaps/TissUUmaps/pull/403) [`ddd7fac`](https://github.com/TissUUmaps/TissUUmaps/commit/ddd7fac9046e0226a7225aa7621ad1a759ffa35a) Thanks [@jwindhager](https://github.com/jwindhager)! - Add the `ProjectProvider` interface for loading projects from, and writing them to, project file formats, including the file types a project provider loads, and a project provider registry in the app store. `SourceUtils.createID` creates unique IDs named after a source, and `SourceUtils.rebaseSource` resolves project-relative sources against their project, without fallbacks.

- [#239](https://github.com/TissUUmaps/TissUUmaps/pull/239) [`6464a42`](https://github.com/TissUUmaps/TissUUmaps/commit/6464a4217f7cd3636a8040dbf2f5f55151b180b2) Thanks [@cavenel](https://github.com/cavenel)! - `ProjectStoreState` keeps the project file in the workspace that the open project is saved to, with its project provider (`saveTarget`), and the project as last loaded or saved (`savedProject`).

- [#411](https://github.com/TissUUmaps/TissUUmaps/pull/411) [`575da90`](https://github.com/TissUUmaps/TissUUmaps/commit/575da90af95d3d17fbf8a8c2bd64e43204a2bc8a) Thanks [@cavenel](https://github.com/cavenel)! - Table data can group its columns into sections (`TableData.getColumnSections`, searched with the `sectionId` option of `suggestColumnQueries`, with `describeColumn` and `mapColumn`) and declare column types (`TableData.getColumnType`, with the `TableUtils.loadColumnType` fallback). `MathUtils.countUniqueValues` counts distinct values up to a limit.

- [#430](https://github.com/TissUUmaps/TissUUmaps/pull/430) [`fa7e975`](https://github.com/TissUUmaps/TissUUmaps/commit/fa7e97576d6748d9d1cb1cf32f51ef950587415a) Thanks [@cavenel](https://github.com/cavenel)! - Shape clouds get `fillVisibility` and `strokeVisibility`, which hide the fill or the stroke of all their shapes on top of the per-shape settings.

- [#313](https://github.com/TissUUmaps/TissUUmaps/pull/313) [`b091137`](https://github.com/TissUUmaps/TissUUmaps/commit/b091137fa80e5c18dd79e12b835a966e1d8bfc13) Thanks [@cavenel](https://github.com/cavenel)! - Projects get the optional `physicalUnitSize` (the size of one world unit, in `physicalUnit`), `physicalUnit` and `scalebarVisibility`, set with the `setPhysicalUnitSize`, `setPhysicalUnit` and `setScalebarVisibility` store actions. `physicalUnit` is free text; `PhysicalUnit` lists the suggested units, from picometers to kilometers, named as in OME-NGFF, and `physicalUnits` gives the symbol and the size in meters of each.

- [#312](https://github.com/TissUUmaps/TissUUmaps/pull/312) [`d6720ba`](https://github.com/TissUUmaps/TissUUmaps/commit/d6720ba27b069f73892a94a07fc3293e94c3faeb) Thanks [@cavenel](https://github.com/cavenel)! - `InteractionMode` is also a const object that lists the interaction modes.

### Patch Changes

- [#410](https://github.com/TissUUmaps/TissUUmaps/pull/410) [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29) Thanks [@jwindhager](https://github.com/jwindhager)! - Error and warning messages are worded consistently, with string values in single quotes.

- [#402](https://github.com/TissUUmaps/TissUUmaps/pull/402) [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5) Thanks [@cavenel](https://github.com/cavenel)! - `TableUtils.computeValueRange` returns `[c, c]` for a column whose values are all `c`, instead of `undefined`.

- [#428](https://github.com/TissUUmaps/TissUUmaps/pull/428) [`0fb9cf3`](https://github.com/TissUUmaps/TissUUmaps/commit/0fb9cf37a30712d732862cf7d89ca1626cef88b3) Thanks [@cavenel](https://github.com/cavenel)! - Properties of raw objects that are set to `undefined`, including OpenSeadragon and WebGL options of projects, get their default values, like missing properties.

## 0.1.0-beta.3

### Minor Changes

- [#302](https://github.com/TissUUmaps/TissUUmaps/pull/302) [`a422564`](https://github.com/TissUUmaps/TissUUmaps/commit/a4225640dd4be545727210c11e5c12d3d81d650a) Thanks [@jwindhager](https://github.com/jwindhager)! - Data providers gain the optional `supports`, `readName` and `prepareDataSource` methods, and `SourceUtils` gains `getPathSegments`, `getStem`, `getExtension` and `makeProjectPath`.

  Breaking: data provider UI schemas no longer contain the `source` control, which the app now renders itself.

- [#300](https://github.com/TissUUmaps/TissUUmaps/pull/300) [`9fc85a9`](https://github.com/TissUUmaps/TissUUmaps/commit/9fc85a9148be339aa0e419620253f03aace7041a) Thanks [@jwindhager](https://github.com/jwindhager)! - The app store tracks the active panel (`activePanelId`) and the expanded objects of each panel (`expandedImageIds` etc.).

  Breaking: `showImageSettings` etc. and the `imageSettingsRequest` etc. state are removed. Plugins call `setExpandedImageIds` etc. and `setActivePanelId` instead.

### Patch Changes

- [#307](https://github.com/TissUUmaps/TissUUmaps/pull/307) [`16e181e`](https://github.com/TissUUmaps/TissUUmaps/commit/16e181efd919f11a54b259c7e96231b724e628dd) Thanks [@jwindhager](https://github.com/jwindhager)! - Documentation links point to the new TissUUmaps website at https://tissuumaps.github.io/: the app's Docs link, and the packages' READMEs and homepages.

## 0.1.0-beta.2

### Minor Changes

- [#296](https://github.com/TissUUmaps/TissUUmaps/pull/296) [`15a214a`](https://github.com/TissUUmaps/TissUUmaps/commit/15a214a7950a1e9aa605ca5445cc4273b826ca2d) Thanks [@jwindhager](https://github.com/jwindhager)! - `immer` is declared as an optional peer dependency ([#277](https://github.com/TissUUmaps/TissUUmaps/issues/277)), since the store types depend on it through `zustand/immer`.

## 0.1.0-beta.1

### Minor Changes

- [#259](https://github.com/TissUUmaps/TissUUmaps/pull/259) [`62b79c2`](https://github.com/TissUUmaps/TissUUmaps/commit/62b79c2218ea5419972553b43cc217084d9a9d23) Thanks [@cavenel](https://github.com/cavenel)! - Plugins can show the settings of an image, labels, points or shapes: `appStore.getState().showImageSettings(imageId)` brings the Images panel to the front and expands the image's settings, and `showLabelsSettings`, `showPointsSettings` and `showShapesSettings` do the same for the other panels. Adds these actions and the `imageSettingsRequest`, `labelsSettingsRequest`, `pointsSettingsRequest` and `shapesSettingsRequest` app store state.

- [#253](https://github.com/TissUUmaps/TissUUmaps/pull/253) [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62) Thanks [@jwindhager](https://github.com/jwindhager)! - `TableData.loadUniqueValueCounts` and `TableData.loadValueRange` are now optional: table data only implements them if it can determine the counts or the range more cheaply than by loading the column (e.g. from file metadata). Callers use the new `TableUtils.loadUniqueValueCounts` and `TableUtils.loadValueRange`, which fall back to the loaded column values (see `TableUtils.computeValueRange`). CSV tables no longer implement either method, and Parquet and hierarchical (AnnData) tables no longer implement `loadUniqueValueCounts`. Unless the table data determines them itself, the app now computes unique value counts and value ranges from the column values it has already loaded, instead of loading the column again.

- [#253](https://github.com/TissUUmaps/TissUUmaps/pull/253) [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62) Thanks [@jwindhager](https://github.com/jwindhager)! - Group-by palettes (colors and markers without a map) now assign their values by group position, in the order the groups first appear in the table column, instead of by hashing the group names; random colors are still hashed. Projects that group by a palette show different colors or markers than before. `ConfigUtils.createGroupValueGetter` takes the ordered groups alongside the palette, and the new `TableUtils.loadGroupCounts` lists them.

- [#252](https://github.com/TissUUmaps/TissUUmaps/pull/252) [`cb49f75`](https://github.com/TissUUmaps/TissUUmaps/commit/cb49f75bb87c17d06198562d53e4ab17cd89aaf5) Thanks [@jwindhager](https://github.com/jwindhager)! - Plugins are no longer mounted when registered: a menu in the tab bar lists the registered plugins and mounts the one picked, and closing a plugin's panel only unmounts it. The menu can also load third-party plugins, ES modules whose default export is the plugin, from a local file or a URL, absolute or relative to the page. Third-party plugins should both default-export the plugin and register it through `window.tissuumaps`, so that the same file can be loaded through the menu or included in the page. Adds `PluginRegistry.mountPlugin`/`unmountPlugin`.

## 0.1.0-beta.0

### Minor Changes

- [#230](https://github.com/TissUUmaps/TissUUmaps/pull/230) [`3e2feac`](https://github.com/TissUUmaps/TissUUmaps/commit/3e2feaccf033d6ebf785e78e129671cd7683bd2e) Thanks [@jwindhager](https://github.com/jwindhager)! - Initial beta release.
