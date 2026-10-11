# tissuumaps

## 4.0.0-beta.3

### Minor Changes

- [#289](https://github.com/TissUUmaps/TissUUmaps/pull/289) [`68c1573`](https://github.com/TissUUmaps/TissUUmaps/commit/68c1573ce0ddc822806b5d8191e4fb0e8f226be9) Thanks [@cavenel](https://github.com/cavenel)! - Add a Home button to the viewer that fits the view to all content.

- [#302](https://github.com/TissUUmaps/TissUUmaps/pull/302) [`a422564`](https://github.com/TissUUmaps/TissUUmaps/commit/a4225640dd4be545727210c11e5c12d3d81d650a) Thanks [@jwindhager](https://github.com/jwindhager)! - Add data objects, projects and folders by drag and drop: dropped files and folders open a pre-filled add dialog, and a project file or folder dropped on the project panel is opened after confirmation.

  The add dialog picks the data provider and name from the source, and source pickers fill in paths relative to the project file if it lies in the connected folder.

- [#300](https://github.com/TissUUmaps/TissUUmaps/pull/300) [`9fc85a9`](https://github.com/TissUUmaps/TissUUmaps/commit/9fc85a9148be339aa0e419620253f03aace7041a) Thanks [@jwindhager](https://github.com/jwindhager)! - Breaking for plugins: the app store's `showImageSettings` etc. are removed. To show an object's settings, add its ID to `expandedImageIds` etc. and set `activePanelId`.

- [#308](https://github.com/TissUUmaps/TissUUmaps/pull/308) [`352ff36`](https://github.com/TissUUmaps/TissUUmaps/commit/352ff36bad440ebbdb56d4c4d7deafe2e0f81c0d) Thanks [@jwindhager](https://github.com/jwindhager)! - Opening a project file from outside the connected folder now asks for confirmation first, and warns that its relative paths may fail to resolve.

- [#290](https://github.com/TissUUmaps/TissUUmaps/pull/290) [`cd0c14e`](https://github.com/TissUUmaps/TissUUmaps/commit/cd0c14e27ccb02645ab37d1da4c829ff3e56e0c2) Thanks [@cavenel](https://github.com/cavenel)! - Add a project setting to show or hide the navigator.

### Patch Changes

- Updated dependencies [[`75433b7`](https://github.com/TissUUmaps/TissUUmaps/commit/75433b7dfe6f553f6c7020fe7088a6b38e099790), [`75433b7`](https://github.com/TissUUmaps/TissUUmaps/commit/75433b7dfe6f553f6c7020fe7088a6b38e099790), [`75433b7`](https://github.com/TissUUmaps/TissUUmaps/commit/75433b7dfe6f553f6c7020fe7088a6b38e099790), [`9fc85a9`](https://github.com/TissUUmaps/TissUUmaps/commit/9fc85a9148be339aa0e419620253f03aace7041a), [`9db26e3`](https://github.com/TissUUmaps/TissUUmaps/commit/9db26e3cb3e06d54e9ad4cae7aaadf25bbf396eb), [`75433b7`](https://github.com/TissUUmaps/TissUUmaps/commit/75433b7dfe6f553f6c7020fe7088a6b38e099790)]:
  - @tissuumaps/react@0.1.0-beta.1
  - @tissuumaps/core@0.1.0-beta.3
  - @tissuumaps/storage@0.1.0-beta.4
  - @tissuumaps/render@0.1.0-beta.3

## 4.0.0-beta.2

### Minor Changes

- [#295](https://github.com/TissUUmaps/TissUUmaps/pull/295) [`67b9236`](https://github.com/TissUUmaps/TissUUmaps/commit/67b92362506385171c41b0b9356a23c463303d29) Thanks [@jwindhager](https://github.com/jwindhager)! - Plain images can be opened from a URL or path ([#274](https://github.com/TissUUmaps/TissUUmaps/issues/274)): an image data source's `source` with a PNG, JPEG or WebP file extension, or a `data:image/` URL, is opened as a plain image, also from the workspace; any other `source` is handed to OpenSeadragon as a tile source descriptor, which cannot be opened from the workspace. The data source `type` becomes `image` (was `openseadragon`) and its `tileSourceConfig` field becomes `tileSource`, which takes the URL of a descriptor or an inline tile source configuration and takes precedence over `source`. Inline tile source configurations taken from a project no longer fail to open because OpenSeadragon writes into them. Projects using `openseadragon` data sources have to be updated.

### Patch Changes

- Updated dependencies [[`15a214a`](https://github.com/TissUUmaps/TissUUmaps/commit/15a214a7950a1e9aa605ca5445cc4273b826ca2d), [`5adda33`](https://github.com/TissUUmaps/TissUUmaps/commit/5adda33862da2f14d6d1be8eb26231f10c7d6f52), [`67b9236`](https://github.com/TissUUmaps/TissUUmaps/commit/67b92362506385171c41b0b9356a23c463303d29), [`4888805`](https://github.com/TissUUmaps/TissUUmaps/commit/4888805fd81a2af44297432a8decc3cfaec7093e)]:
  - @tissuumaps/core@0.1.0-beta.2
  - @tissuumaps/storage@0.1.0-beta.3
  - @tissuumaps/render@0.1.0-beta.2

## 4.0.0-beta.1

### Minor Changes

- [#256](https://github.com/TissUUmaps/TissUUmaps/pull/256) [`dfe46b5`](https://github.com/TissUUmaps/TissUUmaps/commit/dfe46b5cb573170fe046d734f82c572003e14b15) Thanks [@jwindhager](https://github.com/jwindhager)! - Name project files `project.tm4`: the default project on startup, the file pickers and downloaded projects now use the `.tm4` extension instead of `.tmap`/`.json`.

- [#259](https://github.com/TissUUmaps/TissUUmaps/pull/259) [`62b79c2`](https://github.com/TissUUmaps/TissUUmaps/commit/62b79c2218ea5419972553b43cc217084d9a9d23) Thanks [@cavenel](https://github.com/cavenel)! - Plugins can show the settings of an image, labels, points or shapes: `appStore.getState().showImageSettings(imageId)` brings the Images panel to the front and expands the image's settings, and `showLabelsSettings`, `showPointsSettings` and `showShapesSettings` do the same for the other panels. Adds these actions and the `imageSettingsRequest`, `labelsSettingsRequest`, `pointsSettingsRequest` and `shapesSettingsRequest` app store state.

- [#253](https://github.com/TissUUmaps/TissUUmaps/pull/253) [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62) Thanks [@jwindhager](https://github.com/jwindhager)! - Group-by palettes (colors and markers without a map) now assign their values by group position, in the order the groups first appear in the table column, instead of by hashing the group names; random colors are still hashed. Projects that group by a palette show different colors or markers than before. `ConfigUtils.createGroupValueGetter` takes the ordered groups alongside the palette, and the new `TableUtils.loadGroupCounts` lists them.

- [#252](https://github.com/TissUUmaps/TissUUmaps/pull/252) [`cb49f75`](https://github.com/TissUUmaps/TissUUmaps/commit/cb49f75bb87c17d06198562d53e4ab17cd89aaf5) Thanks [@jwindhager](https://github.com/jwindhager)! - Plugins are no longer mounted when registered: a menu in the tab bar lists the registered plugins and mounts the one picked, and closing a plugin's panel only unmounts it. The menu can also load third-party plugins, ES modules whose default export is the plugin, from a local file or a URL, absolute or relative to the page. Third-party plugins should both default-export the plugin and register it through `window.tissuumaps`, so that the same file can be loaded through the menu or included in the page. Adds `PluginRegistry.mountPlugin`/`unmountPlugin`.

- [#238](https://github.com/TissUUmaps/TissUUmaps/pull/238) [`4104c1c`](https://github.com/TissUUmaps/TissUUmaps/commit/4104c1cf4bc79ed7afa57e1c4ee7109291cd1d11) Thanks [@cavenel](https://github.com/cavenel)! - While a folder is connected, the Source field of a data source has buttons that pick a file or a folder (e.g. a Zarr store) inside it and fill in its workspace-relative path. Picking a file outside the folder, or the connected folder itself, is refused with a message.

### Patch Changes

- [#254](https://github.com/TissUUmaps/TissUUmaps/pull/254) [`76faab6`](https://github.com/TissUUmaps/TissUUmaps/commit/76faab6b76287b57231d363381243c434fbf3f7f) Thanks [@jwindhager](https://github.com/jwindhager)! - Hide the shape drawing controls in the viewer until drawing is linked with actions.

- [#253](https://github.com/TissUUmaps/TissUUmaps/pull/253) [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62) Thanks [@jwindhager](https://github.com/jwindhager)! - `TableData.loadUniqueValueCounts` and `TableData.loadValueRange` are now optional: table data only implements them if it can determine the counts or the range more cheaply than by loading the column (e.g. from file metadata). Callers use the new `TableUtils.loadUniqueValueCounts` and `TableUtils.loadValueRange`, which fall back to the loaded column values (see `TableUtils.computeValueRange`). CSV tables no longer implement either method, and Parquet and hierarchical (AnnData) tables no longer implement `loadUniqueValueCounts`. Unless the table data determines them itself, the app now computes unique value counts and value ranges from the column values it has already loaded, instead of loading the column again.
- Updated dependencies [[`62b79c2`](https://github.com/TissUUmaps/TissUUmaps/commit/62b79c2218ea5419972553b43cc217084d9a9d23), [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62), [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62), [`cb49f75`](https://github.com/TissUUmaps/TissUUmaps/commit/cb49f75bb87c17d06198562d53e4ab17cd89aaf5)]:
  - @tissuumaps/core@0.1.0-beta.1
  - @tissuumaps/storage@0.1.0-beta.1
  - @tissuumaps/render@0.1.0-beta.1

## 4.0.0-beta.0

### Major Changes

- [#230](https://github.com/TissUUmaps/TissUUmaps/pull/230) [`3e2feac`](https://github.com/TissUUmaps/TissUUmaps/commit/3e2feaccf033d6ebf785e78e129671cd7683bd2e) Thanks [@jwindhager](https://github.com/jwindhager)! - Initial beta release.

### Patch Changes

- Updated dependencies [[`3e2feac`](https://github.com/TissUUmaps/TissUUmaps/commit/3e2feaccf033d6ebf785e78e129671cd7683bd2e)]:
  - @tissuumaps/core@0.1.0-beta.0
  - @tissuumaps/render@0.1.0-beta.0
  - @tissuumaps/storage@0.1.0-beta.0
  - @tissuumaps/plugins@0.1.0-beta.0
  - @tissuumaps/react@0.1.0-beta.0
