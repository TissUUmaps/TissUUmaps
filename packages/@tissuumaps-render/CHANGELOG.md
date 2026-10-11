# @tissuumaps/render

## 0.1.0-beta.3

### Minor Changes

- [#290](https://github.com/TissUUmaps/TissUUmaps/pull/290) [`cd0c14e`](https://github.com/TissUUmaps/TissUUmaps/commit/cd0c14e27ccb02645ab37d1da4c829ff3e56e0c2) Thanks [@cavenel](https://github.com/cavenel)! - `OpenSeadragonContext` applies `osOptions.viewerOptions.showNavigator` live, without rebuilding the viewer.

### Patch Changes

- Updated dependencies [[`75433b7`](https://github.com/TissUUmaps/TissUUmaps/commit/75433b7dfe6f553f6c7020fe7088a6b38e099790), [`9fc85a9`](https://github.com/TissUUmaps/TissUUmaps/commit/9fc85a9148be339aa0e419620253f03aace7041a)]:
  - @tissuumaps/core@0.1.0-beta.3

## 0.1.0-beta.2

### Patch Changes

- [#295](https://github.com/TissUUmaps/TissUUmaps/pull/295) [`67b9236`](https://github.com/TissUUmaps/TissUUmaps/commit/67b92362506385171c41b0b9356a23c463303d29) Thanks [@jwindhager](https://github.com/jwindhager)! - Plain images can be opened from a URL or path ([#274](https://github.com/TissUUmaps/TissUUmaps/issues/274)): an image data source's `source` with a PNG, JPEG or WebP file extension, or a `data:image/` URL, is opened as a plain image, also from the workspace; any other `source` is handed to OpenSeadragon as a tile source descriptor, which cannot be opened from the workspace. The data source `type` becomes `image` (was `openseadragon`) and its `tileSourceConfig` field becomes `tileSource`, which takes the URL of a descriptor or an inline tile source configuration and takes precedence over `source`. Inline tile source configurations taken from a project no longer fail to open because OpenSeadragon writes into them. Projects using `openseadragon` data sources have to be updated.
- Updated dependencies [[`15a214a`](https://github.com/TissUUmaps/TissUUmaps/commit/15a214a7950a1e9aa605ca5445cc4273b826ca2d)]:
  - @tissuumaps/core@0.1.0-beta.2

## 0.1.0-beta.1

### Minor Changes

- [#253](https://github.com/TissUUmaps/TissUUmaps/pull/253) [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62) Thanks [@jwindhager](https://github.com/jwindhager)! - Group-by palettes (colors and markers without a map) now assign their values by group position, in the order the groups first appear in the table column, instead of by hashing the group names; random colors are still hashed. Projects that group by a palette show different colors or markers than before. `ConfigUtils.createGroupValueGetter` takes the ordered groups alongside the palette, and the new `TableUtils.loadGroupCounts` lists them.

### Patch Changes

- [#253](https://github.com/TissUUmaps/TissUUmaps/pull/253) [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62) Thanks [@jwindhager](https://github.com/jwindhager)! - `TableData.loadUniqueValueCounts` and `TableData.loadValueRange` are now optional: table data only implements them if it can determine the counts or the range more cheaply than by loading the column (e.g. from file metadata). Callers use the new `TableUtils.loadUniqueValueCounts` and `TableUtils.loadValueRange`, which fall back to the loaded column values (see `TableUtils.computeValueRange`). CSV tables no longer implement either method, and Parquet and hierarchical (AnnData) tables no longer implement `loadUniqueValueCounts`. Unless the table data determines them itself, the app now computes unique value counts and value ranges from the column values it has already loaded, instead of loading the column again.
- Updated dependencies [[`62b79c2`](https://github.com/TissUUmaps/TissUUmaps/commit/62b79c2218ea5419972553b43cc217084d9a9d23), [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62), [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62), [`cb49f75`](https://github.com/TissUUmaps/TissUUmaps/commit/cb49f75bb87c17d06198562d53e4ab17cd89aaf5)]:
  - @tissuumaps/core@0.1.0-beta.1

## 0.1.0-beta.0

### Minor Changes

- [#230](https://github.com/TissUUmaps/TissUUmaps/pull/230) [`3e2feac`](https://github.com/TissUUmaps/TissUUmaps/commit/3e2feaccf033d6ebf785e78e129671cd7683bd2e) Thanks [@jwindhager](https://github.com/jwindhager)! - Initial beta release.

### Patch Changes

- Updated dependencies [[`3e2feac`](https://github.com/TissUUmaps/TissUUmaps/commit/3e2feaccf033d6ebf785e78e129671cd7683bd2e)]:
  - @tissuumaps/core@0.1.0-beta.0
