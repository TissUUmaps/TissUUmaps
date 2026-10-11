# @tissuumaps/storage

## 0.1.0-beta.4

### Minor Changes

- [#302](https://github.com/TissUUmaps/TissUUmaps/pull/302) [`a422564`](https://github.com/TissUUmaps/TissUUmaps/commit/a4225640dd4be545727210c11e5c12d3d81d650a) Thanks [@jwindhager](https://github.com/jwindhager)! - All built-in data providers implement `supports`, and the OME-TIFF and OME-Zarr image and labels providers implement `readName`.

  Breaking: the table points data provider's constructor takes `getTableDataProviders` and `addTable`.

### Patch Changes

- [#303](https://github.com/TissUUmaps/TissUUmaps/pull/303) [`9db26e3`](https://github.com/TissUUmaps/TissUUmaps/commit/9db26e3cb3e06d54e9ad4cae7aaadf25bbf396eb) Thanks [@jwindhager](https://github.com/jwindhager)! - CSV tables load again in production builds: papaparse is pinned to 5.6.0, since the minified 5.6.1 and 5.7.0 builds crash in their parser worker (mholt/PapaParse#1122).
- Updated dependencies [[`75433b7`](https://github.com/TissUUmaps/TissUUmaps/commit/75433b7dfe6f553f6c7020fe7088a6b38e099790), [`9fc85a9`](https://github.com/TissUUmaps/TissUUmaps/commit/9fc85a9148be339aa0e419620253f03aace7041a)]:
  - @tissuumaps/core@0.1.0-beta.3

## 0.1.0-beta.3

### Minor Changes

- [#297](https://github.com/TissUUmaps/TissUUmaps/pull/297) [`5adda33`](https://github.com/TissUUmaps/TissUUmaps/commit/5adda33862da2f14d6d1be8eb26231f10c7d6f52) Thanks [@jwindhager](https://github.com/jwindhager)! - HDF5 and Zarr table data no longer implement `loadValueRange`, which decoded the whole column in the worker and so was no cheaper than loading it; their value ranges are now computed from the loaded column values (e.g. by `TableUtils.loadValueRange`), so continuous colors no longer read a column twice. Code calling `loadValueRange` directly on `HDF5TableData` or `ZarrTableData` has to use `TableUtils.loadValueRange` instead. Parquet columns without row group statistics, and point geometry columns without a GeoParquet bounding box, now get a value range computed from their values instead of none.

- [#295](https://github.com/TissUUmaps/TissUUmaps/pull/295) [`67b9236`](https://github.com/TissUUmaps/TissUUmaps/commit/67b92362506385171c41b0b9356a23c463303d29) Thanks [@jwindhager](https://github.com/jwindhager)! - Plain images can be opened from a URL or path ([#274](https://github.com/TissUUmaps/TissUUmaps/issues/274)): an image data source's `source` with a PNG, JPEG or WebP file extension, or a `data:image/` URL, is opened as a plain image, also from the workspace; any other `source` is handed to OpenSeadragon as a tile source descriptor, which cannot be opened from the workspace. The data source `type` becomes `image` (was `openseadragon`) and its `tileSourceConfig` field becomes `tileSource`, which takes the URL of a descriptor or an inline tile source configuration and takes precedence over `source`. Inline tile source configurations taken from a project no longer fail to open because OpenSeadragon writes into them. Projects using `openseadragon` data sources have to be updated.

### Patch Changes

- [#268](https://github.com/TissUUmaps/TissUUmaps/pull/268) [`4888805`](https://github.com/TissUUmaps/TissUUmaps/commit/4888805fd81a2af44297432a8decc3cfaec7093e) Thanks [@cavenel](https://github.com/cavenel)! - CSV tables load again in production builds: papaparse 5.7.0, whose minified build crashes in its parser worker (mholt/PapaParse#1122), is excluded from the supported papaparse versions.
- Updated dependencies [[`15a214a`](https://github.com/TissUUmaps/TissUUmaps/commit/15a214a7950a1e9aa605ca5445cc4273b826ca2d)]:
  - @tissuumaps/core@0.1.0-beta.2

## 0.1.0-beta.2

### Minor Changes

- [#263](https://github.com/TissUUmaps/TissUUmaps/pull/263) [`74e598a`](https://github.com/TissUUmaps/TissUUmaps/commit/74e598a64d32d0e0d5c72a1c2cbd597bb565f63c) Thanks [@jwindhager](https://github.com/jwindhager)! - Bump version to fix package publishing

## 0.1.0-beta.1

### Minor Changes

- [#253](https://github.com/TissUUmaps/TissUUmaps/pull/253) [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62) Thanks [@jwindhager](https://github.com/jwindhager)! - `TableData.loadUniqueValueCounts` and `TableData.loadValueRange` are now optional: table data only implements them if it can determine the counts or the range more cheaply than by loading the column (e.g. from file metadata). Callers use the new `TableUtils.loadUniqueValueCounts` and `TableUtils.loadValueRange`, which fall back to the loaded column values (see `TableUtils.computeValueRange`). CSV tables no longer implement either method, and Parquet and hierarchical (AnnData) tables no longer implement `loadUniqueValueCounts`. Unless the table data determines them itself, the app now computes unique value counts and value ranges from the column values it has already loaded, instead of loading the column again.

### Patch Changes

- Updated dependencies [[`62b79c2`](https://github.com/TissUUmaps/TissUUmaps/commit/62b79c2218ea5419972553b43cc217084d9a9d23), [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62), [`8acd194`](https://github.com/TissUUmaps/TissUUmaps/commit/8acd1946245c89c961af2857e13ae373bc256d62), [`cb49f75`](https://github.com/TissUUmaps/TissUUmaps/commit/cb49f75bb87c17d06198562d53e4ab17cd89aaf5)]:
  - @tissuumaps/core@0.1.0-beta.1

## 0.1.0-beta.0

### Minor Changes

- [#230](https://github.com/TissUUmaps/TissUUmaps/pull/230) [`3e2feac`](https://github.com/TissUUmaps/TissUUmaps/commit/3e2feaccf033d6ebf785e78e129671cd7683bd2e) Thanks [@jwindhager](https://github.com/jwindhager)! - Initial beta release.

### Patch Changes

- Updated dependencies [[`3e2feac`](https://github.com/TissUUmaps/TissUUmaps/commit/3e2feaccf033d6ebf785e78e129671cd7683bd2e)]:
  - @tissuumaps/core@0.1.0-beta.0
