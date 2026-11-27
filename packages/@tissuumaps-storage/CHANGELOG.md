# @tissuumaps/storage

## 0.1.0-beta.5

### Minor Changes

- [#425](https://github.com/TissUUmaps/TissUUmaps/pull/425) [`0efa6a6`](https://github.com/TissUUmaps/TissUUmaps/commit/0efa6a6983fce684095345d3ec978d524024766a) Thanks [@cavenel](https://github.com/cavenel)! - An image data source's `tileSource` only takes an inline tile source configuration. A descriptor URL is given as `source`, the only field of a data source that is resolved as a path.

  Breaking: a descriptor URL given as `tileSource` has to be given as `source`. A descriptor whose URL has an image extension has to be given inline as `tileSource`.

- [#404](https://github.com/TissUUmaps/TissUUmaps/pull/404) [`ab5dcc7`](https://github.com/TissUUmaps/TissUUmaps/commit/ab5dcc787dbe9154b6fe7ef9b8b44475ce7b69be) Thanks [@jwindhager](https://github.com/jwindhager)! - Parquet tables can be read from Dask dataset directories, such as the points of a SpatialData store.

  Breaking: the `ParquetTableData` constructor takes the table's Parquet files (`ParquetSource[]`) instead of one.

- [#403](https://github.com/TissUUmaps/TissUUmaps/pull/403) [`ddd7fac`](https://github.com/TissUUmaps/TissUUmaps/commit/ddd7fac9046e0226a7225aa7621ad1a759ffa35a) Thanks [@jwindhager](https://github.com/jwindhager)! - Add `TissUUmapsProjectProvider`, a project provider that reads and writes TissUUmaps project files (`.tm4`).

- [#411](https://github.com/TissUUmaps/TissUUmaps/pull/411) [`575da90`](https://github.com/TissUUmaps/TissUUmaps/commit/575da90af95d3d17fbf8a8c2bd64e43204a2bc8a) Thanks [@cavenel](https://github.com/cavenel)! - HDF5 and Zarr tables declare boolean datasets as categorical columns, AnnData tables also their `categorical` and `nullable-boolean` columns, and Parquet tables their pandas `category` columns. AnnData tables group their columns into observations (`obs`), features (`X`, the layers and `raw`) and embeddings (`obsm`).

- [#405](https://github.com/TissUUmaps/TissUUmaps/pull/405) [`86a890e`](https://github.com/TissUUmaps/TissUUmaps/commit/86a890e6e38e1acbf02a4708591ad6229d9b57a7) Thanks [@jwindhager](https://github.com/jwindhager)! - Add `SpatialDataProjectProvider` (`@tissuumaps/storage/spatialdata`), a project provider that loads SpatialData stores: images and labels as OME-Zarr objects, polygons as Parquet shapes, circles and points as point clouds, tables as Zarr tables, and coordinate systems as layers.

### Patch Changes

- [#410](https://github.com/TissUUmaps/TissUUmaps/pull/410) [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29) Thanks [@jwindhager](https://github.com/jwindhager)! - Zarr, HDF5 and AnnData error messages are worded consistently, with string values in single quotes.

- [#410](https://github.com/TissUUmaps/TissUUmaps/pull/410) [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29) Thanks [@jwindhager](https://github.com/jwindhager)! - OME-Zarr, OpenSeadragon, SpatialData and TissUUmaps project messages are worded consistently, with string values in single quotes.

- [#410](https://github.com/TissUUmaps/TissUUmaps/pull/410) [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29) Thanks [@jwindhager](https://github.com/jwindhager)! - CSV, Parquet, GeoParquet and GeoJSON messages are worded consistently, with string values in single quotes.

- [#410](https://github.com/TissUUmaps/TissUUmaps/pull/410) [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29) Thanks [@jwindhager](https://github.com/jwindhager)! - TIFF error messages are worded consistently, and unreadable TIFF channel histograms are logged as a warning instead of an error.

- [#402](https://github.com/TissUUmaps/TissUUmaps/pull/402) [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5) Thanks [@cavenel](https://github.com/cavenel)! - Parquet tables report the value range `[c, c]` for a column whose values are all `c`, instead of none.

- [#422](https://github.com/TissUUmaps/TissUUmaps/pull/422) [`dcc9567`](https://github.com/TissUUmaps/TissUUmaps/commit/dcc956726416841da254793521346908e51bd2ad) Thanks [@cavenel](https://github.com/cavenel)! - SpatialData stores now open with only the first element of each coordinate system shown, images first, instead of every element.
- Updated dependencies [[`3503273`](https://github.com/TissUUmaps/TissUUmaps/commit/3503273f101dda019c8c2f170d3297f93e474b86), [`e369291`](https://github.com/TissUUmaps/TissUUmaps/commit/e36929107a23b1881a99d80536004daaf88b3bd3), [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29), [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5), [`b774531`](https://github.com/TissUUmaps/TissUUmaps/commit/b7745316a23f68c84d57d8cf51f2b914258f74ec), [`0fb9cf3`](https://github.com/TissUUmaps/TissUUmaps/commit/0fb9cf37a30712d732862cf7d89ca1626cef88b3), [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5), [`cd2e922`](https://github.com/TissUUmaps/TissUUmaps/commit/cd2e922bdf98a4713c6a28940dbc31cf6d53fb6b), [`ddd7fac`](https://github.com/TissUUmaps/TissUUmaps/commit/ddd7fac9046e0226a7225aa7621ad1a759ffa35a), [`6464a42`](https://github.com/TissUUmaps/TissUUmaps/commit/6464a4217f7cd3636a8040dbf2f5f55151b180b2), [`575da90`](https://github.com/TissUUmaps/TissUUmaps/commit/575da90af95d3d17fbf8a8c2bd64e43204a2bc8a), [`fa7e975`](https://github.com/TissUUmaps/TissUUmaps/commit/fa7e97576d6748d9d1cb1cf32f51ef950587415a), [`b091137`](https://github.com/TissUUmaps/TissUUmaps/commit/b091137fa80e5c18dd79e12b835a966e1d8bfc13), [`d6720ba`](https://github.com/TissUUmaps/TissUUmaps/commit/d6720ba27b069f73892a94a07fc3293e94c3faeb)]:
  - @tissuumaps/core@0.1.0-beta.4

## 0.1.0-beta.4

### Minor Changes

- [#302](https://github.com/TissUUmaps/TissUUmaps/pull/302) [`a422564`](https://github.com/TissUUmaps/TissUUmaps/commit/a4225640dd4be545727210c11e5c12d3d81d650a) Thanks [@jwindhager](https://github.com/jwindhager)! - All built-in data providers implement `supports`, and the OME-TIFF and OME-Zarr image and labels providers implement `readName`.

  Breaking: the table points data provider's constructor takes `getTableDataProviders` and `addTable`.

### Patch Changes

- [#303](https://github.com/TissUUmaps/TissUUmaps/pull/303) [`9db26e3`](https://github.com/TissUUmaps/TissUUmaps/commit/9db26e3cb3e06d54e9ad4cae7aaadf25bbf396eb) Thanks [@jwindhager](https://github.com/jwindhager)! - CSV tables load again in production builds: papaparse is pinned to 5.6.0, since the minified 5.6.1 and 5.7.0 builds crash in their parser worker (mholt/PapaParse#1122).

- [#307](https://github.com/TissUUmaps/TissUUmaps/pull/307) [`16e181e`](https://github.com/TissUUmaps/TissUUmaps/commit/16e181efd919f11a54b259c7e96231b724e628dd) Thanks [@jwindhager](https://github.com/jwindhager)! - Documentation links point to the new TissUUmaps website at https://tissuumaps.github.io/: the app's Docs link, and the packages' READMEs and homepages.
- Updated dependencies [[`75433b7`](https://github.com/TissUUmaps/TissUUmaps/commit/75433b7dfe6f553f6c7020fe7088a6b38e099790), [`9fc85a9`](https://github.com/TissUUmaps/TissUUmaps/commit/9fc85a9148be339aa0e419620253f03aace7041a), [`16e181e`](https://github.com/TissUUmaps/TissUUmaps/commit/16e181efd919f11a54b259c7e96231b724e628dd)]:
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
