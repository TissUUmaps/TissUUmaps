# @tissuumaps/render

## 0.1.0-beta.4

### Minor Changes

- [#402](https://github.com/TissUUmaps/TissUUmaps/pull/402) [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5) Thanks [@cavenel](https://github.com/cavenel)! - Opacities from a table column are mapped linearly from the config's `range` to `[0, 1]`. An unset end of the range is the column's minimum or maximum, as for colors.

  Breaking: table values are no longer used as opacities directly. Set `range: [0, 1]` to keep the old mapping.

- [#420](https://github.com/TissUUmaps/TissUUmaps/pull/420) [`cd2e922`](https://github.com/TissUUmaps/TissUUmaps/commit/cd2e922bdf98a4713c6a28940dbc31cf6d53fb6b) Thanks [@cavenel](https://github.com/cavenel)! - Points with an order configuration are drawn in ascending order of its column's values, so that high values are on top.

- [#430](https://github.com/TissUUmaps/TissUUmaps/pull/430) [`fa7e975`](https://github.com/TissUUmaps/TissUUmaps/commit/fa7e97576d6748d9d1cb1cf32f51ef950587415a) Thanks [@cavenel](https://github.com/cavenel)! - The shapes renderer hides the fill or the stroke of a shape cloud when its `fillVisibility` or `strokeVisibility` is `false`.

### Patch Changes

- [#394](https://github.com/TissUUmaps/TissUUmaps/pull/394) [`3503273`](https://github.com/TissUUmaps/TissUUmaps/commit/3503273f101dda019c8c2f170d3297f93e474b86) Thanks [@cavenel](https://github.com/cavenel)! - A color range with only one end set takes the other end from the table column.

- [#410](https://github.com/TissUUmaps/TissUUmaps/pull/410) [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29) Thanks [@jwindhager](https://github.com/jwindhager)! - OpenSeadragon renderer messages are worded consistently, log the full error, and an object that cannot be resolved but is still rendered is logged as a warning.

- [#410](https://github.com/TissUUmaps/TissUUmaps/pull/410) [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29) Thanks [@jwindhager](https://github.com/jwindhager)! - WebGL, SVG and resolver messages are worded consistently, with string values in single quotes.

- [#402](https://github.com/TissUUmaps/TissUUmaps/pull/402) [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5) Thanks [@cavenel](https://github.com/cavenel)! - Colors and opacities from a table column map a range whose ends are equal as a step: values below it get the first color or are fully transparent, all others get the last color or are fully opaque.

- [#398](https://github.com/TissUUmaps/TissUUmaps/pull/398) [`a99d552`](https://github.com/TissUUmaps/TissUUmaps/commit/a99d552103ae8906467214c71d94a480e72a4aa2) Thanks [@cavenel](https://github.com/cavenel)! - Images rotated by an angle that is not a multiple of 90 degrees get correct bounds, so the home view frames them.

- [#387](https://github.com/TissUUmaps/TissUUmaps/pull/387) [`67de073`](https://github.com/TissUUmaps/TissUUmaps/commit/67de073b7d3e5a55e276ca7035dc057c8a643f59) Thanks [@cavenel](https://github.com/cavenel)! - Points smaller than 8 device pixels are no longer drawn on a faint square.

- [#399](https://github.com/TissUUmaps/TissUUmaps/pull/399) [`02a0b23`](https://github.com/TissUUmaps/TissUUmaps/commit/02a0b23526bc424580dd1abcd33b62abb115c24d) Thanks [@cavenel](https://github.com/cavenel)! - Rotated tiled images, such as OME-Zarr images, no longer show a copy of their first tile row or column next to them when zoomed out.
- Updated dependencies [[`3503273`](https://github.com/TissUUmaps/TissUUmaps/commit/3503273f101dda019c8c2f170d3297f93e474b86), [`e369291`](https://github.com/TissUUmaps/TissUUmaps/commit/e36929107a23b1881a99d80536004daaf88b3bd3), [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29), [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5), [`b774531`](https://github.com/TissUUmaps/TissUUmaps/commit/b7745316a23f68c84d57d8cf51f2b914258f74ec), [`0fb9cf3`](https://github.com/TissUUmaps/TissUUmaps/commit/0fb9cf37a30712d732862cf7d89ca1626cef88b3), [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5), [`cd2e922`](https://github.com/TissUUmaps/TissUUmaps/commit/cd2e922bdf98a4713c6a28940dbc31cf6d53fb6b), [`ddd7fac`](https://github.com/TissUUmaps/TissUUmaps/commit/ddd7fac9046e0226a7225aa7621ad1a759ffa35a), [`6464a42`](https://github.com/TissUUmaps/TissUUmaps/commit/6464a4217f7cd3636a8040dbf2f5f55151b180b2), [`575da90`](https://github.com/TissUUmaps/TissUUmaps/commit/575da90af95d3d17fbf8a8c2bd64e43204a2bc8a), [`fa7e975`](https://github.com/TissUUmaps/TissUUmaps/commit/fa7e97576d6748d9d1cb1cf32f51ef950587415a), [`b091137`](https://github.com/TissUUmaps/TissUUmaps/commit/b091137fa80e5c18dd79e12b835a966e1d8bfc13), [`d6720ba`](https://github.com/TissUUmaps/TissUUmaps/commit/d6720ba27b069f73892a94a07fc3293e94c3faeb)]:
  - @tissuumaps/core@0.1.0-beta.4

## 0.1.0-beta.3

### Minor Changes

- [#306](https://github.com/TissUUmaps/TissUUmaps/pull/306) [`2603f6f`](https://github.com/TissUUmaps/TissUUmaps/commit/2603f6f2c48748c07d2ead3413fc95f4c1745675) Thanks [@jwindhager](https://github.com/jwindhager)! - Points, shapes, labels and images are each drawn as soon as their own data is ready, so a slow or failing object no longer holds back the other objects of its kind ([#269](https://github.com/TissUUmaps/TissUUmaps/issues/269)). The WebGL renderers' `synchronize` takes an `onChange` callback, which is called whenever a rendered object has been added, updated or dropped, and no longer resolves to whether anything changed.

- [#290](https://github.com/TissUUmaps/TissUUmaps/pull/290) [`cd0c14e`](https://github.com/TissUUmaps/TissUUmaps/commit/cd0c14e27ccb02645ab37d1da4c829ff3e56e0c2) Thanks [@cavenel](https://github.com/cavenel)! - `OpenSeadragonContext` applies `osOptions.viewerOptions.showNavigator` live, without rebuilding the viewer.

### Patch Changes

- [#307](https://github.com/TissUUmaps/TissUUmaps/pull/307) [`16e181e`](https://github.com/TissUUmaps/TissUUmaps/commit/16e181efd919f11a54b259c7e96231b724e628dd) Thanks [@jwindhager](https://github.com/jwindhager)! - Documentation links point to the new TissUUmaps website at https://tissuumaps.github.io/: the app's Docs link, and the packages' READMEs and homepages.
- Updated dependencies [[`75433b7`](https://github.com/TissUUmaps/TissUUmaps/commit/75433b7dfe6f553f6c7020fe7088a6b38e099790), [`9fc85a9`](https://github.com/TissUUmaps/TissUUmaps/commit/9fc85a9148be339aa0e419620253f03aace7041a), [`16e181e`](https://github.com/TissUUmaps/TissUUmaps/commit/16e181efd919f11a54b259c7e96231b724e628dd)]:
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
