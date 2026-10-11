# @tissuumaps/react

## 0.1.0-beta.1

### Minor Changes

- [#289](https://github.com/TissUUmaps/TissUUmaps/pull/289) [`68c1573`](https://github.com/TissUUmaps/TissUUmaps/commit/68c1573ce0ddc822806b5d8191e4fb0e8f226be9) Thanks [@cavenel](https://github.com/cavenel)! - The `Viewer` exposes its `OpenSeadragonContext` through its ref.

### Patch Changes

- [#306](https://github.com/TissUUmaps/TissUUmaps/pull/306) [`2603f6f`](https://github.com/TissUUmaps/TissUUmaps/commit/2603f6f2c48748c07d2ead3413fc95f4c1745675) Thanks [@jwindhager](https://github.com/jwindhager)! - Points, shapes, labels and images are each drawn as soon as their own data is ready, so a slow or failing object no longer holds back the other objects of its kind ([#269](https://github.com/TissUUmaps/TissUUmaps/issues/269)). The WebGL renderers' `synchronize` takes an `onChange` callback, which is called whenever a rendered object has been added, updated or dropped, and no longer resolves to whether anything changed.

- [#307](https://github.com/TissUUmaps/TissUUmaps/pull/307) [`16e181e`](https://github.com/TissUUmaps/TissUUmaps/commit/16e181efd919f11a54b259c7e96231b724e628dd) Thanks [@jwindhager](https://github.com/jwindhager)! - Documentation links point to the new TissUUmaps website at https://tissuumaps.github.io/: the app's Docs link, and the packages' READMEs and homepages.
- Updated dependencies [[`75433b7`](https://github.com/TissUUmaps/TissUUmaps/commit/75433b7dfe6f553f6c7020fe7088a6b38e099790), [`9fc85a9`](https://github.com/TissUUmaps/TissUUmaps/commit/9fc85a9148be339aa0e419620253f03aace7041a), [`2603f6f`](https://github.com/TissUUmaps/TissUUmaps/commit/2603f6f2c48748c07d2ead3413fc95f4c1745675), [`75433b7`](https://github.com/TissUUmaps/TissUUmaps/commit/75433b7dfe6f553f6c7020fe7088a6b38e099790), [`16e181e`](https://github.com/TissUUmaps/TissUUmaps/commit/16e181efd919f11a54b259c7e96231b724e628dd)]:
  - @tissuumaps/core@0.1.0-beta.3
  - @tissuumaps/render@0.1.0-beta.3

## 0.1.0-beta.0

### Minor Changes

- [#230](https://github.com/TissUUmaps/TissUUmaps/pull/230) [`3e2feac`](https://github.com/TissUUmaps/TissUUmaps/commit/3e2feaccf033d6ebf785e78e129671cd7683bd2e) Thanks [@jwindhager](https://github.com/jwindhager)! - Initial beta release.

### Patch Changes

- Updated dependencies [[`3e2feac`](https://github.com/TissUUmaps/TissUUmaps/commit/3e2feaccf033d6ebf785e78e129671cd7683bd2e)]:
  - @tissuumaps/core@0.1.0-beta.0
  - @tissuumaps/render@0.1.0-beta.0
