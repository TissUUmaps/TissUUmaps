# @tissuumaps/react

## 0.1.0-beta.2

### Minor Changes

- [#313](https://github.com/TissUUmaps/TissUUmaps/pull/313) [`b091137`](https://github.com/TissUUmaps/TissUUmaps/commit/b091137fa80e5c18dd79e12b835a966e1d8bfc13) Thanks [@cavenel](https://github.com/cavenel)! - A new `useViewportChange` hook reports viewport changes to components inside the `Viewer`, without re-rendering.

### Patch Changes

- [#410](https://github.com/TissUUmaps/TissUUmaps/pull/410) [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29) Thanks [@jwindhager](https://github.com/jwindhager)! - Error messages of the viewer hooks read "Failed to …", like those of the other packages.
- Updated dependencies [[`3503273`](https://github.com/TissUUmaps/TissUUmaps/commit/3503273f101dda019c8c2f170d3297f93e474b86), [`3503273`](https://github.com/TissUUmaps/TissUUmaps/commit/3503273f101dda019c8c2f170d3297f93e474b86), [`e369291`](https://github.com/TissUUmaps/TissUUmaps/commit/e36929107a23b1881a99d80536004daaf88b3bd3), [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29), [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29), [`5cc3dd1`](https://github.com/TissUUmaps/TissUUmaps/commit/5cc3dd1875e84e8d88dabdb671cb8eb8039f8f29), [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5), [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5), [`b774531`](https://github.com/TissUUmaps/TissUUmaps/commit/b7745316a23f68c84d57d8cf51f2b914258f74ec), [`0fb9cf3`](https://github.com/TissUUmaps/TissUUmaps/commit/0fb9cf37a30712d732862cf7d89ca1626cef88b3), [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5), [`ace6c9b`](https://github.com/TissUUmaps/TissUUmaps/commit/ace6c9be6b459135e25bd2afd11ecb5cc58712a5), [`cd2e922`](https://github.com/TissUUmaps/TissUUmaps/commit/cd2e922bdf98a4713c6a28940dbc31cf6d53fb6b), [`cd2e922`](https://github.com/TissUUmaps/TissUUmaps/commit/cd2e922bdf98a4713c6a28940dbc31cf6d53fb6b), [`ddd7fac`](https://github.com/TissUUmaps/TissUUmaps/commit/ddd7fac9046e0226a7225aa7621ad1a759ffa35a), [`a99d552`](https://github.com/TissUUmaps/TissUUmaps/commit/a99d552103ae8906467214c71d94a480e72a4aa2), [`6464a42`](https://github.com/TissUUmaps/TissUUmaps/commit/6464a4217f7cd3636a8040dbf2f5f55151b180b2), [`575da90`](https://github.com/TissUUmaps/TissUUmaps/commit/575da90af95d3d17fbf8a8c2bd64e43204a2bc8a), [`fa7e975`](https://github.com/TissUUmaps/TissUUmaps/commit/fa7e97576d6748d9d1cb1cf32f51ef950587415a), [`fa7e975`](https://github.com/TissUUmaps/TissUUmaps/commit/fa7e97576d6748d9d1cb1cf32f51ef950587415a), [`67de073`](https://github.com/TissUUmaps/TissUUmaps/commit/67de073b7d3e5a55e276ca7035dc057c8a643f59), [`02a0b23`](https://github.com/TissUUmaps/TissUUmaps/commit/02a0b23526bc424580dd1abcd33b62abb115c24d), [`b091137`](https://github.com/TissUUmaps/TissUUmaps/commit/b091137fa80e5c18dd79e12b835a966e1d8bfc13), [`d6720ba`](https://github.com/TissUUmaps/TissUUmaps/commit/d6720ba27b069f73892a94a07fc3293e94c3faeb)]:
  - @tissuumaps/core@0.1.0-beta.4
  - @tissuumaps/render@0.1.0-beta.4

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
