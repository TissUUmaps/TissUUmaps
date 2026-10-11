# @tissuumaps/storage

Official data provider distribution for
[TissUUmaps](https://github.com/TissUUmaps/TissUUmaps): implementations of the
data provider interfaces of `@tissuumaps/core` for images, labels, points,
shapes and tables.

## Installation

```sh
npm install @tissuumaps/storage
```

Its peer dependencies `@tissuumaps/core`, `openseadragon`, `ome-zarr.js` and
`zarrita` are installed automatically by npm 7+, pnpm and Bun; with Yarn,
install them yourself.

## Entry points

Each format has its own entry point, so that only the providers you use are
bundled; the package root exports all of them.

| Entry point                         | Data providers                                                                                        |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `@tissuumaps/storage/csv`           | [CSV](https://tissuumaps.github.io/TissUUmaps/docs/docs/supported-data/csv/) tables                   |
| `@tissuumaps/storage/geojson`       | [GeoJSON](https://tissuumaps.github.io/TissUUmaps/docs/docs/supported-data/geojson/) shapes           |
| `@tissuumaps/storage/hdf5`          | [HDF5](https://tissuumaps.github.io/TissUUmaps/docs/docs/supported-data/hdf5/) tables                 |
| `@tissuumaps/storage/ome-zarr`      | [OME-Zarr](https://tissuumaps.github.io/TissUUmaps/docs/docs/supported-data/ome-zarr/) images, labels |
| `@tissuumaps/storage/openseadragon` | OpenSeadragon [images](https://tissuumaps.github.io/TissUUmaps/docs/docs/supported-data/image/)       |
| `@tissuumaps/storage/parquet`       | [Parquet](https://tissuumaps.github.io/TissUUmaps/docs/docs/supported-data/parquet/) shapes, tables   |
| `@tissuumaps/storage/table`         | [points from tables](https://tissuumaps.github.io/TissUUmaps/docs/docs/supported-data/table/)         |
| `@tissuumaps/storage/tiff`          | [TIFF](https://tissuumaps.github.io/TissUUmaps/docs/docs/supported-data/tiff/) images, labels         |
| `@tissuumaps/storage/zarr`          | [Zarr](https://tissuumaps.github.io/TissUUmaps/docs/docs/supported-data/zarr/) tables                 |

The CSV, GeoJSON, HDF5 and Parquet providers parse, and the TIFF providers
decode, in web workers loaded from `blob:` URLs. Under a Content Security Policy,
allow them with `worker-src blob:`. The HDF5 provider, and the Parquet and TIFF
providers for some compressions, run WebAssembly in these workers; allow it with
`script-src 'wasm-unsafe-eval'`.

## Documentation

- [API reference](https://tissuumaps.github.io/TissUUmaps/docs/docs/api/@tissuumaps/storage/)
- [Supported data](https://tissuumaps.github.io/TissUUmaps/docs/docs/supported-data/)

## Known issues

### geotiff needs to be patched

geotiff 3.0.5 misreads deferred tag values of big-endian TIFFs, and corrupts
LZW-compressed tiles that it decodes on the main thread. The fixes are merged
upstream but not yet released. Until they are, apply
[`geotiff@3.0.5.patch`](https://github.com/TissUUmaps/TissUUmaps/blob/8a972b9de0b86f0ace08ca41541a746ee2405022/patches/geotiff@3.0.5.patch)
to geotiff 3.0.5 in your project. This package depends on exactly that
version, so the patch applies as long as your project does not override it.
With pnpm, save it as
`patches/geotiff@3.0.5.patch` and register it in `pnpm-workspace.yaml`:

```yaml
patchedDependencies:
  geotiff@3.0.5: patches/geotiff@3.0.5.patch
```

The patch is a git diff relative to the geotiff package root, so it can also
be applied with other package managers' patching mechanisms. See
[Dependencies](https://tissuumaps.github.io/TissUUmaps/docs/docs/development/dependencies/)
for details.
