---
sidebar_position: 5
---

# Dependencies

## Development stack

- pnpm
- Vite
- TypeScript
- ESLint (linting)
- Prettier + import-sort plugin (formatting)
- Vitest + jsdom + node-canvas (testing with coverage)
- API Extractor + unplugin-dts (type declaration rollups for packages)
- vite-plugin-singlefile (single-file production build of the application)
- Docusaurus + TypeDoc + GitHub Pages (documentation)
- Husky + lint-staged (pre-commit hooks)
- GitHub Actions (CI/CD)

## Core dependencies

- React (frontend library)
- Zustand + Immer (state management with immutable updates)
- Tailwind CSS (CSS framework)
- Dockview (docking layout manager)
- Base UI (headless component library) with shadcn/ui-generated wrappers vendored into `components/ui` (`components.json`) + Lucide (icon component library)
- OpenSeadragon (zoomable image and labels rendering)
- JSON Forms (JSON Schema-based form renderer)
- TanStack Table + TanStack Virtual (virtualized data tables)
- react-colorful (color picker)

## Data loading

- Hyparquet + hyparquet-compressors (Parquet tables; bundled into the Parquet worker)
- h5wasm (HDF5 and AnnData tables; bundled into the HDF5 worker)
- zarrita.js (Zarr and AnnData tables)
- PapaParse (CSV tables)
- omezarr-tilesource (OME-Zarr images/labels)
- geotiff.js + geotiff-tilesource (TIFF images, see below)

## Utilities

- d3-scale-chromatic + d3-color (built-in color palettes)
- dnd-kit (drag and drop interfaces)
- fast-equals (equality comparison)
- gl-matrix (WebGL matrix operations)
- class-variance-authority + clsx + tailwind-merge (class name composition)

## Patched and worked-around dependencies

Patches live in `patches/` and are applied by pnpm. Drop each once upstream
ships the fix.

- **geotiff 3.0.5**: the LZW dictionary is one code short
  ([#546](https://github.com/geotiffjs/geotiff.js/pull/546)), and deferred tag
  arrays are read with the wrong byte order
  ([#536](https://github.com/geotiffjs/geotiff.js/pull/536)). Both fixes are
  merged upstream, and a release has been requested. Tiles are decoded in our
  own worker (`tiff.worker.ts`) so that it uses the patched code too.

A patch is applied by this workspace's `pnpm install`, not by installing the
packages we publish. `@tissuumaps/storage` leaves `geotiff` external, so a
project that depends on the published package resolves its own unpatched copy
for everything outside the inlined worker: big-endian TIFFs then read deferred
tag values wrong, and LZW-compressed tiles decoded on the main thread corrupt
past 4093 dictionary codes. Until both fixes are released upstream, such a
project has to apply `patches/geotiff@3.0.5.patch` itself (see the
`@tissuumaps/storage` README). The app in this repository is unaffected.

`@tissuumaps/storage` pins `geotiff` to exactly 3.0.5, so that such a project
resolves the version the patch applies to, and the version bundled into the
TIFF worker, which has to speak the message protocol of the main thread's
`Pool`. Restore a caret range once the patch is dropped.

Bugs in peer dependencies cannot be patched for such projects, as they bring
their own copy. These are worked around at runtime instead:

- **OpenSeadragon 6.1.1**: `TileCache.injectCache` never counts the record it
  injects, so recoloring tiles lowers the cache's record count until eviction
  stops and the tile cache grows without bound. Fixed upstream
  ([#2960](https://github.com/openseadragon/openseadragon/pull/2960)) but not
  yet released. `OpenSeadragonUtils.fixTileCacheCounter` in
  `@tissuumaps/render` wraps `injectCache` to count the record, for 6.1.1 only.
  Remove it, and raise the `openseadragon` version ranges, once a release
  includes the fix.

## Forked dependencies

- **@zarrita/storage**: our fork of manzt/zarrita.js, in
  [TissUUmaps/zarrita.js](https://github.com/TissUUmaps/zarrita.js). We use its
  `FileSystemHandleStore` (`@zarrita/storage/fs-handle`) to open OME-Zarr
  images and Zarr tables from local directories. The fork ships TypeScript sources, which our
  strict type checking then covers, so import only that entry point.
- **geotiff-tilesource**: our fork of pearcetm/GeoTIFFTileSource, branch
  `tissuumaps` of
  [TissUUmaps/GeoTIFFTileSource](https://github.com/TissUUmaps/GeoTIFFTileSource).
  It carries several changes we need, each also submitted upstream. The package
  has no types, so `geotiff-tilesource.d.ts` declares what we use.

Both forks are git dependencies, which projects installing from npm may be
unable or unwilling to install. `@tissuumaps/storage` therefore bundles them
into its build and lists them as `devDependencies`, so the published package
has no git dependencies (see its `vite.config.ts`). Drop each fork, and
un-bundle it, once its changes are released upstream.

The commits are pinned in the `devDependencies` of `@tissuumaps/storage`. To
move one, change the hash there and run `pnpm install`.

## Web APIs (selection)

- File System (local data access)
- Web Workers (TIFF decoding, Parquet/GeoJSON/HDF5 parsing)
- WebGL 2 (points and shapes rendering)
