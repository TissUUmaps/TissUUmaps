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

## Patched dependencies

Patches live in `patches/` and are applied by pnpm. Drop each once upstream
ships the fix.

- **geotiff 3.0.5**: the LZW dictionary is one code short
  ([#546](https://github.com/geotiffjs/geotiff.js/pull/546)), and deferred tag
  arrays are read with the wrong byte order
  ([#536](https://github.com/geotiffjs/geotiff.js/pull/536)). Tiles are
  decoded in our own worker (`tiff.worker.ts`) so that it uses the patched
  code too.

A patch is applied by this workspace's `pnpm install`, not by installing the
packages we publish. `@tissuumaps/storage` leaves `geotiff` external, so a
project that depends on the published package resolves its own unpatched copy
for everything outside the inlined worker: big-endian TIFFs then read deferred
tag values wrong, and LZW-compressed tiles decoded on the main thread corrupt
past 4093 dictionary codes. Until both fixes are released upstream, such a
project has to apply `patches/geotiff@3.0.5.patch` itself. The app in this
repository is unaffected.

## Forked dependencies

- **geotiff-tilesource**: our fork of pearcetm/GeoTIFFTileSource, branch
  `tissuumaps` of
  [TissUUmaps/GeoTIFFTileSource](https://github.com/TissUUmaps/GeoTIFFTileSource).
  It carries several changes we need, each also submitted upstream. Drop the fork
  once they are released. The package has no types, so
  `geotiff-tilesource.d.ts` declares what we use.

The commit is pinned in `package.json`. To move it, change the hash there and
run `pnpm install`.

## Web APIs (selection)

- File System (local data access)
- Web Workers (TIFF decoding, Parquet/GeoJSON/HDF5 parsing)
- WebGL 2 (points and shapes rendering)
