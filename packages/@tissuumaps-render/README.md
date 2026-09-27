# @tissuumaps/render

Rendering backends of [TissUUmaps](https://github.com/TissUUmaps/TissUUmaps4),
exposed as an imperative API: OpenSeadragon for images and labels, WebGL 2 for
points and shapes, and an SVG overlay for interactive shape drawing. It does
not depend on React; see `@tissuumaps/viewer` for a React component built on
top of it.

## Installation

```sh
npm install @tissuumaps/render
```

Its peer dependencies `@tissuumaps/core` and `openseadragon` are installed
automatically by npm 7+, pnpm and Bun; with Yarn, install them yourself.

The WebGL points renderer loads its marker atlas from a `data:` URL. Under a
Content Security Policy, allow it with `img-src data:`.

## Documentation

- [API reference](https://tissuumaps.github.io/TissUUmaps4/docs/docs/api/@tissuumaps/render/)
- [Rendering](https://tissuumaps.github.io/TissUUmaps4/docs/docs/development/rendering/)
- [Code architecture](https://tissuumaps.github.io/TissUUmaps4/docs/docs/development/code-architecture/)
