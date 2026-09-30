# @tissuumaps/react

The [TissUUmaps](https://github.com/TissUUmaps/TissUUmaps) `Viewer` React
component. It renders through `@tissuumaps/render` and is decoupled from
application state management by the `ViewerAdapter` interface.

## Installation

```sh
npm install @tissuumaps/react
```

Its peer dependencies `@tissuumaps/core`, `@tissuumaps/render`, `react` and
`react-dom`, and `openseadragon` for `@tissuumaps/render`, are installed
automatically by npm 7+, pnpm and Bun; with Yarn, install them yourself. Its
peer dependency `@types/react` is optional: install it if you use TypeScript.

Under a Content Security Policy, allow `img-src data:` for the marker atlas of
`@tissuumaps/render`'s WebGL points renderer.

## Documentation

- [API reference](https://tissuumaps.github.io/TissUUmaps/docs/docs/api/@tissuumaps/react/)
- [Code architecture](https://tissuumaps.github.io/TissUUmaps/docs/docs/development/code-architecture/)
