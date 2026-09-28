# @tissuumaps/core

Core of [TissUUmaps](https://github.com/TissUUmaps/TissUUmaps): the data
model, the abstract data provider interfaces, the types shared between the
TissUUmaps packages and plugins (stores, plugin contract, OpenSeadragon and
WebGL options), and utilities.

## Installation

```sh
npm install @tissuumaps/core
```

Its peer dependencies `@jsonforms/core`, `openseadragon` and `zustand` are
optional: they are only referenced by its type declarations, so install them
if you use TypeScript. Without them, the types that reference them (e.g. the
stores and the data provider interfaces) silently degrade to `any` when
`skipLibCheck` is enabled, and fail to resolve otherwise.

## Documentation

- [API reference](https://tissuumaps.github.io/TissUUmaps/docs/docs/api/@tissuumaps/core/)
- [Data model](https://tissuumaps.github.io/TissUUmaps/docs/docs/concepts/data-model/)
- [Code architecture](https://tissuumaps.github.io/TissUUmaps/docs/docs/development/code-architecture/)
