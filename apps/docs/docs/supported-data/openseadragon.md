---
sidebar_position: 1
---

# OpenSeadragon

The built-in **OpenSeadragon data provider** opens **images** from any tile source that [OpenSeadragon](https://openseadragon.github.io/) reads on its own: Deep Zoom (DZI), IIIF, Zoomify, OSM and TMS tiles, legacy image pyramids and plain images. Tile sources with a descriptor file are referenced by `source`, the others are configured inline with `tileSourceConfig`. This is a virtual format: the provider reads nothing itself and hands the source to the viewer.

## Data source

OpenSeadragon data sources have the `type` `"openseadragon"` and accept the following fields:

| Field              | Type     | Description                                                                                                                                                                                                         |
| ------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`             | `string` | Always `"openseadragon"`.                                                                                                                                                                                           |
| `source`           | `string` | URL or path of the tile source descriptor, such as a DZI file or a IIIF `info.json` (see [Referencing data](../concepts/projects.md#referencing-data)).                                                             |
| `tileSourceConfig` | `object` | An inline [tile source configuration](https://openseadragon.github.io/docs/OpenSeadragon.TileSource.html), for tile sources without a descriptor file, such as a plain image (`{ "type": "image", "url": "..." }`). |

Exactly one of `source` and `tileSourceConfig` is given.

## Images

Tiles are drawn as the tile source serves them. OpenSeadragon images have no channels, so the project file's `channels` array does not apply to them.

## Limitations

- A `source` is always a descriptor that OpenSeadragon fetches and parses. A plain image cannot be given as a `source`; configure it as a `tileSourceConfig` of type `image` instead.
- A `source` in the open workspace is opened through an object URL, so a descriptor in the workspace cannot find tile files stored next to it (such as the tiles of a DZI).

## API

The data provider is implemented in the [`@tissuumaps/storage`](/docs/api/@tissuumaps/storage) package as [`OpenSeadragonImageDataProvider`](/docs/api/@tissuumaps/storage/classes/OpenSeadragonImageDataProvider).
