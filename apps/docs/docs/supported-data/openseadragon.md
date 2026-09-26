---
sidebar_position: 1
---

# OpenSeadragon

The built-in **OpenSeadragon data provider** opens **images** from any tile source that [OpenSeadragon](https://openseadragon.github.io/) reads on its own: Deep Zoom (DZI), IIIF, Zoomify, OSM and TMS tiles, legacy image pyramids and plain images. This is a virtual format: the provider reads nothing itself and hands the source to the viewer.

## Data source

OpenSeadragon data sources have the `type` `"openseadragon"` and accept the following fields:

| Field              | Type     | Description                                                                                                                                                       |
| ------------------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`             | `string` | Always `"openseadragon"`.                                                                                                                                         |
| `source`           | `string` | URL or path of the tile source, such as a DZI descriptor, a IIIF `info.json` or an image file (see [Referencing data](../concepts/projects.md#referencing-data)). |
| `tileSourceConfig` | `object` | An inline [tile source configuration](https://openseadragon.github.io/docs/OpenSeadragon.TileSource.html), for tile sources without a descriptor file.            |

Exactly one of `source` and `tileSourceConfig` is given.

## Images

Tiles are drawn as the tile source serves them. OpenSeadragon images have no channels, so the project file's `channels` array does not apply to them.

## Limitations

- A `source` in the open workspace is opened through an object URL. That works for single files such as plain images, but a DZI in the workspace cannot find its tile files.

## API

The data provider is implemented in the [`@tissuumaps/storage`](/docs/api/@tissuumaps/storage) package as [`OpenSeadragonImageDataProvider`](/docs/api/@tissuumaps/storage/classes/OpenSeadragonImageDataProvider).
