---
sidebar_position: 1
---

# OpenSeadragon

The built-in **OpenSeadragon data provider** opens **images** from any tile source that [OpenSeadragon](https://openseadragon.github.io/) reads on its own: Deep Zoom (DZI), IIIF, Zoomify, OSM and TMS tiles, legacy image pyramids and plain images. Tile sources with a descriptor file and plain images are referenced by `source`, the others are configured inline with `tileSourceConfig`. This is a virtual format: the provider reads nothing itself and hands the source to the viewer.

## Data source

OpenSeadragon data sources have the `type` `"openseadragon"` and accept the following fields:

| Field              | Type     | Description                                                                                                                                                                                        |
| ------------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`             | `string` | Always `"openseadragon"`.                                                                                                                                                                          |
| `source`           | `string` | URL or path of the tile source descriptor, such as a DZI file or a IIIF `info.json`, or of a plain image such as a JPEG or PNG (see [Referencing data](../concepts/projects.md#referencing-data)). |
| `tileSourceConfig` | `object` | An inline [tile source configuration](https://openseadragon.github.io/docs/OpenSeadragon.TileSource.html), for tile sources without a descriptor file, such as OSM or TMS tiles.                   |

Exactly one of `source` and `tileSourceConfig` is given.

## Images

Tiles are drawn as the tile source serves them. OpenSeadragon images have no channels, so the project file's `channels` array does not apply to them.

## Limitations

- A `source` URL is taken for a plain image by its file extension; a workspace file by its MIME type. Any other `source` is a descriptor that OpenSeadragon fetches and parses.
- A plain image is loaded whole, so large images are slow to open; convert them to a pyramidal format such as TIFF or OME-Zarr instead.
- A `source` in the open workspace is opened through an object URL, so a descriptor in the workspace cannot find tile files stored next to it (such as the tiles of a DZI).

## API

The data provider is implemented in the [`@tissuumaps/storage`](/docs/api/@tissuumaps/storage) package as [`OpenSeadragonImageDataProvider`](/docs/api/@tissuumaps/storage/classes/OpenSeadragonImageDataProvider).
