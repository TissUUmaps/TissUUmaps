---
sidebar_position: 1
---

# Image

The built-in **Image data provider** opens **images** that the browser or [OpenSeadragon](https://openseadragon.github.io/) reads on its own: plain PNG, JPEG and WebP images, and tile sources that OpenSeadragon reads, such as Deep Zoom (DZI), IIIF, Zoomify, OSM and TMS tiles or legacy image pyramids. This is a virtual format: the provider reads nothing itself and hands the image or tile source to the viewer.

## Data source

Image data sources have the `type` `"image"` and accept the following fields:

| Field        | Type               | Description                                                                                                                                                                                                                             |
| ------------ | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`       | `string`           | Always `"image"`.                                                                                                                                                                                                                       |
| `source`     | `string`           | URL or path of a plain image file or of a tile source descriptor, such as a DZI file or a IIIF `info.json` (see [Referencing data](../concepts/projects.md#referencing-data)).                                                          |
| `tileSource` | `string \| object` | An OpenSeadragon [tile source](https://openseadragon.github.io/docs/OpenSeadragon.TileSource.html): the URL or path of a tile source descriptor, or an inline tile source configuration (e.g. `{ "type": "zoomifytileservice", ... }`). |

At least one of `source` and `tileSource` is given. If both are, `tileSource` takes precedence and `source` is ignored.

A `source` is opened as a plain image if its file name has a PNG, JPEG or WebP extension, or if it is a `data:image/...` URL. Any other `source` is handed to OpenSeadragon as the URL of a tile source descriptor. To open a descriptor whose URL has an image extension, give it as `tileSource` instead. To open an image in another format the browser decodes, such as GIF, or one whose URL has no image extension, give it as an inline `tileSource` of type `image` (`{ "type": "image", "url": "..." }`).

## Images

A plain image is shown as a single tile at its full resolution. Tiles of a tile source are drawn as the tile source serves them. These images have no channels, so the project file's `channels` array does not apply to them.

## Limitations

- A plain image is loaded in full, without a pyramid. Use a tile source for large images.
- A remote plain image is requested with CORS, so its server has to send CORS headers.
- Tile source descriptors are fetched by OpenSeadragon, so they cannot be opened from the workspace; only plain images can.
- An inline tile source configuration is handed to OpenSeadragon as is: its URLs are not resolved like a `source` or a `tileSource` URL (see [Referencing data](../concepts/projects.md#referencing-data)), so relative URLs are resolved against the TissUUmaps page, and workspace paths are not supported.

## API

The data provider is implemented in the [`@tissuumaps/storage`](../api/@tissuumaps/storage/index.md) package as [`OpenSeadragonImageDataProvider`](../api/@tissuumaps/storage/classes/OpenSeadragonImageDataProvider.md).
