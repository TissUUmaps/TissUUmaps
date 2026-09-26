---
sidebar_position: 2
---

# TIFF

The built-in **TIFF data provider** opens OME-TIFF, QPTIFF and plain TIFF files as **images** and as **labels**. Tiles are decoded in the browser with [geotiff.js](https://geotiffjs.github.io/), so no server-side tiling is needed; remote files need a server that supports HTTP range requests.

## Data source

TIFF data sources have the `type` `"tiff"` and accept the following fields:

| Field    | Type      | Description                                                                                         |
| -------- | --------- | --------------------------------------------------------------------------------------------------- |
| `type`   | `string`  | Always `"tiff"`.                                                                                    |
| `source` | `string`  | URL or path of the TIFF file (see [Referencing data](../concepts/projects.md#referencing-data)).    |
| `z`      | `integer` | Z-slice to open (0-based), for OME-TIFF files with a z-stack. Defaults to `0`.                      |
| `t`      | `integer` | Timepoint to open (0-based), for OME-TIFF files with a time series. Defaults to `0`.                |
| `table`  | `string`  | _Labels only._ ID of the table annotating the labels (see [Data model](../concepts/data-model.md)). |

## Profiles

The profile of a file is detected from its metadata. It determines how the pyramid and the channels of the file are read, for images and labels alike:

### OME-TIFF

The OME-XML in the file describes the channels. `z` and `t` select the plane. Pyramid levels are read from SubIFDs. Files with several images show the largest one.

### QPTIFF

Each IFD has an XML description with the channel name and color. Thumbnail, overview and label images are skipped.

### Plain TIFF

Any other file: the largest IFDs are the channels, smaller IFDs with the same aspect ratio are the pyramid levels. SubIFDs are used when present.

Files without a pyramid open too, but slowly.

## Images

### Channels

Multi-channel files are opened as **value images**: tiles carry the raw samples of the file, which are contrast-stretched and colorized at display time. TissUUmaps renders every channel itself, stretching it between its contrast limits, multiplying it with its color, and blending the channels additively. Files that carry their own colors (RGB, YCbCr, palette and white-is-zero) are drawn as they are.

Per-channel rendering settings come from the file where it records them:

| Setting         | Source                                    | Fallback                      |
| --------------- | ----------------------------------------- | ----------------------------- |
| Name            | OME-XML `Channel` `Name`, QPTIFF `Name`   | none                          |
| Color           | OME-XML `Channel` `Color`, QPTIFF `Color` | none                          |
| Contrast limits | not recorded by TIFF                      | `[0, 255]` for 8-bit channels |
| Visibility      | not recorded by TIFF                      | none                          |

Settings without a fallback are reported as unset, leaving the renderer's defaults to apply: channels without a visibility are shown, channels without a color are colorized with a color derived from the channel index (white for single-channel images), and channels without contrast limits are stretched between quantile-based limits derived from their histogram (see [Rendering](../development/rendering.md#images)).

All of these can be overridden per channel in the project file through the image's `channels` array (see the [example](#example) below).

## Labels

TIFF files can also be opened as **labels**, where every pixel value is a label (segment) ID and `0` is background. The [profile](#profiles) and the pyramid are read as for images, with two restrictions: the file has to hold a **single channel** (an RGB file, or a multi-channel file, is rejected), and its samples have to be **integers of at most 32 bits** (signed or unsigned).

Pyramid levels of a label mask have to be downsampled with nearest-neighbor sampling. Averaging or interpolating label IDs invents IDs that are not in the mask, and those show up as wrong labels at lower zoom levels.

Label IDs are read per tile as the tiles are drawn, so arbitrarily large label masks can be opened without scanning them first. TIFF records no label colors; they are configured in the project file instead, optionally through an annotating `table`.

## Example

A project showing a multiplexed OME-TIFF with a segmentation on top of it, where the segmentation is annotated by a CSV table and colored by one of its columns:

```json title="project.tmap"
{
  "name": "TIFF example",
  "layers": [{ "id": "layer", "name": "Sample" }],
  "images": [
    {
      "id": "image",
      "name": "Multiplexed image",
      "layer": "layer",
      "dataSource": {
        "type": "tiff",
        "source": "images/sample.ome.tif",
        "z": 4
      },
      "channels": [
        { "name": "DAPI", "color": { "r": 0, "g": 0, "b": 255 } },
        { "name": "CD3", "contrastLimits": [100, 4000] },
        { "visibility": false }
      ]
    }
  ],
  "labels": [
    {
      "id": "cells",
      "name": "Cell segmentation",
      "layer": "layer",
      "dataSource": {
        "type": "tiff",
        "source": "labels/cells.ome.tif",
        "z": 4,
        "table": "cell-table"
      },
      "labelColor": { "from": { "column": "area", "palette": "batlow" } }
    }
  ],
  "tables": [
    {
      "id": "cell-table",
      "name": "Cells",
      "dataSource": { "type": "csv", "source": "tables/cells.csv" }
    }
  ]
}
```

Both the image and the labels are TIFF files served next to the project file, and both open the same z-slice. Channel settings that are left out fall back to the file's metadata and to the estimated contrast limits.

## Limitations

- JPEG 2000 compression is not supported.
- Multi-file OME-TIFF is not supported; the planes have to be in the file that is opened.
- Files with several palette or white-is-zero images are rejected; a single one is drawn in its own colors.
- RGB, multi-channel and floating point files are not opened as labels. Convert them to a single integer channel of at most 32 bits.

## API

The data provider is implemented in the [`@tissuumaps/storage`](/docs/api/@tissuumaps/storage) package as [`TIFFImageDataProvider`](/docs/api/@tissuumaps/storage/classes/TIFFImageDataProvider) and [`TIFFLabelsDataProvider`](/docs/api/@tissuumaps/storage/classes/TIFFLabelsDataProvider). Reading the file is delegated to [geotiff.js](https://geotiffjs.github.io/), and tiling to our fork of [GeoTIFFTileSource](https://github.com/TissUUmaps/GeoTIFFTileSource) (see [Dependencies](../development/dependencies.md)).
