---
sidebar_position: 3
---

# OME-Zarr

The built-in **OME-Zarr data provider** opens [OME-NGFF](https://ngff.openmicroscopy.org/) images stored in [Zarr](https://zarr.dev/) format as **images** and as **labels**. It reads OME-Zarr versions 0.1 through 0.5 (Zarr v2 and v3) and builds an OpenSeadragon tile source directly on the multiscales pyramid, so no server-side tiling is needed.

The provider is named after OME-Zarr rather than Zarr: a Zarr array without the `multiscales` metadata is not an image, so there is no plain Zarr image provider.

## Data source

OME-Zarr data sources have the `type` `"ome-zarr"` and accept the following fields:

| Field    | Type      | Description                                                                                                                                                         |
| -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`   | `string`  | Always `"ome-zarr"`.                                                                                                                                                |
| `source` | `string`  | URL or path of the OME-Zarr image (see [Referencing data](../concepts/projects.md#referencing-data)).                                                               |
| `z`      | `integer` | Z-slice to open (0-based), for images with a `z` axis. Defaults to the `defaultZ` of the image's `omero` metadata, or to the middle of the axis if there is none.   |
| `t`      | `integer` | Timepoint to open (0-based), for images with a `t` axis. Defaults to the `defaultT` of the image's `omero` metadata, or to the middle of the axis if there is none. |
| `table`  | `string`  | _Labels only._ ID of the table annotating the labels (see [Data model](../concepts/data-model.md)).                                                                 |

A URL `source` refers to a remote OME-Zarr store, or to a remote zipped OME-Zarr file if it ends in `.ozx`. A `source` that resolves to a directory in the open workspace is opened as an OME-Zarr store, and one that resolves to a file as a zipped OME-Zarr file.

The `source` has to point to a **multiscales image**, i.e. the Zarr group holding the `multiscales` metadata. Plate (HCS) groups and `bioformats2raw.layout` groups are not opened directly; point to one of the images they contain instead.

## Images

OME-Zarr images are opened as **value images**: tiles carry the raw sample values of the Zarr array, which are contrast-stretched and colorized at display time rather than served as pre-rendered pixels. Integer arrays of up to 32 bits and floating point arrays are supported; 64-bit integer arrays are not.

Images with more than two spatial dimensions are shown one **plane** at a time, selected by `z` and `t`.

### Channels

Images **with a channel axis**, even one holding a single channel, are opened as multi-channel image data with one tile source per channel. TissUUmaps renders every channel itself, stretching it between its contrast limits, multiplying it with its color, and blending the channels additively.

Per-channel rendering settings are taken from the image's `omero` metadata where present:

| Setting         | `omero` source                        | Fallback                      |
| --------------- | ------------------------------------- | ----------------------------- |
| Name            | `channels[c].label`                   | none                          |
| Visibility      | `channels[c].active`                  | none                          |
| Color           | `channels[c].color` (6-digit hex)     | none                          |
| Contrast limits | `channels[c].window.start` and `.end` | `[0, 255]` for `uint8` arrays |

Settings without a fallback are reported as unset, leaving the renderer's defaults to apply: channels without a visibility are shown, channels without a color are colorized with a color derived from the channel index (white for single-channel images), and channels without contrast limits are stretched between quantile-based limits derived from their histogram (see [Rendering](../development/rendering.md#images)).

All of these can be overridden per channel in the project file through the image's `channels` array (see the [example](#example) below).

The channel histograms are computed once when the image is loaded, from the plane being shown at a downsampled resolution level (the lowest level that still holds at least 512 × 512 pixels, or the full-resolution level if no level does), so that channels without an `omero` window are stretched to their actual value distribution rather than to the full range of the array's data type (see [Rendering](../development/rendering.md#images)). Loading a multi-channel image therefore reads one downsampled plane per channel up front.

Images **without a channel axis** are opened as single-channel image data with one tile source. Their tiles are rendered by the tile source itself, which applies the color and window of the `omero` channel (falling back to white and the data type's value range). The project file's `channels` array does not apply to single-channel images.

## Labels

OME-Zarr label images are opened as **labels**, where every pixel value is a label (segment) ID and `0` is background. The label array has to hold **integers of up to 32 bits** (`int8`, `int16`, `int32`, `uint8`, `uint16` or `uint32`); 64-bit and floating-point arrays are rejected.

Label IDs are read per tile as the tiles are drawn, so arbitrarily large label images can be opened without scanning them first. The `image-label` metadata of the group (color and property tables) is not used; label colors are configured in the project file instead, optionally through an annotating `table`.

A label image with a channel axis is opened on its first channel. As for images, `z` and `t` select the plane.

## Example

A project showing a multi-channel OME-Zarr image with a segmentation on top of it, where the segmentation is annotated by a CSV table and colored by one of its columns:

```json title="project.tmap"
{
  "name": "OME-Zarr example",
  "layers": [{ "id": "layer", "name": "Sample" }],
  "images": [
    {
      "id": "image",
      "name": "Multiplexed image",
      "layer": "layer",
      "dataSource": {
        "type": "ome-zarr",
        "source": "images/sample.ome.zarr",
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
        "type": "ome-zarr",
        "source": "labels/cells.ome.zarr",
        "z": 4,
        "table": "cell-table"
      },
      "labelColor": { "type": "from", "from": { "column": "cell_type" } }
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

Both the image and the labels are OME-Zarr images served next to the project file, and both open the same z-slice. Channel settings that are left out fall back to the image's `omero` metadata.

## Limitations

- 64-bit integer image arrays and 64-bit or floating-point label arrays are not supported. Convert them to a narrower integer type when writing the OME-Zarr.
- Plate and `bioformats2raw.layout` groups have to be referenced by one of their contained images.
- `image-label` metadata is not read.

## API

The data provider is implemented in the [`@tissuumaps/storage`](/docs/api/@tissuumaps/storage) package as [`OMEZarrImageDataProvider`](/docs/api/@tissuumaps/storage/classes/OMEZarrImageDataProvider) and [`OMEZarrLabelsDataProvider`](/docs/api/@tissuumaps/storage/classes/OMEZarrLabelsDataProvider). Reading OME-Zarr metadata and arrays and tiling them are delegated to [omezarr-tilesource](https://www.npmjs.com/package/omezarr-tilesource), which builds on [ome-zarr.js](https://github.com/biongff/ome-zarr.js) and [zarrita](https://github.com/manzt/zarrita.js).
