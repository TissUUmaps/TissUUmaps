---
sidebar_position: 9
---

# Zarr

The built-in **Zarr data provider** opens [Zarr](https://zarr.dev/) stores as **tables**, including the AnnData tables of [SpatialData](https://spatialdata.scverse.org/) stores. Stores are read in the browser with [zarrita](https://github.com/manzt/zarrita.js); reads are fetches, so no worker is needed. Images of a Zarr store are read by the [OME-Zarr](./ome-zarr.md) data provider instead.

## Data source

Zarr data sources have the `type` `"zarr"` and accept the following fields:

| Field        | Type     | Description                                                                                                                     |
| ------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `type`       | `string` | Always `"zarr"`.                                                                                                                |
| `source`     | `string` | URL of the store, or path of its directory in the workspace (see [Referencing data](../concepts/projects.md#referencing-data)). |
| `idColumn`   | `string` | Column of the row IDs, integers or strings without missing values. Sequential IDs are used if omitted.                          |
| `nameColumn` | `string` | Column of the row names.                                                                                                        |

The source may point at the store itself, or at a group inside it. A SpatialData table is a group of the store, so both of these work if the store holds one table:

- `https://example.org/visium.zarr/tables/adata`, whose columns are `obs/area`, `obsm/spatial[0]`, and so on.
- `https://example.org/visium.zarr`, whose columns are `tables/adata/obs/area`, and so on.

A store holding several tables must be opened at one of them. Consolidated metadata is written at the root of a store, so it is looked up at the source and then at each of its ancestors.

## Tables

Any Zarr store with consolidated metadata is read as a table of its arrays. Columns are addressed as in [HDF5](./hdf5.md#tables) files: a column is the path of an array within the store, and one column of a matrix is `path[i]`, or `path[name]` for an AnnData expression matrix.

## Profiles

One profile is detected from the store's attributes:

### AnnData

AnnData objects are recognized and decoded as in [HDF5](./hdf5.md#anndata) files, whether the source points at one or at a store containing them.

## Example

A project showing the spots of the table of a SpatialData store as points, placed by their spatial coordinates and colored by cluster:

```json title="project.tm4"
{
  "layers": [{ "id": "layer", "name": "Visium" }],
  "tables": [
    {
      "id": "spots-table",
      "name": "Spots",
      "dataSource": { "type": "zarr", "source": "visium.zarr/tables/adata" }
    }
  ],
  "points": [
    {
      "id": "spots",
      "name": "Spots",
      "layer": "layer",
      "dataSource": {
        "type": "table",
        "table": "spots-table",
        "x": "obsm/spatial[0]",
        "y": "obsm/spatial[1]"
      },
      "pointColor": { "groupBy": { "column": "obs/cluster" } }
    }
  ]
}
```

The source points at the table inside the store; its consolidated metadata is found at `visium.zarr`.

## Limitations

- The store must have consolidated metadata (in Zarr v2 `.zmetadata` or Zarr v3 `zarr.json`). A Zarr store is a key-value store, so without it the columns cannot be listed. SpatialData writes consolidated metadata.
- Zarr v3 arrays whose chunks are compressed inside shards (the `sharding_indexed` codec with an inner compressor, which zarr-python 3 and anndata write by default) cannot be read yet. Write them without sharding (`shards=None`) or as Zarr v2, which SpatialData does.
- Nodes whose metadata cannot be read are skipped instead of failing the store. AnnData writes a few of them under `uns`.
- Zipped stores are not supported.
- The [HDF5 limitations](./hdf5.md#limitations) on sparse matrices, 64-bit integers, row counts, variable names and column lengths apply as well. AnnData writes `X` as CSR by default.

## API

The data provider is implemented in the [`@tissuumaps/storage`](../api/@tissuumaps/storage/index.md) package as [`ZarrTableDataProvider`](../api/@tissuumaps/storage/classes/ZarrTableDataProvider.md). Reading the store is delegated to [zarrita](https://github.com/manzt/zarrita.js).
