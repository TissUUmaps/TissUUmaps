---
sidebar_position: 8
---

# HDF5

The built-in **HDF5 data provider** opens [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files as **tables**, including [AnnData](https://anndata.readthedocs.io/) `.h5ad` files. Files are read in the browser with [h5wasm](https://github.com/usnistgov/h5wasm), in a worker that lives as long as the table; a remote file is read through HTTP range requests.

## Data source

HDF5 data sources have the `type` `"hdf5"` and accept the following fields:

| Field        | Type     | Description                                                                                            |
| ------------ | -------- | ------------------------------------------------------------------------------------------------------ |
| `type`       | `string` | Always `"hdf5"`.                                                                                       |
| `source`     | `string` | URL or path of the HDF5 file (see [Referencing data](../concepts/projects.md#referencing-data)).       |
| `idColumn`   | `string` | Column of the row IDs, integers or strings without missing values. Sequential IDs are used if omitted. |
| `nameColumn` | `string` | Column of the row names.                                                                               |

## Tables

Any HDF5 file is read as a table of its datasets. A column is addressed by the path of a dataset within the file, for example `obs/area`. Column inputs offer autocompletion: an empty query lists the root of the file, and a query ending in `/` lists the children of that group. Paths are matched exactly, or ignoring case if that matches a single column.

- One-dimensional datasets are columns.
- Two-dimensional datasets expose one column per matrix column as `path[i]`, for example `obsm/spatial[0]` and `obsm/spatial[1]` for spatial coordinates.

## Profiles

One profile is detected from the file's attributes:

### AnnData

Groups with an AnnData `encoding-type` attribute are decoded wherever they are in the file:

- Sparse matrices expose one column per matrix column, as two-dimensional datasets do.
- `categorical` groups are columns of their category labels; missing values are empty strings for string categories and `NaN` for numeric ones.
- `nullable-integer` and `nullable-boolean` groups are numeric columns; missing values are `NaN`.
- `nullable-string-array` groups are string columns; missing values are empty strings.

A group whose `encoding-type` is `anndata` is an [AnnData object](https://anndata.readthedocs.io/). A file may hold only one. Its `obs` index gives the number of rows. Its `X`, `layers` and their `raw` counterparts are addressed by variable name as well as by index, for example `X[CD3]` besides `X[12]`. The names are the index of the `var` dataframe of the object, and `raw/var` for `raw/X`. Names are matched exactly, or ignoring case if that matches a single name. A number between the brackets is always an index, so numeric, duplicate and empty names are addressed by index.

## Example

A project showing the cells of an AnnData file as points, placed by their spatial coordinates and colored by cell type:

```json title="project.tm4"
{
  "layers": [{ "id": "layer", "name": "Sample" }],
  "tables": [
    {
      "id": "cells-table",
      "name": "Cells",
      "dataSource": { "type": "hdf5", "source": "cells.h5ad" }
    }
  ],
  "points": [
    {
      "id": "cells",
      "name": "Cells",
      "layer": "layer",
      "dataSource": {
        "type": "table",
        "table": "cells-table",
        "x": "obsm/spatial[0]",
        "y": "obsm/spatial[1]"
      },
      "pointColor": { "groupBy": { "column": "obs/cell_type" } }
    }
  ]
}
```

The table has one row per cell, as the `obs` index gives the number of rows. Any other column can be used the same way, such as `obs/total_counts`, or `X[CD3]` if `X` is stored dense or as CSC.

## Limitations

- Sparse matrices must be stored in CSC format (`csc_matrix`). Reading one column of a CSR matrix would require the whole matrix, so CSR matrices are listed but cannot be read.
- 64-bit integer columns are read as numbers; a value beyond ±2⁵³ is rejected rather than rounded.
- HDF5 files do not store a row count. The number of rows is taken from the ID column if given, otherwise from the name column, otherwise from the AnnData `obs` index, otherwise from the first column of the file.
- Variable names are only read for AnnData objects. A `var` dataframe whose index is as long as the matrix is required; otherwise the matrix keeps its column indices.
- Remote files are read through HTTP range requests. The server must send `Accept-Ranges: bytes` and, for cross-origin requests, expose it via `Access-Control-Expose-Headers`; otherwise the whole file is downloaded before the first read. The parts of the file read so far stay in memory while the table is open.
- A column must have as many rows as the table.
- Scalars, arrays of more than two dimensions, compound datasets and nodes whose name contains a bracket are skipped.
- Files written by anndata before 0.8 carry no `encoding-type` attributes and are read as plain HDF5 files: their `obs` and `var` are compound datasets, which are skipped, and their sparse `X` is read as its `data`, `indices` and `indptr` arrays. Rewrite them with a current anndata version.

## API

The data provider is implemented in the [`@tissuumaps/storage`](../api/@tissuumaps/storage/index.md) package as [`HDF5TableDataProvider`](../api/@tissuumaps/storage/classes/HDF5TableDataProvider.md). Reading the file is delegated to [h5wasm](https://github.com/usnistgov/h5wasm), in a Web Worker (see [Dependencies](../development/dependencies.md)).
