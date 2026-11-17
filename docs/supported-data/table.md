---
sidebar_position: 4
---

# Table

The built-in **Table data provider** derives **points** from a table of the project: every row is a point, placed by two of the table's numeric columns. This is a virtual format: the provider reads no file of its own.

## Data source

Table data sources have the `type` `"table"` and accept the following fields:

| Field   | Type     | Description                                                                                                           |
| ------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| `type`  | `string` | Always `"table"`.                                                                                                     |
| `table` | `string` | ID of the table holding the points (see [Data model](../concepts/data-model.md)).                                     |
| `x`     | `string` | Column holding the x coordinate of each point. Defaults to `"x"` (see [Coordinate detection](#coordinate-detection)). |
| `y`     | `string` | Column holding the y coordinate of each point. Defaults to `"y"` (see [Coordinate detection](#coordinate-detection)). |

## Points

The points take the IDs and names of the table's rows, and the table annotates them: every column of the table can color or size them. The coordinate columns are read as 32-bit floats. Derived columns work too, such as the `geometry[x]` and `geometry[y]` columns of a [GeoParquet](./parquet.md#geoparquet) point column.

## Coordinate detection

When points are added from a table file, and `x` or `y` is not given, the coordinate columns are detected from the table: the first coordinate pair the table records (the `obsm/spatial` pair of an [AnnData](./hdf5.md#anndata) table), else the first pair of common column names that the table has: `x`/`y`, `x_location`/`y_location` (10x Xenium transcripts), `x_centroid`/`y_centroid` (Xenium cells), `global_x`/`global_y` (Vizgen MERSCOPE transcripts), `center_x`/`center_y` (MERSCOPE cells), `x_global_px`/`y_global_px` (NanoString CosMx). Names are matched ignoring case. Otherwise, `x` and `y` default to `"x"` and `"y"`. Points referencing an existing table are not detected; their columns are picked in the style section.

## Example

A project showing the cells of a CSV table as points, placed by two of its columns and colored by a third:

```json title="project.tm4"
{
  "layers": [{ "id": "layer", "name": "Sample" }],
  "tables": [
    {
      "id": "cells-table",
      "name": "Cells",
      "dataSource": { "type": "csv", "source": "cells.csv" }
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
        "x": "global_X_pos",
        "y": "global_Y_pos"
      },
      "pointColor": {
        "groupBy": { "column": "cell_type", "palette": "distinct" }
      }
    }
  ]
}
```

## API

The data provider is implemented in the [`@tissuumaps/storage`](../api/@tissuumaps/storage/index.md) package as [`TablePointsDataProvider`](../api/@tissuumaps/storage/classes/TablePointsDataProvider.md).
