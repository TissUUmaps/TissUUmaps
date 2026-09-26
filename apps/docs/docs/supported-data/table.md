---
sidebar_position: 4
---

# Table

The built-in **Table data provider** derives **points** from a table of the project: every row is a point, placed by two of the table's numeric columns. This is a virtual format: the provider reads no file of its own.

## Data source

Table data sources have the `type` `"table"` and accept the following fields:

| Field   | Type     | Description                                                                       |
| ------- | -------- | --------------------------------------------------------------------------------- |
| `type`  | `string` | Always `"table"`.                                                                 |
| `table` | `string` | ID of the table holding the points (see [Data model](../concepts/data-model.md)). |
| `x`     | `string` | Column holding the x coordinate of each point. Defaults to `"x"`.                 |
| `y`     | `string` | Column holding the y coordinate of each point. Defaults to `"y"`.                 |

## Points

The points take the IDs and names of the table's rows, and the table annotates them: every column of the table can color or size them. The coordinate columns are read as 32-bit floats. Derived columns work too, such as the `geometry[x]` and `geometry[y]` columns of a [GeoParquet](./parquet.md#geoparquet) point column.

## Example

A project showing the cells of a CSV table as points, placed by two of its columns and colored by a third:

```json title="project.tmap"
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
      "pointColor": { "groupBy": { "column": "cell_type" } }
    }
  ]
}
```

## API

The data provider is implemented in the [`@tissuumaps/storage`](/docs/api/@tissuumaps/storage) package as [`TablePointsDataProvider`](/docs/api/@tissuumaps/storage/classes/TablePointsDataProvider).
