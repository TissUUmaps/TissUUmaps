---
sidebar_position: 2
---

# Data types

All data types are two-dimensional (2D), single-channel, single-timepoint variants.

| Data type | Data geometry                                       | Value type                                            |
| --------- | --------------------------------------------------- | ----------------------------------------------------- |
| Images    | Raster geometry                                     | Signed/unsigned integer or floating point values      |
| Labels    | Raster geometry                                     | Signed/unsigned integer values (numeric pixel labels) |
| Points    | Vector geometry                                     | Floating point values (point/vertex coordinates)      |
| Shapes    | Vector geometry                                     | Floating point values (multi-polygon coordinates)     |
| Tables    | Vector geometry <br /> _(when used as data source)_ | Item value (any type) or category (group name)        |

Points, shapes and table rows are addressed by item IDs, which are either integers (below 2^53) or strings.
