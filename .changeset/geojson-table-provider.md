---
"@tissuumaps/storage": minor
"tissuumaps": minor
---

GeoJSON files can be read as tables of their feature properties (`"geojson"` table data sources), keyed by the same IDs as their shapes. GeoJSON and GeoParquet shapes added without a `table` add their own file as their attribute table, so that they can be colored by its columns.
