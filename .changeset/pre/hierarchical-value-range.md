---
"@tissuumaps/storage": minor
---

HDF5 and Zarr table data no longer implement `loadValueRange`, which decoded the whole column in the worker and so was no cheaper than loading it; their value ranges are now computed from the loaded column values (e.g. by `TableUtils.loadValueRange`), so continuous colors no longer read a column twice. Code calling `loadValueRange` directly on `HDF5TableData` or `ZarrTableData` has to use `TableUtils.loadValueRange` instead. Parquet columns without row group statistics, and point geometry columns without a GeoParquet bounding box, now get a value range computed from their values instead of none.
