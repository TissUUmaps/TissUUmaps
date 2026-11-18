---
"@tissuumaps/storage": minor
---

Parquet tables can be read from Dask dataset directories, such as the points of a SpatialData store.

Breaking: the `ParquetTableData` constructor takes the table's Parquet files (`ParquetSource[]`) instead of one.
