---
"@tissuumaps/storage": minor
---

HDF5 and Zarr tables declare boolean datasets as categorical columns, and AnnData tables also their `categorical` and `nullable-boolean` columns, as Parquet tables do for pandas `category` columns. AnnData tables group their columns into observations (`obs`), features (`X`, the layers and `raw`) and embeddings (`obsm`).
