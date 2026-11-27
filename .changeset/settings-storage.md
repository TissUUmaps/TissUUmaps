---
"@tissuumaps/storage": minor
---

HDF5 and Zarr tables declare boolean datasets as categorical columns, AnnData tables also their `categorical` and `nullable-boolean` columns, and Parquet tables their pandas `category` columns. AnnData tables group their columns into observations (`obs`), features (`X`, the layers and `raw`) and embeddings (`obsm`).
