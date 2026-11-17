---
"@tissuumaps/core": minor
"@tissuumaps/storage": minor
"tissuumaps": minor
---

Table points detect their coordinate columns when added from a file: the `obsm` pairs of AnnData tables (`spatial` first, via the new `TableData.getCoordinateColumns`), else common column names of spatial omics exports. The style section of points shows the position, and data provider forms can mark table column fields (`options.tableColumn`, `options.coordinate`).
