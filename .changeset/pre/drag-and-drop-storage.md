---
"@tissuumaps/storage": minor
---

pr: #302
commit: a4225640dd4be545727210c11e5c12d3d81d650a

All built-in data providers implement `supports`, and the OME-TIFF and OME-Zarr image and labels providers implement `readName`.

Breaking: the table points data provider's constructor takes `getTableDataProviders` and `addTable`.
