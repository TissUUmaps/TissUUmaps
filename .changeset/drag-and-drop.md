---
"@tissuumaps/core": minor
"@tissuumaps/storage": minor
---

`SourceUtils` gains `getPathSegments`, which splits a source into its path segments, `getStem` and `getExtension`, which return the name of a source without its last extension and that extension, and `makeProjectPath`, which turns a workspace-relative path into one relative to a project file in the workspace. `getParentSource` no longer throws for URLs with malformed percent-escapes, and returns `null` for URLs without a path, such as `data:` URLs.

Data providers gain an optional `supports` method, which tells whether a source is likely of their format. All built-in data providers implement it: by file extension, and for OME-Zarr groups and Parquet shapes by also reading their metadata. Remote zipped OME-Zarr files are recognized by their `.ozx` extension in any case.

Data providers also gain the optional `readName` and `prepare` methods. `readName` returns the name a source stores in its metadata, which OME-TIFF and OME-Zarr images implement. Table points can name a table file as their `source`, which `prepare` adds as a table that the points then reference. The table points data provider's constructor therefore now takes `getTableDataProviders` and `addTable`.
