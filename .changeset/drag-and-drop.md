---
"@tissuumaps/core": minor
"@tissuumaps/storage": minor
"tissuumaps": minor
---

Data objects, projects and folders can be added by drag and drop. Files and folders dropped on a data panel or its tab open one pre-filled add dialog each, and a project file or folder dropped on the project panel is opened after confirmation. A drag that stays on a tab that accepts it brings its panel to the front.

`SourceUtils` gains `getPathSegments`, which splits a source into its path segments, `getStem` and `getExtension`, which return the name of a source without its last extension and that extension, and `makeProjectPath`, which turns a workspace-relative path into one relative to a project file in the workspace. `getParentSource` no longer throws for URLs with malformed percent-escapes, and returns `null` for URLs without a path, such as `data:` URLs.

Data providers gain an optional `supports` method, which tells whether a source is likely of their format. All built-in data providers implement it: by file extension, and for OME-Zarr groups and Parquet shapes by also reading their metadata. Remote zipped OME-Zarr files are recognized by their `.ozx` extension in any case.

Data providers also gain the optional `readName` and `prepareDataSource` methods. `readName` returns the name a source stores in its metadata, which OME-TIFF and OME-Zarr images implement. Table points can name a table file as their `source`, which `prepareDataSource` adds as a table that the points then reference. The table points data provider's constructor therefore now takes `getTableDataProviders` and `addTable`.

Data provider UI schemas no longer contain the `source` control, which the app now renders itself, with pickers that choose files and folders relative to the project file if it lies in the workspace: third-party data providers have to remove it from their UI schema. The data provider's `prepareDataSource` runs when a data object is added, and when an edited data source is saved with a changed source or type.

The add data object dialog is built around the source: the data providers that support it are detected and the first one is selected, the name is taken from the source, and a missing required source or other invalid input is shown. Added data objects get IDs derived from their file names, and are expanded in their panel.
