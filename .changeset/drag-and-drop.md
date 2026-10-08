---
"@tissuumaps/core": minor
"@tissuumaps/storage": minor
"tissuumaps": minor
---

Data objects, projects and folders can be added by drag and drop. Files and folders dropped on a data panel or its tab open one pre-filled add dialog each, and a project file or folder dropped on the project panel is opened after confirmation. A drag that stays on a tab that accepts it brings its panel to the front.

The add data object dialog is built around the source: it selects the first data provider that supports the source, takes the name from the source, and shows invalid input. Added data objects get IDs derived from their file names, and are expanded in their panel. Source pickers choose files and folders relative to the project file if it lies in the workspace.

Data providers gain the optional `supports`, `readName` and `prepareDataSource` methods. All built-in data providers implement `supports`, by file extension and, for OME-Zarr groups and Parquet shapes, also by their metadata; OME-TIFF and OME-Zarr images and labels implement `readName`. `prepareDataSource` runs when a data object is added, and when an edited data source is saved with a changed source or type: table points use it to add a table file named as their `source` as a table. Remote zipped OME-Zarr files are recognized by their `.ozx` extension in any case.

`SourceUtils` gains `getPathSegments`, `getStem`, `getExtension` and `makeProjectPath`, which turns a workspace-relative path into one relative to a project file in the workspace. `getParentSource` no longer throws for URLs with malformed percent-escapes, and returns `null` for URLs without a path, such as `data:` URLs.

Breaking: data provider UI schemas no longer contain the `source` control, which the app renders itself, so third-party data providers have to remove it from theirs. The table points data provider's constructor now takes `getTableDataProviders` and `addTable`.
