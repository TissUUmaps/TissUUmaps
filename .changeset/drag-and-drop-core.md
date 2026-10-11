---
"@tissuumaps/core": minor
---

pr: #302
commit: a4225640dd4be545727210c11e5c12d3d81d650a

Data providers gain the optional `supports`, `readName` and `prepareDataSource` methods, and `SourceUtils` gains `getPathSegments`, `getStem`, `getExtension` and `makeProjectPath`.

Breaking: data provider UI schemas no longer contain the `source` control, which the app now renders itself.
