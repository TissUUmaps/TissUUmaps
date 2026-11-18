---
"@tissuumaps/core": minor
---

Table data can group its columns into query groups (`TableData.getQueryGroups`, searched with the `queryGroupId` option of `suggestColumnQueries`, with `describeColumn` and `mapColumn`) and declare categorical columns (`TableData.isColumnCategorical`, with the `TableUtils.loadIsColumnCategorical` fallback).
