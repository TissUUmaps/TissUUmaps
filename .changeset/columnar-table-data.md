---
"@tissuumaps/storage": patch
"tissuumaps": patch
---

Internal: the in-memory table of CSV files moves to a shared `ColumnarTableData` base, and the application's `addTable` helper for data providers is extracted, so that other providers can reuse both.
