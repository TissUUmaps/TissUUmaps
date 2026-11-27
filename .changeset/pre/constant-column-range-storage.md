---
"@tissuumaps/storage": patch
---

Parquet tables report the value range `[c, c]` for a column whose values are all `c`, instead of none.
