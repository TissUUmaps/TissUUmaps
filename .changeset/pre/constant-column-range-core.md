---
"@tissuumaps/core": patch
---

`TableUtils.computeValueRange` returns `[c, c]` for a column whose values are all `c`, instead of `undefined`.
