---
"@tissuumaps/storage": minor
---

CSV tables read `naValues` as missing values: by default the missing-value markers of pandas (e.g. `NA`, `NaN`, `null`), so that a column holding such markers among numbers is now read as a numeric column with `NaN`, where it used to be read as a string column.
