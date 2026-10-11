---
"@tissuumaps/storage": patch
---

CSV tables load again in production builds: papaparse 5.7.0, whose minified build crashes in its parser worker (mholt/PapaParse#1122), is excluded from the supported papaparse versions.
