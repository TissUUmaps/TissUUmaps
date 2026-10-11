---
"@tissuumaps/storage": patch
---

CSV tables load again in production builds: papaparse is pinned to 5.6.0, since the minified 5.6.1 and 5.7.0 builds crash in their parser worker (mholt/PapaParse#1122).
