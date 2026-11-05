---
"@tissuumaps/storage": minor
---

An image data source's `tileSource` only takes an inline tile source configuration. A descriptor URL is given as `source`, the only field of a data source that is resolved as a path.

Breaking: a descriptor URL given as `tileSource` has to be given as `source`. A descriptor whose URL has an image extension has to be given inline as `tileSource`.
