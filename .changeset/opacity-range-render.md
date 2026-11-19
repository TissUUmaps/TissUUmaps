---
"@tissuumaps/render": minor
---

Opacities from a table column are mapped linearly from the config's `range` to `[0, 1]`. An unset minimum is `0` and an unset maximum is the column's maximum.

Breaking: table values are no longer used as opacities directly. Set `range: [0, 1]` to keep the old mapping.
