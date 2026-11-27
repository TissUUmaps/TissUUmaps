---
"@tissuumaps/render": minor
---

Opacities from a table column are mapped linearly from the config's `range` to `[0, 1]`. An unset end of the range is the column's minimum or maximum, as for colors.

Breaking: table values are no longer used as opacities directly. Set `range: [0, 1]` to keep the old mapping.
