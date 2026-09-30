---
"@tissuumaps/core": minor
"@tissuumaps/render": minor
"tissuumaps": minor
---

Group-by palettes (colors and markers without a map) now assign their values by group position, in the order the groups first appear in the table column, instead of by hashing the group names; random colors are still hashed. Projects that group by a palette show different colors or markers than before. `ConfigUtils.createGroupValueGetter` takes the ordered groups alongside the palette, and the new `TableUtils.loadGroupCounts` lists them.
