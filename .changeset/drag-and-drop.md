---
"@tissuumaps/core": minor
---

`SourceUtils` gains `getPathSegments`, which splits a source into its path segments, `getStem`, which returns the name of a source without its last extension, and `makeProjectPath`, which turns a workspace-relative path into one relative to a project file in the workspace. `getParentSource` no longer throws for URLs with malformed percent-escapes, and returns `null` for URLs without a path, such as `data:` URLs.
