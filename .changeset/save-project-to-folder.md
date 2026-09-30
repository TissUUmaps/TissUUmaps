---
"@tissuumaps/core": minor
"tissuumaps": minor
---

Save a project to the connected folder: Save writes a project loaded from the folder back to its file, and Save as picks a new file in the folder and rewrites the data sources relative to it. The Project tab marks unsaved changes and asks before they are discarded. "Copy share link" copies an app link for a project opened from a URL. Adds `sourceFile` and `savedProject` to `ProjectStoreState`, and `SourceUtils.isAppPath` and `SourceUtils.makeRelativePath`.
