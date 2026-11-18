---
sidebar_position: 1
---

# Projects

A **project** consists of all data objects and the viewer configuration of a single TissUUmaps instance. It describes how the data objects spatially relate to each other and how they are displayed. A project does not hold any data itself. Instead, its data objects reference separately stored data of various types and formats through "data sources".

Projects are stored as **TissUUmaps project files** (in JSON format, file name ending with `.tm4`). This is TissUUmaps' own format, which can represent any project. Further project providers can support other formats as import and export formats (see below). These may represent only part of a project, e.g. its data but not how it is displayed, so a project is best kept as a TissUUmaps project file.

Project files and their data may be stored locally (client-side) or hosted remotely (server-side; together with the TissUUmaps application or elsewhere).

## Opening, importing and downloading

Projects are read and written by **project providers**, one per project format. The TissUUmaps project provider handles TissUUmaps project files. Further formats are handled by other project providers, built into TissUUmaps or added by plugins. The format of a file, folder or URL is recognized automatically. If none or several formats match, TissUUmaps asks which one it is.

A project can be **opened**, which replaces the open project, or **imported**, which adds its layers, data objects, tables and maps to the open project. An imported project's name is used only if the open project has no data yet, and its viewer settings are ignored. Its project-relative paths are resolved against where the imported project was loaded from (see [Referencing data](#referencing-data)). A project whose IDs are already taken in the open project, e.g. one that was imported before, cannot be imported.

Open and import projects with the actions in the Project tab's menu, or by dropping a project file on the Project panel. A dropped folder is opened as the workspace, unless it is a project of a known format. Such a folder is opened or imported like a project file if it is inside the workspace. Otherwise, it can be opened as the workspace and as a project at once.

"Download project" saves the open project as a TissUUmaps project file. The file is named after the file the project was opened from, or else after the project. If several formats can be written, "Download as" offers each of them.

### Links

Each project format has its own GET parameter, which is `project` for TissUUmaps project files. The first such parameter in the URL opens a project, and every following one imports a project into it, in order. For example, `?project=liver.tm4&project=annotations.tm4` opens `liver.tm4` and imports `annotations.tm4`. Without such parameters, `project.tm4` next to the TissUUmaps application is opened.

Opening or importing a project from a URL updates the address bar accordingly, so that reloading the page loads the same projects again. Projects opened or imported from local files cannot be recorded there.

## Workspaces

In Chromium-based browsers (e.g. Chrome), a workspace can be opened alongside a project. A workspace is a local folder that TissUUmaps gets read access to, so that file paths within it can be resolved. Other browsers (e.g. Firefox, Safari) do not support workspaces.
Open a workspace with "Open workspace…" in the welcome view in the Project tab or in the Project tab's menu, then a project with "Open project from the workspace…" in the welcome view, or "Open project file…" in the menu. The workspace is shown at the bottom of the Project panel.
A project file opened from outside the workspace is opened only after confirmation. Its project-relative paths are then resolved within the workspace instead of next to the project file, so its data may fail to load.
While a workspace is open, the file and folder buttons next to a data source's Source field pick a file or a folder (e.g. a Zarr store) inside it and fill in its workspace-relative path.

## Referencing data

Data sources reference their data by a single `source` string, which is one of the following:

- A **URL** with a scheme (e.g. `https://example.org/data/he.dzi`) refers to remote data and is used as is.
- An **app-relative path**, prefixed with `//` (e.g. `//data/he.dzi`), is resolved against the URL of the TissUUmaps application.
- A **workspace-relative path**, prefixed with `/` (e.g. `/data/he.dzi`), refers to local data within the **TissUUmaps workspace directory** the user has opened. Without an open workspace, it falls back to being app-relative.
- A **project-relative path**, without prefix (e.g. `images/he.dzi`, `./images/he.dzi` or `../shared/he.dzi`), is resolved **against where the project itself was loaded from**, e.g. its project file. This way, a project and its data can be moved or copied to another host or workspace together, without changing the project:

  - A project served at `https://example.org/studies/liver/project.tm4` resolves `images/he.dzi` to `https://example.org/studies/liver/images/he.dzi`.
  - A project opened from within a workspace resolves it within the project file's directory: at `/studies/project.tm4`, `he.dzi` becomes `/studies/he.dzi` as long as `/studies` is part of the workspace.
  - A project opened from a local file outside of the workspace through the project panel has no such base; the path is taken as workspace-relative from the workspace root, so `he.dzi` becomes `/he.dzi`.

  The latter two yield a workspace-relative path, which, as above, falls back to being app-relative without an open workspace.

Paths use `/` as separator and may contain `.` and `..` segments. A path cannot leave the workspace; against a project URL, `..` segments that would lead above the root of the URL's path are dropped, as in any URL. A project-relative path whose first segment contains a colon (e.g. `sample1:ch2.tif`) would be taken for a URL with the scheme `sample1:`; write it as `./sample1:ch2.tif` instead. For consistency, it is recommended to store the project file in the root of the workspace. Conventionally, locally stored TissUUmaps project files are named `project.tm4`, and the TissUUmaps workspace directory name is used to identify the project.
