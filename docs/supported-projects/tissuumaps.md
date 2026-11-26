---
sidebar_position: 1
---

# TissUUmaps

The built-in **TissUUmaps project provider** reads and writes TissUUmaps' own project format, `.tm4` files. Unlike other formats, a `.tm4` file holds a whole project, including how it is displayed. Projects opened from other formats are therefore best kept as `.tm4` files.

## Format

A `.tm4` file is a JSON file. It holds the project as described in the [Data model](../concepts/data-model.md). Non-finite numbers, which JSON cannot represent (e.g. an unlimited maximum zoom of the viewer), are encoded as reserved strings (see [`JSONUtils`](../api/@tissuumaps/core/classes/JSONUtils.md)).

Its project-relative data sources are relative to the file's location (see [Referencing data](../concepts/projects.md#referencing-data)).

## Opening, saving and downloading

A `.tm4` file is opened or imported like any project (see [Projects](../concepts/projects.md#opening-importing-and-saving)). Its GET parameter is `project`, e.g. `?project=https://example.org/liver/project.tm4`. Without any project parameter in the URL, `project.tm4` next to the TissUUmaps application is opened.

"Save project" writes a project opened from a `.tm4` file in the workspace back to it. "Save as…" saves any project as a new `.tm4` file in the workspace, unless it was opened from another format that can be written, and rewrites its paths for the new file (see [Projects](../concepts/projects.md#saving)).

"Download project" saves the open project as a `.tm4` file, even if it was opened from another format. The file keeps the name of the file or URL the project was opened from, unless that was a file outside the workspace; otherwise, it is named after the project. A downloaded project keeps its project-relative paths, so they only stay valid next to the file it was opened from; to store a project elsewhere in the workspace, use "Save as…" instead. The paths of imported projects were resolved on import and stay valid wherever it is stored.

## Conventions

Locally stored `.tm4` files are conventionally named `project.tm4`, and the name of the workspace folder identifies the project. Store the `.tm4` file in the root of the workspace, so that its project-relative paths are workspace paths as well.

## Example

A `.tm4` file showing one image:

```json title="project.tm4"
{
  "name": "Liver",
  "layers": [{ "id": "layer", "name": "Layer" }],
  "images": [
    {
      "id": "he",
      "name": "H&E",
      "layer": "layer",
      "dataSource": { "type": "image", "source": "images/he.dzi" }
    }
  ]
}
```

## API

The project provider is implemented in the [`@tissuumaps/storage`](../api/@tissuumaps/storage/index.md) package as [`TissUUmapsProjectProvider`](../api/@tissuumaps/storage/classes/TissUUmapsProjectProvider.md).
