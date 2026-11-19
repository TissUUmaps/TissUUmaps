---
sidebar_position: 1
---

# TissUUmaps

The built-in **TissUUmaps project provider** reads and writes TissUUmaps' own project format, `.tm4` files. Unlike other formats, a `.tm4` file holds a whole project, including how it is displayed. Projects opened from other formats are therefore best kept as `.tm4` files.

## Format

A `.tm4` file is a JSON file. It holds the project as described in the [Data model](../concepts/data-model.md). Non-finite numbers, which JSON cannot represent (e.g. an unlimited maximum zoom of the viewer), are encoded as reserved strings (see [`JSONUtils`](../api/@tissuumaps/core/classes/JSONUtils.md)).

Its project-relative data sources are relative to the file's location (see [Referencing data](../concepts/projects.md#referencing-data)).

## Opening and downloading

A `.tm4` file is opened or imported like any project (see [Projects](../concepts/projects.md#opening-importing-and-downloading)). Its GET parameter is `project`, e.g. `?project=https://example.org/liver/project.tm4`. Without any project parameter in the URL, `project.tm4` next to the TissUUmaps application is opened.

"Download project" saves the open project as a `.tm4` file, even if it was opened from another format. The file is named after the file the project was opened from, or else after the project.

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
