# Supported projects

Projects are read and written by **project providers**, one per project format (see [Projects](../concepts/projects.md#opening-importing-and-saving)). A project provider that reads a format opens and imports projects from it, and has its own GET parameter. One that writes a format saves and downloads projects in it.

The following table lists the built-in project providers:

| Project provider                       | Opens and imports | Downloads | GET parameter |
| -------------------------------------- | ----------------- | --------- | ------------- |
| [TissUUmaps](./tissuumaps.md) (`.tm4`) | Yes               | Yes       | `project`     |
| [SpatialData](./spatialdata.md)        | Yes               |           | `spatialdata` |

Additional project formats may be supported by plugins.
