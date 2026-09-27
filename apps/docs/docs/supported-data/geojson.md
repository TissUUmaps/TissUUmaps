---
sidebar_position: 5
---

# GeoJSON

The built-in **GeoJSON data provider** opens [GeoJSON](https://geojson.org/) files as **shapes**. Files are parsed in the browser, in a worker. The provider is named after GeoJSON rather than JSON: a JSON document without GeoJSON structure is not shapes, so there is no plain JSON provider.

## Data source

GeoJSON data sources have the `type` `"geojson"` and accept the following fields:

| Field          | Type     | Description                                                                                                               |
| -------------- | -------- | ------------------------------------------------------------------------------------------------------------------------- |
| `type`         | `string` | Always `"geojson"`.                                                                                                       |
| `source`       | `string` | URL or path of the GeoJSON file (see [Referencing data](../concepts/projects.md#referencing-data)).                       |
| `idProperty`   | `string` | Feature property holding the ID of each shape (see [Feature IDs](#feature-ids)). Defaults to the position of the feature. |
| `nameProperty` | `string` | Feature property holding the name of each shape.                                                                          |
| `table`        | `string` | ID of the table annotating the shapes (see [Data model](../concepts/data-model.md)).                                      |

## Shapes

The file holds a `FeatureCollection`, a single `Feature`, a `GeometryCollection` or a bare geometry. `Polygon` and `MultiPolygon` geometries are read as shapes; a feature with another geometry, or without one, is skipped with a warning.

### Feature IDs

The `idProperty` names the feature property holding the shape IDs, which keep their JSON types: integers stay integers and strings stay strings, so a table annotating the shapes has to key its rows the same way. Without an `idProperty`, shapes are keyed by their position in the file. A feature whose ID is missing, empty, or neither a string nor an integer fails the load.

## Limitations

- `idProperty` and `nameProperty` can only be set for a `FeatureCollection`. Setting one for another root fails the load.
- Only polygons and multi-polygons are read. Points and lines are skipped.

## API

The data provider is implemented in the [`@tissuumaps/storage`](/docs/api/@tissuumaps/storage) package as [`GeoJSONShapesDataProvider`](/docs/api/@tissuumaps/storage/classes/GeoJSONShapesDataProvider). The file is parsed in a worker.
