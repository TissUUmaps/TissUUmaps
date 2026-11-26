---
sidebar_position: 2
---

# SpatialData

The built-in **SpatialData project provider** opens [SpatialData](https://spatialdata.scverse.org/) stores as projects. It reads the metadata of a store and creates data objects of the existing types, whose data the built-in data providers then read. Stores are only read: a project opened from a store is downloaded in the [TissUUmaps](./tissuumaps.md) format.

## Opening a store

A store is opened or imported like any project (see [Projects](../concepts/projects.md#opening-importing-and-downloading)):

- From a URL, with Open or Import, or with the `spatialdata` GET parameter, e.g. `?spatialdata=https://example.org/data.zarr`.
- From a folder, by dropping the store's folder on the Project panel: inside the workspace, it is opened or imported; otherwise, it can be opened as the workspace and as a project at once.

A store is recognized by the SpatialData attributes of its root group. Its elements are listed from its consolidated metadata, which spatialdata writes by default.

The data sources of the project are relative to the store's location, e.g. `data.zarr/images/he` for a store at `https://example.org/data.zarr`, or `images/he` for a store that is the workspace.

## Elements

Every element becomes data objects named after it:

| Element           | Data objects                                                                                                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Images            | [OME-Zarr](../supported-data/ome-zarr.md) images.                                                                                                                                        |
| Labels            | [OME-Zarr](../supported-data/ome-zarr.md) labels.                                                                                                                                        |
| Shapes (polygons) | [Parquet](../supported-data/parquet.md) shapes.                                                                                                                                          |
| Shapes (circles)  | A [Parquet](../supported-data/parquet.md) table of the shapes file, and a [table](../supported-data/table.md) point cloud of it, sized by the radius of the circles.                     |
| Points            | A [Parquet](../supported-data/parquet.md) table of the points' [Dask](../supported-data/parquet.md#dask) dataset directory, and a [table](../supported-data/table.md) point cloud of it. |
| Tables            | [Zarr](../supported-data/zarr.md) tables, keyed by their instance key.                                                                                                                   |

Polygons and circles are told apart from the geometry types of the shapes file. Points that name a feature (e.g. the gene of a transcript) are colored by it, with the color palette that labels use.

A table that annotates a single labels or polygons element is linked to it, so its columns can color the element. A table annotating several elements is not linked, as their instance keys repeat across them. Neither is a table annotating circles or points, as their point cloud's table holds their coordinates. Unlinked tables are part of the project nonetheless.

## Coordinate systems

Every coordinate system becomes a layer named after it. An element is placed in each coordinate system it maps into, as one object per layer. Only one layer is shown at first: `global`, or else the first one that holds the most objects. Each layer shows only its first element: its first image, or else its first labels, shapes or points. The other elements are hidden, so opening a store does not show them all at once.

An object is placed by the element's transformation into the coordinate system. Only the x and y axes are kept. TissUUmaps places objects by flipping, scaling uniformly, rotating and translating them, so a transformation that scales the axes differently or shears them is approximated: its uniform scale keeps the area. A warning in the browser console names every approximated transformation.

## Limitations

- Stores need consolidated metadata. Zipped stores are not supported.
- Tables annotating several elements, circles or points are not linked to them.
- Transformations that scale the axes differently or shear them are approximated.
- Axes other than x and y, e.g. the z coordinates of points, are ignored.
- A remote points directory is read up to its first missing partition (see [Dask](../supported-data/parquet.md#dask)).
- Format versions are not checked. Stores from before format versioning (no root SpatialData attributes, shapes as Zarr arrays) are not supported.
- Shapes written with geoarrow encoding are not supported (spatialdata writes WKB by default).
- The points and shapes of hidden elements and coordinate systems are loaded too.

## API

The project provider is implemented in the [`@tissuumaps/storage`](../api/@tissuumaps/storage/index.md) package as [`SpatialDataProjectProvider`](../api/@tissuumaps/storage/classes/SpatialDataProjectProvider.md). Reading the store's metadata is delegated to [zarrita](https://github.com/manzt/zarrita.js).
