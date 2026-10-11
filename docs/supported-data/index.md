# Supported data

Data is read by **data providers**, one per file format and kind of data: a format such as TIFF has an image and a labels provider, both configured through a data source of the same `type`. Every page describes one format: the fields of its data sources, the **profiles** of the format it detects from a file (such as OME-TIFF, or GeoParquet), the **conventions** it honors (such as the pandas index of a Parquet file), what it reads for each kind of data, and its limitations.

The following table lists the capabilities of the built-in data providers:

| Data provider             | Images               | Labels          | Points          | Shapes         | Tables         |
| ------------------------- | -------------------- | --------------- | --------------- | -------------- | -------------- |
| [Image](./image.md)       | Image or tile source |                 |                 |                |                |
| [TIFF](./tiff.md)         | TIFF source          | TIFF source     |                 |                |                |
| [OME-Zarr](./ome-zarr.md) | OME-Zarr source      | OME-Zarr source |                 |                |                |
| [Table](./table.md)       |                      |                 | Table reference |                |                |
| [GeoJSON](./geojson.md)   |                      |                 |                 | GeoJSON source |                |
| [Parquet](./parquet.md)   |                      |                 |                 | Parquet source | Parquet source |
| [CSV](./csv.md)           |                      |                 |                 |                | CSV source     |
| [HDF5](./hdf5.md)         |                      |                 |                 |                | HDF5 source    |
| [Zarr](./zarr.md)         |                      |                 |                 |                | Zarr source    |

Additional data formats may be supported by third-party data providers.
