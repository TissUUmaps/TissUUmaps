---
sidebar_position: 6
---

# CSV

The built-in **CSV data provider** opens delimited text files as **tables**. Files are parsed in the browser with [PapaParse](https://www.papaparse.com/), in its worker mode; a remote file is parsed while it downloads.

## Data source

CSV data sources have the `type` `"csv"` and accept the following fields:

| Field         | Type       | Description                                                                                                                                                                                                                                                                                      |
| ------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `type`        | `string`   | Always `"csv"`.                                                                                                                                                                                                                                                                                  |
| `source`      | `string`   | URL or path of the CSV file (see [Referencing data](../concepts/projects.md#referencing-data)).                                                                                                                                                                                                  |
| `columns`     | `string[]` | Names of the columns, for files without a header row. Defaults to the first row of the file.                                                                                                                                                                                                     |
| `idColumn`    | `string`   | Column holding the ID of each row (see [Data model](../concepts/data-model.md)). Defaults to row numbers.                                                                                                                                                                                        |
| `nameColumn`  | `string`   | Column holding the name of each row.                                                                                                                                                                                                                                                             |
| `loadColumns` | `string[]` | Columns to load. Defaults to all columns.                                                                                                                                                                                                                                                        |
| `parseConfig` | `object`   | [PapaParse options](https://www.papaparse.com/docs#config): `delimiter` (defaults to `","`), `newline`, `quoteChar`, `escapeChar`, `preview`, `comments`, `fastMode`, `skipFirstNLines`, `encoding`, and for remote files `downloadRequestHeaders`, `downloadRequestBody` and `withCredentials`. |

Empty lines are skipped.

## Column types

A column whose cells all hold numbers is read as 32-bit floats, with blank cells as `NaN`. Any other cell makes the column a string column. The `idColumn` is read as integers if every cell holds one, and as strings otherwise; a blank ID fails the load.

## Limitations

- The loaded columns are held in memory; use `loadColumns` to leave out the columns a project does not use.

## API

The data provider is implemented in the [`@tissuumaps/storage`](/docs/api/@tissuumaps/storage) package as [`CSVTableDataProvider`](/docs/api/@tissuumaps/storage/classes/CSVTableDataProvider). Parsing is delegated to [PapaParse](https://www.papaparse.com/) (see [Dependencies](../development/dependencies.md)).
