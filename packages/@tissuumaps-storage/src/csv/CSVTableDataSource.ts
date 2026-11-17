import type {
  ParseConfig,
  ParseLocalConfig,
  ParseRemoteConfig,
} from "papaparse";

import type { TableDataSource } from "@tissuumaps/core";

export const csvTableDataSourceType = "csv";

export const csvTableDataSourceDefaults = {
  parseConfig: {
    delimiter: ",",
  },
  // the default missing value markers of pandas.read_csv, but for the empty
  // string, which always marks a missing value
  naValues: [
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
  ],
};

export interface CSVTableDataSource extends TableDataSource<
  typeof csvTableDataSourceType
> {
  source: string;
  columns?: string[];
  idColumn?: string;
  nameColumn?: string;
  loadColumns?: string[];
  /**
   * Cell values that mark a missing value, besides the empty string
   *
   * A missing value is read as `NaN` in numeric columns and as the empty
   * string in other columns; the ID column takes its values as they are.
   *
   * @defaultValue {@link csvTableDataSourceDefaults.naValues}
   */
  naValues?: string[];
  parseConfig?: Pick<
    ParseConfig,
    | "delimiter"
    | "newline"
    | "quoteChar"
    | "escapeChar"
    | "preview"
    | "comments"
    | "fastMode"
    | "skipFirstNLines"
  > &
    Pick<ParseLocalConfig, "encoding"> &
    Pick<
      ParseRemoteConfig,
      "downloadRequestHeaders" | "downloadRequestBody" | "withCredentials"
    >;
}

export type NormalizedCSVTableDataSource = Required<
  Pick<CSVTableDataSource, keyof typeof csvTableDataSourceDefaults>
> &
  Omit<CSVTableDataSource, keyof typeof csvTableDataSourceDefaults>;
