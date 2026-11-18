import type { ImageDataSource, TileSourceConfig } from "@tissuumaps/core";

export const openSeadragonImageDataSourceType = "image";

export const openSeadragonImageDataSourceDefaults = {};

export interface OpenSeadragonImageDataSource extends ImageDataSource<
  typeof openSeadragonImageDataSourceType
> {
  tileSource?: TileSourceConfig;
}

export type NormalizedOpenSeadragonImageDataSource = Required<
  Pick<
    OpenSeadragonImageDataSource,
    keyof typeof openSeadragonImageDataSourceDefaults
  >
> &
  Omit<
    Omit<
      OpenSeadragonImageDataSource,
      keyof typeof openSeadragonImageDataSourceDefaults
    >,
    "source" | "tileSource"
  > &
  (
    | { source: string; tileSource?: undefined }
    | { source?: undefined; tileSource: TileSourceConfig }
  );
