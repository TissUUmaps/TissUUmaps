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
    OpenSeadragonImageDataSource,
    keyof typeof openSeadragonImageDataSourceDefaults
  >;
