import type { ImageDataSource, TileSourceConfig } from "@tissuumaps/core";

export const openSeadragonImageDataSourceType = "image";

export const openSeadragonImageDataSourceDefaults = {};

export interface OpenSeadragonImageDataSource extends ImageDataSource<
  typeof openSeadragonImageDataSourceType
> {
  tileSource?: string | TileSourceConfig;
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
  > &
  Required<Pick<OpenSeadragonImageDataSource, "tileSource">>;
