import type {
  Color,
  GroupValueMap,
  Image,
  ImageData,
  InteractionMode,
  Labels,
  LabelsData,
  Layer,
  Marker,
  MultiPolygon,
  OpenSeadragonOptions,
  Points,
  PointsData,
  Shapes,
  ShapesData,
  Table,
  TableData,
  WebGLOptions,
} from "@tissuumaps/core";

export interface ViewerAdapter {
  projectInstanceId: string;
  layers: Layer[];
  images: Image[];
  labels: Labels[];
  points: Points[];
  shapes: Shapes[];
  tables: Table[];
  markerMaps: GroupValueMap<Marker>[];
  sizeMaps: GroupValueMap<number>[];
  colorMaps: GroupValueMap<Color>[];
  visibilityMaps: GroupValueMap<boolean>[];
  opacityMaps: GroupValueMap<number>[];
  osOptions: OpenSeadragonOptions;
  glOptions: WebGLOptions;
  loadImage: (
    image: Image,
    options?: { signal?: AbortSignal },
  ) => Promise<ImageData>;
  loadLabels: (
    labels: Labels,
    options?: { signal?: AbortSignal },
  ) => Promise<LabelsData>;
  loadPoints: (
    points: Points,
    options?: { signal?: AbortSignal },
  ) => Promise<PointsData>;
  loadShapes: (
    shapes: Shapes,
    options?: { signal?: AbortSignal },
  ) => Promise<ShapesData>;
  loadTable: (
    table: Table,
    options?: { signal?: AbortSignal },
  ) => Promise<TableData>;

  // shapes drawing
  interactionMode: InteractionMode;
  addShape?: (shape: MultiPolygon) => void;
}
