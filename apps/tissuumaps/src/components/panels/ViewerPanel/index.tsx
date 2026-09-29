import { useMemo } from "react";
import { useShallow } from "zustand/shallow";

import { ImageChannelViewMode } from "@tissuumaps/core";
import {
  Viewer,
  type ViewerAdapter,
  ViewerControl,
  ViewerControlAnchor,
} from "@tissuumaps/react";

import {
  useImageDataLoader,
  useLabelsDataLoader,
  usePointsDataLoader,
  useShapesDataLoader,
  useTableDataLoader,
} from "@/hooks/useDataLoader";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import { InteractionModeViewerControls } from "./InteractionModeViewerControls";
import { PointSizeViewerControl } from "./PointSizeViewerControl";
import { highlightItemGroup } from "./highlightItemGroup";

export type ViewerPanelProps = {
  className?: string;
};

export function ViewerPanel({ className }: ViewerPanelProps) {
  const interactionMode = useAppStore((state) => state.interactionMode);
  const imageChannelPreview = useAppStore((state) => state.imageChannelPreview);
  const viewerBackgroundColor = useProjectStore(
    (state) => state.viewerBackgroundColor,
  );

  const images = useProjectStore((state) => state.images);
  const previewedImages = useMemo(
    () =>
      imageChannelPreview === null
        ? images
        : images.map((image) =>
            image.id === imageChannelPreview.imageId
              ? {
                  ...image,
                  channelViewMode: ImageChannelViewMode.color,
                  activeChannel: imageChannelPreview.channelIndex,
                }
              : image,
          ),
    [images, imageChannelPreview],
  );

  const highlightedItemGroup = useAppStore(
    (state) => state.highlightedItemGroup,
  );

  const projectState = useProjectStore(
    useShallow((state) => ({
      projectInstanceId: state.instanceId,
      layers: state.layers,
      labels: state.labels,
      points: state.points,
      shapes: state.shapes,
      tables: state.tables,
      markerMaps: state.markerMaps,
      sizeMaps: state.sizeMaps,
      colorMaps: state.colorMaps,
      visibilityMaps: state.visibilityMaps,
      opacityMaps: state.opacityMaps,
      osOptions: state.osOptions,
      glOptions: state.glOptions,
    })),
  );

  const loadImage = useImageDataLoader();
  const loadLabels = useLabelsDataLoader();
  const loadPoints = usePointsDataLoader();
  const loadShapes = useShapesDataLoader();
  const loadTable = useTableDataLoader();

  // Memoized on its own: the renderers compare the opacity map it builds by
  // identity, so rebuilding it for an unrelated change of the adapter would
  // re-resolve and re-upload the colors of every highlighted object.
  const { labels, points, shapes, opacityMaps } = projectState;
  const highlightedState = useMemo(
    () =>
      highlightItemGroup(
        { labels, points, shapes, opacityMaps },
        highlightedItemGroup,
      ),
    [labels, points, shapes, opacityMaps, highlightedItemGroup],
  );

  const viewerAdapter: ViewerAdapter = useMemo(
    () => ({
      ...projectState,
      ...highlightedState,
      images: previewedImages,
      interactionMode,
      loadImage,
      loadLabels,
      loadPoints,
      loadShapes,
      loadTable,
    }),
    [
      projectState,
      previewedImages,
      highlightedState,
      interactionMode,
      loadImage,
      loadLabels,
      loadPoints,
      loadShapes,
      loadTable,
    ],
  );

  return (
    <Viewer
      adapter={viewerAdapter}
      backgroundColor={viewerBackgroundColor}
      className={className}
    >
      <ViewerControl anchor={ViewerControlAnchor.TOP_LEFT}>
        <InteractionModeViewerControls />
      </ViewerControl>
      {projectState.points.length > 0 && (
        <ViewerControl anchor={ViewerControlAnchor.TOP_RIGHT}>
          <PointSizeViewerControl />
        </ViewerControl>
      )}
    </Viewer>
  );
}
