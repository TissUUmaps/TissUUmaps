import { useMemo, useRef } from "react";
import { useShallow } from "zustand/shallow";

import { ImageChannelViewMode } from "@tissuumaps/core";
import {
  Viewer,
  type ViewerAdapter,
  ViewerControl,
  ViewerControlAnchor,
} from "@tissuumaps/react";
import type { OpenSeadragonContext } from "@tissuumaps/render";

import {
  useImageDataLoader,
  useLabelsDataLoader,
  usePointsDataLoader,
  useShapesDataLoader,
  useTableDataLoader,
} from "@/data/hooks/useDataLoader";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import { ColorbarsViewerControl } from "./ColorbarsViewerControl";
import { ScalebarViewerControl } from "./ScalebarViewerControl";
import { ViewerToolbar } from "./ViewerToolbar";
import { highlightItemGroup } from "./highlightItemGroup";

export type ViewerPanelProps = {
  className?: string;
};

export function ViewerPanel({ className }: ViewerPanelProps) {
  const viewerRef = useRef<OpenSeadragonContext>(null);
  const interactionMode = useAppStore((state) => state.interactionMode);
  const scalebarVisibility = useProjectStore(
    (state) => state.scalebarVisibility,
  );
  const colorbarsVisibility = useProjectStore(
    (state) => state.colorbarsVisibility,
  );
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

  // Filtered last, so that loading or unloading an object leaves the inputs of
  // the highlight memo unchanged; plain filters keep the project order.
  const loadedImageIds = useAppStore((state) => state.loadedImageIds);
  const loadedLabelsIds = useAppStore((state) => state.loadedLabelsIds);
  const loadedPointsIds = useAppStore((state) => state.loadedPointsIds);
  const loadedShapesIds = useAppStore((state) => state.loadedShapesIds);
  const loadedImages = useMemo(
    () => previewedImages.filter(({ id }) => loadedImageIds.includes(id)),
    [previewedImages, loadedImageIds],
  );
  const loadedLabels = useMemo(
    () =>
      highlightedState.labels.filter(({ id }) => loadedLabelsIds.includes(id)),
    [highlightedState.labels, loadedLabelsIds],
  );
  const loadedPoints = useMemo(
    () =>
      highlightedState.points.filter(({ id }) => loadedPointsIds.includes(id)),
    [highlightedState.points, loadedPointsIds],
  );
  const loadedShapes = useMemo(
    () =>
      highlightedState.shapes.filter(({ id }) => loadedShapesIds.includes(id)),
    [highlightedState.shapes, loadedShapesIds],
  );

  const viewerAdapter: ViewerAdapter = useMemo(
    () => ({
      ...projectState,
      ...highlightedState,
      images: loadedImages,
      labels: loadedLabels,
      points: loadedPoints,
      shapes: loadedShapes,
      interactionMode,
      loadImage,
      loadLabels,
      loadPoints,
      loadShapes,
      loadTable,
    }),
    [
      projectState,
      highlightedState,
      loadedImages,
      loadedLabels,
      loadedPoints,
      loadedShapes,
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
      ref={viewerRef}
      adapter={viewerAdapter}
      backgroundColor={viewerBackgroundColor}
      className={className}
    >
      {/* not faded: the toolbar's menus and popovers render outside the
          viewer, and OpenSeadragon fades its controls once the pointer
          leaves it */}
      <ViewerControl anchor={ViewerControlAnchor.TOP_RIGHT} autoFade={false}>
        <ViewerToolbar
          onResetViewport={() => viewerRef.current?.resetViewport()}
        />
      </ViewerControl>
      {/* lines up to the right of the navigator while that keeps its default
          position, bottom left */}
      {scalebarVisibility && (
        <ViewerControl
          anchor={ViewerControlAnchor.BOTTOM_LEFT}
          autoFade={false}
        >
          <ScalebarViewerControl />
        </ViewerControl>
      )}
      {/* never fades out, as the colorbars explain what the viewer shows */}
      {colorbarsVisibility && (
        <ViewerControl
          anchor={ViewerControlAnchor.BOTTOM_RIGHT}
          autoFade={false}
        >
          <ColorbarsViewerControl />
        </ViewerControl>
      )}
    </Viewer>
  );
}
