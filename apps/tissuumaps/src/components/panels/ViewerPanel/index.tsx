import { useCallback, useMemo, useRef } from "react";
import { useShallow } from "zustand/shallow";

import { type Dims, ImageChannelViewMode, type Rect } from "@tissuumaps/core";
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

import {
  ScaleBarViewerControl,
  type ScaleBarViewerControlHandle,
} from "./ScaleBarViewerControl";
import { ViewerToolbar } from "./ViewerToolbar";
import { highlightItemGroup } from "./highlightItemGroup";

export type ViewerPanelProps = {
  className?: string;
};

export function ViewerPanel({ className }: ViewerPanelProps) {
  const viewerRef = useRef<OpenSeadragonContext>(null);
  // kept here, as the scale bar mounts after the viewer reports its first
  // viewport, and draws from it again whenever it shows
  const viewportRef = useRef<{ viewport: Rect; containerSize: Dims } | null>(
    null,
  );
  const scaleBarRef = useRef<ScaleBarViewerControlHandle>(null);
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

  const onViewportChange = useCallback(
    (viewport: Rect, containerSize: Dims) => {
      viewportRef.current = { viewport, containerSize };
      scaleBarRef.current?.update();
    },
    [],
  );

  return (
    <Viewer
      ref={viewerRef}
      adapter={viewerAdapter}
      backgroundColor={viewerBackgroundColor}
      className={className}
      onViewportChange={onViewportChange}
    >
      {/* not faded: the toolbar's menus and popovers render outside the
          viewer, and OpenSeadragon fades its controls once the pointer
          leaves it */}
      <ViewerControl anchor={ViewerControlAnchor.TOP_RIGHT} autoFade={false}>
        <ViewerToolbar
          onResetViewport={() => viewerRef.current?.resetViewport()}
        />
      </ViewerControl>
      {/* lines up to the right of the navigator, which is mounted first */}
      <ViewerControl anchor={ViewerControlAnchor.BOTTOM_LEFT} autoFade={false}>
        <ScaleBarViewerControl ref={scaleBarRef} viewportRef={viewportRef} />
      </ViewerControl>
    </Viewer>
  );
}
