import { useEffect, useState } from "react";

import { useAppStore } from "@/stores/app";

import { useLatestCallback } from "./useLatestCallback";

/**
 * Keeps the IDs of the images whose settings are expanded, adding the image of
 * each request of the app store's `showImageSettings`
 *
 * @param onShow - Called on each request to bring the settings to the front
 * @returns The expanded image IDs and a setter for them
 */
export function useExpandedImageIds(
  onShow: () => void,
): [string[], (expandedIds: string[]) => void] {
  const request = useAppStore((state) => state.imageSettingsRequest);
  return useExpandedIds(request, request?.imageId ?? null, onShow);
}

/**
 * Keeps the IDs of the labels whose settings are expanded, adding the labels of
 * each request of the app store's `showLabelsSettings`
 *
 * @param onShow - Called on each request to bring the settings to the front
 * @returns The expanded labels IDs and a setter for them
 */
export function useExpandedLabelsIds(
  onShow: () => void,
): [string[], (expandedIds: string[]) => void] {
  const request = useAppStore((state) => state.labelsSettingsRequest);
  return useExpandedIds(request, request?.labelsId ?? null, onShow);
}

/**
 * Keeps the IDs of the points whose settings are expanded, adding the points of
 * each request of the app store's `showPointsSettings`
 *
 * @param onShow - Called on each request to bring the settings to the front
 * @returns The expanded points IDs and a setter for them
 */
export function useExpandedPointsIds(
  onShow: () => void,
): [string[], (expandedIds: string[]) => void] {
  const request = useAppStore((state) => state.pointsSettingsRequest);
  return useExpandedIds(request, request?.pointsId ?? null, onShow);
}

/**
 * Keeps the IDs of the shapes whose settings are expanded, adding the shapes of
 * each request of the app store's `showShapesSettings`
 *
 * @param onShow - Called on each request to bring the settings to the front
 * @returns The expanded shapes IDs and a setter for them
 */
export function useExpandedShapesIds(
  onShow: () => void,
): [string[], (expandedIds: string[]) => void] {
  const request = useAppStore((state) => state.shapesSettingsRequest);
  return useExpandedIds(request, request?.shapesId ?? null, onShow);
}

function useExpandedIds(
  request: object | null,
  requestedId: string | null,
  onShow: () => void,
): [string[], (expandedIds: string[]) => void] {
  const [expandedIds, setExpandedIds] = useState<string[]>([]);

  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevRequest, setPrevRequest] = useState(request);
  if (request !== prevRequest) {
    setPrevRequest(request);
    if (requestedId !== null && !expandedIds.includes(requestedId)) {
      setExpandedIds([...expandedIds, requestedId]);
    }
  }

  // stable, as a new callback is passed on every render
  const show = useLatestCallback(onShow);
  useEffect(() => {
    if (request !== null) {
      show();
    }
  }, [request, show]);

  return [expandedIds, setExpandedIds];
}
