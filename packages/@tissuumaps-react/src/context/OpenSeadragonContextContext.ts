import { createContext, useContext } from "react";

import type { OpenSeadragonContext } from "@tissuumaps/render";

export const OpenSeadragonContextContext =
  createContext<OpenSeadragonContext | null>(null);

export function useOpenSeadragonContext(): OpenSeadragonContext | null {
  return useContext(OpenSeadragonContextContext);
}
