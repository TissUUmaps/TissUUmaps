import type { ReactNode } from "react";

import type { OpenSeadragonContext } from "@tissuumaps/render";

import { OpenSeadragonContextContext } from "./OpenSeadragonContextContext";

export type OpenSeadragonContextProviderProps = {
  context: OpenSeadragonContext | null;
  children: ReactNode;
};

export function OpenSeadragonContextProvider({
  context,
  children,
}: OpenSeadragonContextProviderProps) {
  return (
    <OpenSeadragonContextContext.Provider value={context}>
      {children}
    </OpenSeadragonContextContext.Provider>
  );
}
